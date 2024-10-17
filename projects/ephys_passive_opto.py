from collections import OrderedDict

import numpy as np

import one.alf.io as alfio
from ibllib.io.extractors import ephys_fpga
from ibldsp.utils import sync_timestamps
from ibllib.plots import squares, vertical_lines
from ibllib.pipes import tasks

import ibllib.pipes.ephys_preprocessing as epp


LASER_PULSE_DURATION_SECS = .5
LASER_PROBABILITY = .8
DISPLAY = False


class EphysPassiveOptoTrials(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 1
    signature = {
        'input_files': [
            ('_iblrig_taskSettings.raw.json', 'raw_behavior_data', True),
            ('_spikeglx_sync.times.npy', 'raw_ephys_data', True),
            ('_spikeglx_sync.polarities.npy', 'raw_ephys_data', True),
            ('_spikeglx_sync.channels.npy', 'raw_ephys_data', True),
            ('*.nidq.wiring.json', 'raw_ephys_data', False),
            ('*.nidq.meta', 'raw_ephys_data', False),
        ],
        'output_files': [
            ('_ibl_trials.laserIntervals.npy', 'alf', True),
            ('_ibl_trials.laserProbability.npy', 'alf', True),
            ('_ibl_trials.intervals.npy', 'alf', True),
            ('_ibl_wheel.timestamps.npy', 'alf', True),
            ('_ibl_wheel.position.npy', 'alf', True),
            ('_ibl_wheelMoves.intervals.npy', 'alf', True),
            ('_ibl_wheelMoves.peakAmplitude.npy', 'alf', True),
        ]
    }

    def _run(self):
        sync, sync_map = ephys_fpga.get_main_probe_sync(self.session_path)
        bpod = ephys_fpga.get_sync_fronts(sync, sync_map['bpod'])
        laser_ttl = ephys_fpga.get_sync_fronts(sync, sync_map['laser_ttl'])
        t_bpod = bpod['times'][bpod['polarities'] == 1]
        t_laser = laser_ttl['times'][laser_ttl['polarities'] == 1]
        _, _, ibpod, ilaser = sync_timestamps(t_bpod, t_laser, return_indices=True)

        if DISPLAY:
            for ch in np.arange(3):
                ch0 = ephys_fpga.get_sync_fronts(sync, 16 + ch)
                squares(ch0['times'], ch0['polarities'], yrange=[-.5 + ch, .5 + ch])
            vertical_lines(t_bpod[ibpod], ymax=4)

        trial_starts = t_bpod
        trial_starts[ibpod] = t_laser[ilaser]
        ntrials = trial_starts.size

        # create the trials dictionary
        trials = {}
        trials['laserIntervals'] = np.zeros((ntrials, 2)) * np.nan
        trials['laserIntervals'][ibpod, 0] = t_laser[ilaser]
        trials['laserIntervals'][ibpod, 1] = t_laser[ilaser] + LASER_PULSE_DURATION_SECS
        trials['intervals'] = np.zeros((ntrials, 2)) * np.nan
        trials['intervals'][:, 0] = trial_starts
        trials['intervals'][:, 1] = np.r_[trial_starts[1:], np.nan]
        trials['laserProbability'] = trial_starts * 0 + LASER_PROBABILITY

        # creates the wheel object
        wheel, moves = ephys_fpga.get_wheel_positions(sync=sync, chmap=sync_map)

        # save objects
        alf_path = self.session_path.joinpath('alf')
        alf_path.mkdir(parents=True, exist_ok=True)
        out_files = []
        out_files += alfio.save_object_npy(alf_path, object='trials', namespace='ibl', dico=trials)
        out_files += alfio.save_object_npy(alf_path, object='wheel', namespace='ibl', dico=wheel)
        out_files += alfio.save_object_npy(alf_path, object='wheelMoves', namespace='ibl', dico=moves)
        return out_files


class EphysPassiveOptoPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(EphysPassiveOptoPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks["EphysRegisterRaw"] = epp.EphysRegisterRaw(self.session_path)
        tasks["EphysPulses"] = epp.EphysPulses(self.session_path)
        tasks["EphysRawQC"] = epp.RawEphysQC(self.session_path)
        tasks["EphysAudio"] = epp.EphysAudio(self.session_path)
        tasks["EphysMtscomp"] = epp.EphysMtscomp(self.session_path)
        tasks['EphysVideoCompress'] = epp.EphysVideoCompress(self.session_path)
        # level 1
        tasks["SpikeSorting"] = epp.SpikeSorting(
            self.session_path, parents=[tasks["EphysMtscomp"], tasks["EphysPulses"]])
        tasks["EphysPassiveOptoTrials"] = EphysPassiveOptoTrials(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = epp.EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysPassiveOptoTrials"]])
        tasks["EphysCellsQc"] = epp.EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = epp.EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        self.tasks = tasks


__pipeline__ = EphysPassiveOptoPipeline

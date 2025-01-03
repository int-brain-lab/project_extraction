import logging

import numpy as np
import pandas as pd
import one.alf.io as alfio
import ibldsp.utils
from iblutil.spacer import Spacer

from ibllib.pipes.base_tasks import BehaviourTask
from ibllib.exceptions import SyncBpodFpgaException
from ibllib.io.extractors.ephys_fpga import get_protocol_period, get_sync_fronts
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap
from ibllib.io.extractors.mesoscope import plot_timeline

_logger = logging.getLogger('ibllib').getChild(__name__)


class PassiveVideoTimeline(BehaviourTask):
    """Extraction task for _sp_passiveVideo protocol."""
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {}
        signature['input_files'] = [
            ('_sp_taskData.raw.*', self.collection, True),  # TODO Create dataset type?
            ('_iblrig_taskSettings.raw.*', self.collection, True),
            (f'_{self.sync_namespace}_DAQdata.raw.npy', self.sync_collection, True),
            (f'_{self.sync_namespace}_DAQdata.timestamps.npy', self.sync_collection, True),
            (f'_{self.sync_namespace}_DAQdata.meta.json', self.sync_collection, True),
        ]
        signature['output_files'] = [('_sp_video.times.npy', self.output_collection, True),]
        return signature

    def generate_sync_sequence(seed=1234, ns=3600, res=8):
        """Generate the sync square frame colour sequence.

        Instead of changing each frame, the video sync square flips between black and white
        in a particular sequence defined within this function (in random multiples of res).

        Parameters
        ----------
        ns : int
            Related to the length in frames of the sequence (n_frames = ns * res).
        res : int
            The minimum number of sequential frames in each colour state. The N sequential frames
            is a multiple of this number.
        seed : int, optional
            The numpy random seed integer, by default 1234

        Returns
        -------
        numpy.array
            An integer array of sync square states (one per frame) where 0 represents black and 1
            represents white.
        """
        state = np.random.get_state()
        try:
            np.random.seed(1234)
            seq = np.tile(np.random.random(ns), (res, 1)).T.flatten()
            return (seq > .5).astype(np.int8)
        finally:
            np.random.set_state(state)

    def extract_frame_times(self, save=True, frame_rate=60, display=False, **kwargs):
        """Extract the Bpod trials data and Timeline acquired signals.

        Sync requires three steps:
            1. Find protocol period using spacers
            2. Find each video repeat with Bpod out
            3. Find frame times with frame2ttl

        Parameters
        ----------
        save : bool, optional
            Whether to save the video frame times to file, by default True.
        frame_rate : int, optional
            The frame rate of the video presented, by default 60.
        display : bool, optional
            When true, plot the aligned frame times. By default False.

        Returns
        -------
        numpy.array
            The extracted frame times where N rows represent the number of frames and M columns
            represent the number of video repeats. The exact number of frames is not known and
            NaN values represent shorter video repeats.
        pathlib.Path
            The file path of the saved video times, or None if save=False.

        Raises
        ------
        ValueError
            The `protocol_number` property is None and no `tmin` or `tmax` values were passed as
            keyword arguments.
        SyncBpodFpgaException
            The synchronization of frame times was likely unsuccessful.
        """
        _, (p,), _ = self.input_files[0].find_files(self.session_path)
        # Load raw data
        proc_data = pd.read_parquet(p)
        sync_path = self.session_path / self.sync_collection
        self.timeline = alfio.load_object(sync_path, 'DAQdata', namespace='timeline')
        sync, chmap = load_timeline_sync_and_chmap(sync_path, timeline=self.timeline)

        bpod = get_sync_fronts(sync, chmap['bpod'])
        # Get the spacer times for this protocol
        if any(arg in kwargs for arg in ('tmin', 'tmax')):
            tmin, tmax = kwargs.get('tmin'), kwargs.get('tmax')
        elif self.protocol_number is None:
            raise ValueError('Protocol number not defined')
        else:
            # The spacers are TTLs generated by Bpod at the start of each protocol
            tmin, tmax = get_protocol_period(self.session_path, self.protocol_number, bpod)
            tmin += (Spacer().times[-1] + Spacer().tup + 0.05)  # exclude spacer itself

        # Remove unnecessary data from sync
        selection = np.logical_and(
            sync['times'] <= (tmax if tmax is not None else sync['times'][-1]),
            sync['times'] >= (tmin if tmin is not None else sync['times'][0]),
        )
        sync = alfio.AlfBunch({k: v[selection] for k, v in sync.items()})
        bpod = get_sync_fronts(sync, chmap['bpod'])
        _logger.debug('Protocol period from %.2fs to %.2fs (~%.0f min duration)',
                      *sync['times'][[0, -1]], np.diff(sync['times'][[0, -1]]) / 60)

        # For each period of video playback the Bpod should output voltage HIGH
        bpod_rep_starts, = np.where(bpod['polarities'] == 1)
        _logger.info('N video repeats: %i; N Bpod pulses: %i', len(proc_data), len(bpod_rep_starts))
        assert len(bpod_rep_starts) == len(proc_data)

        # These durations are longer than video actually played and will be cut down after
        durations = (proc_data['intervals_1'] - proc_data['intervals_0']).values
        max_n_frames = np.max(np.ceil(durations * frame_rate).astype(int))
        frame_times = np.full((max_n_frames, len(proc_data)), np.nan)

        sync_sequence = kwargs.get('sync_sequence', self.generate_sync_sequence())
        for i, rep in proc_data.iterrows():
            # Get the frame2ttl times for the video presentation
            idx = bpod_rep_starts[i]
            start = bpod['times'][idx]
            try:
                end = bpod['times'][idx + 1]
            except IndexError:
                _logger.warning('Final Bpod LOW missing')
                end = start + (rep['intervals_1'] - rep['intervals_0'])
            f2ttl = get_sync_fronts(sync, chmap['frame2ttl'])
            ts = f2ttl['times'][np.logical_and(f2ttl['times'] >= start, f2ttl['times'] < end)]

            # video_runtime is the video length reported by VLC.
            # As it was added later, the less accurate media player timestamps may be used if the former is not available
            duration = rep.get('video_runtime') or (rep['MediaPlayerEndReached'] - rep['MediaPlayerPlaying'])
            # Start the sync sequence times at the start of the first frame2ttl flip (ts[0]) as this makes syncing more
            # performant because the offset is small
            sequence_times = np.arange(0, duration, 1 / frame_rate)
            sequence_times += ts[0]
            # The below assertion could be caused by an incorrect frame rate or sync sequence
            assert sequence_times.size <= sync_sequence.size, 'video duration appears longer than sync sequence'
            # Keep only the part of the sequence that was shown
            x = sync_sequence[:len(sequence_times)]
            # Find change points (black <-> white indices)
            x, = np.where(np.abs(np.diff(x)))
            # Include first frame as change point
            x = np.r_[0, x]
            # Synchronize the two by aligning flip times
            DRIFT_THRESHOLD_PPM = 50
            Fs = self.timeline['meta']['daqSampleRate']
            fcn, drift = ibldsp.utils.sync_timestamps(sequence_times[x], ts, tbin=1 / Fs, linear=True)
            # Log any major drift or raise if too large
            if np.abs(drift) > DRIFT_THRESHOLD_PPM * 2 and x.size - ts.size > 100:
                raise SyncBpodFpgaException(f'sync cluster f*ck: drift = {drift:.2f}, changepoint difference = {x.size - ts.size}')
            elif drift > DRIFT_THRESHOLD_PPM:
                _logger.warning('BPOD/FPGA synchronization shows values greater than %.2f ppm',
                                DRIFT_THRESHOLD_PPM)

            # Get the frame times in timeline time
            frame_times[:len(sequence_times), i] = fcn(sequence_times)

        # Trim down to length of repeat with most frames
        frame_times = frame_times[:np.where(np.all(np.isnan(frame_times), axis=1))[0][0], :]

        if display:
            import matplotlib.pyplot as plt
            from matplotlib import colormaps
            from ibllib.plots import squares
            plot_timeline(self.timeline, channels=['bpod', 'frame2ttl'])
            _, ax = plt.subplots(2, 1, sharex=True)
            squares(f2ttl['times'], f2ttl['polarities'], ax=ax[0])
            ax[0].set_yticks((-1, 1))
            ax[0].title.set_text('frame2ttl')
            cmap = colormaps['plasma']
            for i, times in enumerate(frame_times.T):
                rgba = cmap(i / frame_times.shape[1])
                ax[1].plot(times, sync_sequence[:len(times)], c=rgba, label=f'{i}')
            ax[1].title.set_text('aligned sync square sequence')
            ax[1].set_yticks((0, 1))
            ax[1].set_yticklabels([-1, 1])
            plt.legend(markerfirst=False, title='repeat #', loc='upper right', facecolor='white')
            plt.show()

        if save:
            filename = self.session_path.joinpath(self.output_collection, '_sp_video.times.npy')
            out_files = [filename]
        else:
            out_files = []

        return {'video_times': frame_times}, out_files

    def run_qc(self, **_):
        raise NotImplementedError

    def _run(self, save=True, **kwargs):
        _, output_files = self.extract_frame_times(save=save, **kwargs)
        return output_files

"""Bpod extractor for max_optoStaticChoiceWorld task.

This is the same as advancedChoiceWorld with the addition of one dataset, `laserStimulation.intervals`; The times the
laser was on.

The pipeline task subclasses, OptoTrialsBpod and OptoTrialsNidq, aren't strictly necessary. They simply assert that the
laserStimulation datasets were indeed saved and registered by the Bpod extractor class.
"""

import numpy as np
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.bpod_trials import TrainingTrials # was BiasedTrials
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsNidq, ChoiceWorldTrialsBpod

from .nate_optoBiasedChoiceWorld import OptoTrialsBpod, OptoTrialsNidq
#class OptoTrialsBpod(ChoiceWorldTrialsBpod):
#    """Extract bpod only trials and laser stimulation data."""
#
#    @property
#    def signature(self):
#        signature = super().signature
#        signature['output_files'].append(('*laserStimulation.intervals.npy', self.output_collection, True))
#        return signature
#
#
#class OptoTrialsNidq(ChoiceWorldTrialsNidq):
#    """Extract trials and laser stimulation data aligned to NI-DAQ clock."""
#
#    @property
#    def signature(self):
#        signature = super().signature
#        signature['output_files'].append(('*laserStimulation.intervals.npy', self.output_collection, True))
#        return signature


class TrialsOpto(BaseBpodTrialsExtractor):
    var_names = TrainingTrials.var_names + ('laser_intervals',)
    save_names = TrainingTrials.save_names + ('_ibl_laserStimulation.intervals.npy',)

    def _extract(self, extractor_classes=None, **kwargs) -> dict:
        settings = self.settings.copy()
        if 'OPTO_STIM_STATES' in settings:
            # It seems older versions did not distinguish start and stop states
            settings['OPTO_TTL_STATES'] = settings['OPTO_STIM_STATES'][:1]
            settings['OPTO_STOP_STATES'] = settings['OPTO_STIM_STATES'][1:]
        assert {'OPTO_STOP_STATES', 'OPTO_TTL_STATES', 'PROBABILITY_OPTO_STIM'} <= set(settings)
        # Get all detected TTLs. These are stored for QC purposes
        self.frame2ttl, self.audio = raw.load_bpod_fronts(self.session_path, data=self.bpod_trials)
        # Extract common biased choice world datasets
        out, _ = run_extractor_classes(
            [TrainingTrials], session_path=self.session_path, bpod_trials=self.bpod_trials,
            settings=settings, save=False, task_collection=self.task_collection)

        # Extract laser dataset
        laser_intervals = []
        for trial in filter(lambda t: t['opto_stimulation'], self.bpod_trials): # only extract laser intervals for trials where opto_stimulation is True
            # TODO: we're gonna log ttl's instead of getting the timestamps for when a state started
            states = trial['behavior_data']['States timestamps']
            # Assumes one of these states per trial: takes the timestamp of the first matching state
            start = next((v[0][0] for k, v in states.items() if k in settings['OPTO_TTL_STATES']), np.nan)
            stop = next((v[0][0] for k, v in states.items() if k in settings['OPTO_STOP_STATES']), np.nan)
            laser_intervals.append((start, stop))
        out['laser_intervals'] = np.array(laser_intervals, dtype=np.float64)

        return {k: out[k] for k in self.var_names}  # Ensures all datasets present and ordered
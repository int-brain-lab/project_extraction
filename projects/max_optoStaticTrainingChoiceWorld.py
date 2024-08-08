"""Bpod extractor for max_optoStaticChoiceWorld task.

This is the same as advancedChoiceWorld with the addition of one dataset, `laserStimulation.intervals`; The times the
laser was on.

The pipeline task subclasses, OptoTrialsBpod and OptoTrialsNidq, aren't strictly necessary. They simply assert that the
laserStimulation datasets were indeed saved and registered by the Bpod extractor class.
"""

import numpy as np
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.bpod_trials import TrainingTrials, BiasedTrials # was BiasedTrials
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsNidq, ChoiceWorldTrialsBpod
from ibllib.qc.task_metrics import TaskQC as BaseTaskQC
from inspect import getmembers, ismethod

class PulsePalTrialsBpod(ChoiceWorldTrialsBpod):
    """Extract bpod only trials and pulsepal stimulation data."""
    @property
    def signature(self):
        signature = super().signature
        signature['output_files'].append(('*optoStimulation.intervals.npy', self.output_collection, True))
        return signature


# TODO: will eventually need to write the nidaq extractor

class TaskQC(BaseTaskQC):
    def _get_checks(self):
        def is_metric(x):
            return ismethod(x) and x.__name__.startswith('check_')

        checks = super()._get_checks()
        checks.update(dict(getmembers(self, is_metric)))
        return checks

    def check_opto_stim_intervals(self, data, **_):
        """
        1. Verify that the laser stimulation intervals are within the trial intervals of an opto_on trial.
        2. Verify that the laser stimulation intervals are greater than 0 and less than t_max.


        Parameters
        ----------
        data : dict
            Map of trial data with keys ('opto_intervals', 'opto_stimulation').

        Returns
        -------
        numpy.array
            An array the length of trials of metric M.
        numpy.array
            An boolean array the length of trials where True indicates the metric passed the
            criterion.
        """
        #TODO: implement QC logic here
        #metric = np.nan_to_num(data['laser_intervals'] - data['intervals'][:, 0], nan=np.inf)
        #passed = metric > 0
        #assert data['intervals'].shape[0] == len(metric) == len(passed)
        return metric, passed


class TrialsOpto(BaseBpodTrialsExtractor):
    var_names = BiasedTrials.var_names + ('opto_intervals',)
    save_names = BiasedTrials.save_names + ('_ibl_optoStimulation.intervals.npy',)

    def _extract(self, extractor_classes=None, **kwargs) -> dict:
        settings = self.settings.copy()
        assert {'OPTO_STOP_STATES', 'OPTO_TTL_STATES', 'PROBABILITY_OPTO_STIM'} <= set(settings)
        # Get all detected TTLs. These are stored for QC purposes
        self.frame2ttl, self.audio = raw.load_bpod_fronts(self.session_path, data=self.bpod_trials)
        # Extract common biased choice world datasets
        out, _ = run_extractor_classes(
            [BiasedTrials], session_path=self.session_path, bpod_trials=self.bpod_trials,
            settings=settings, save=False, task_collection=self.task_collection)

        # Extract opto dataset
        laser_intervals = []
        #for trial in filter(lambda t: t['opto_stimulation'], self.bpod_trials):
        for trial in self.bpod_trials:
            # the PulsePal TTL is wired into Bpod port 2. Hi for led on, lo for led off
            events = trial['behavior_data']['Events timestamps']
            if 'Port2In' in events and 'Port2Out' in events:
                start = events['Port2In'][0]
                stop = events['Port2Out'][0] # TODO: make this handle multiple opto events per trial
            else:
                start = np.nan
                stop = np.nan
            laser_intervals.append((start, stop))
        out['opto_intervals'] = np.array(laser_intervals, dtype=np.float64)

        return {k: out[k] for k in self.var_names}  # Ensures all datasets present and ordered

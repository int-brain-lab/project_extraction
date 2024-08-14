"""Bpod extractor for max_optoStaticTrainingChoiceWorld task.

This is the same as advancedChoiceWorld with the addition of one dataset, `optoStimulation.intervals`; The times the
led was on.
"""

import numpy as np
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.bpod_trials import TrainingTrials, BiasedTrials # was BiasedTrials
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsNidq, ChoiceWorldTrialsBpod
from ibllib.qc.task_metrics import TaskQC as BaseTaskQC
from inspect import getmembers, ismethod

class TaskQC(BaseTaskQC):
    def _get_checks(self):
        def is_metric(x):
            return ismethod(x) and x.__name__.startswith('check_')

        checks = super()._get_checks()
        checks.update(dict(getmembers(self, is_metric)))
        return checks

    def check_opto_percentage(self, data, **_):
        p_opto = self.extractor.settings['PROBABILITY_OPTO_STIM']
        is_opto_trial = ~np.isnan(data['opto_intervals'][:, 0])
        n_trials = len(is_opto_trial)
        actual_p_opto = np.sum(is_opto_trial) / n_trials
        passed = np.isclose(p_opto, actual_p_opto, rtol=0, atol=.2)
        return actual_p_opto, passed
        
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
        t_max = self.extractor.settings['MAX_LASER_TIME']
        is_opto_trial = ~np.isnan(data['opto_intervals'][:, 0])

        opto_on_length = data['opto_intervals'][:,1] - data['opto_intervals'][:,0]
        tol = .01 # seconds
        passed = (opto_on_length < t_max + tol) | ~is_opto_trial # less than t_max
        passed = passed & ((opto_on_length > 0) | ~is_opto_trial) # greater than zero
        return opto_on_length, passed

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

class PulsePalTrialsBpod(ChoiceWorldTrialsBpod):
    """Extract bpod only trials and pulsepal stimulation data."""
    @property
    def signature(self):
        signature = super().signature
        signature['output_files'].append(('*optoStimulation.intervals.npy', self.output_collection, True))
        return signature

    def run_qc(self, trials_data=None, update=True, QC=TaskQC,**kwargs):
        return super().run_qc(trials_data=trials_data, update=update, QC=QC, **kwargs)


# TODO: will eventually need to write the nidaq extractor
"""Bpod extractor for alejandro's FPLOptoChoiceWorld and FPROptoChoiceWorld task.

This is the same as biasedChoiceWorld with the addition of one dataset, `trials.laserStimulation`; The trials which the
laser was on. For th FPLOptoChoiceWorld protocol the laser was on when the stimulus was on the left hand side and for
the FPROptoChoiceWorld protocol the laser was on when the stimulus was on the right hand side of the screen

"""
import numpy as np
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.bpod_trials import BiasedTrials


class TrialsFPLROpto(BaseBpodTrialsExtractor):
    var_names = BiasedTrials.var_names + ('laser_stimulation', 'laser_intervals',)
    save_names = BiasedTrials.save_names + ('_ibl_trials.laserStimulation.npy', '_ibl_trials.laserIntervals.npy',)

    def _extract(self, extractor_classes=None, **kwargs) -> dict:

        # Extract common biased choice world datasets
        out, _ = run_extractor_classes(
            [BiasedTrials], session_path=self.session_path, bpod_trials=self.bpod_trials,
            settings=self.settings, save=False, task_collection=self.task_collection)

        # Extract laser stimulation dataset
        laser_stimulation = np.zeros_like(out['table']['contrastLeft'])
        laser_intervals = np.full((out['table']['contrastLeft'].size, 2), np.nan)
        if 'FPR' in self.settings['PYBPOD_PROTOCOL']:
            idx = ~np.isnan(out['table']['contrastRight'])
        elif 'FPL' in self.settings['PYBPOD_PROTOCOL']:
            idx = ~np.isnan(out['table']['contrastLeft'])

        laser_stimulation[idx] = 1
        # Laser stimulated at goCue for 200 ms
        laser_intervals[idx, 0] = out['table']['goCue_times'][idx]
        laser_intervals[idx, 1] = out['table']['goCue_times'][idx] + 200 / 1e3

        out['laser_stimulation'] = laser_stimulation
        out['laser_intervals'] = laser_intervals

        return {k: out[k] for k in self.var_names}  # Ensures all datasets present and ordered

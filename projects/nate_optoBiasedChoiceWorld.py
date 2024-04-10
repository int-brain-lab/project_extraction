"""Bpod extractor for nate_optoBiasedChoiceWorld task.

This is the same as biasedChoiceWorld with the addition of one dataset, `laserStimulation.intervals`; The times the
laser was on.

The pipeline task subclasses, OptoTrialsBpod and OptoTrialsNidq, aren't strictly necessary. They simply assert that the
laserStimulation datasets were indeed saved and registered by the Bpod extractor class.
"""
import yaml
import numpy as np
from packaging import version
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.bpod_trials import BiasedTrials
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsNidq, ChoiceWorldTrialsBpod


class OptoTrialsBpod(ChoiceWorldTrialsBpod):
    """Extract bpod only trials and laser stimulation data."""

    @property
    def signature(self):
        signature = super().signature
        signature['output_files'].append(('*laserStimulation.intervals.npy', self.output_collection, True))
        return signature


class OptoTrialsNidq(ChoiceWorldTrialsNidq):
    """Extract trials and laser stimulation data aligned to NI-DAQ clock."""

    @property
    def signature(self):
        signature = super().signature
        signature['output_files'].append(('*laserStimulation.intervals.npy', self.output_collection, True))
        return signature


class TrialsOpto(BaseBpodTrialsExtractor):
    var_names = BiasedTrials.var_names + ('laser_intervals',)
    save_names = BiasedTrials.save_names + ('_ibl_laserStimulation.intervals.npy',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_version = version.parse(self.settings.get('PROJECT_EXTRACTION_VERSION', '0.0.0'))
        if self.proj_version >= version.parse('0.3.0'):
            self.var_names = BaseBpodTrialsExtractor.var_names + ('laser_mplapdv',)
            self.save_names = BaseBpodTrialsExtractor.var_names + ('_ibl_laserStimulation.mlapdv.npy',)

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
            [BiasedTrials], session_path=self.session_path, bpod_trials=self.bpod_trials,
            settings=settings, save=False, task_collection=self.task_collection)

        # Extract laser dataset
        laser_intervals = []
        location_index = []
        for trial in filter(lambda t: t['opto_stimulation'], self.bpod_trials):
            location_index.append(trial.get('laser_location_idx', 0))
            states = trial['behavior_data']['States timestamps']
            # Assumes one of these states per trial: takes the timestamp of the first matching state
            start = next((v[0][0] for k, v in states.items() if k in settings['OPTO_TTL_STATES']), np.nan)
            stop = next((v[0][0] for k, v in states.items() if k in settings['OPTO_STOP_STATES']), np.nan)
            laser_intervals.append((start, stop))
        out['laser_intervals'] = np.array(laser_intervals, dtype=np.float64)

        # Extract laser coordinates
        if self.proj_version >= version.parse('0.3.0'):
            location_index = np.fromiter(filter(None, location_index), dtype=int)
            assert len(location_index) == out['laser_intervals'].shape[0]
            out['laser_mplapdv'] = np.full((out['laser_intervals'].shape[0], 3), np.NaN)
            # Load lookup table
            try:
                zapit_file = next(self.alf_path.glob('zapit_log_*.yml'))
            except StopIteration:
                raise FileNotFoundError('Failed to load zapit log file.')

            with open(zapit_file, 'r') as fp:
                zapit = yaml.safe_load(fp)
            if any(x['Type'] != 'unilateral_points' for x in (v for k, v in zapit.items() if k.startswith('stimLocations'))):
                raise NotImplementedError  # TODO verify and document
            for i in np.unique(location_index):
                location = zapit[f'stimLocations{i:02}']
                mlapdv = (location['ML'][0], location['AP'][0], 0.)  # TODO ensure len == 3
                out['laser_mplapdv'][location_index == i, :] = mlapdv

        return {k: out[k] for k in self.var_names}  # Ensures all datasets present and ordered

"""
NB:
- run on iblutil branch jsonable
- run on ibllib branch neuromodulator
"""

from iblutil.io.jsonable import load_task_jsonable
from ibllib.io.extractors.bpod_trials import BiasedTrials


class NeuromodulatorChoiceWorldTrialsBpod(BiasedTrials):

    def _extract(self, extractor_classes=None, **kwargs) -> dict:
        """
        The superclass outputs a dictionary of numpy arrays that will be
         later saved acccording to the `save_names` and `var_names` attributes
        """
        # self.settings # this is a dictionary of all task settings
        dict_outputs = super(NeuromodulatorChoiceWorldTrialsBpod, self)._extract(extractor_classes=None, **kwargs)
        file_jsonable = self.session_path.joinpath(self.task_collection, '_iblrig_taskData.raw.jsonable')
        trials_table, bpod_trials = load_task_jsonable(file_jsonable)

        # register the outputs with the extractor
        dict_outputs['rich_probability_left'] = trials_table['rich_probability_left'].values
        self.save_names = self.save_names + ('_ibl_trials.richProbabilityLeft.npy',)
        self.var_names = self.var_names + ('rich_probability_left',)

        return dict_outputs

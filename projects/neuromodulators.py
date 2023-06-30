import numpy as np
from ibllib.io.extractors.biased_trials import TrialsTableBiased


class TrialsTableNeuromodulator(TrialsTableBiased):

    def _extract(self, *args, **kwargs):
        out = super(TrialsTableNeuromodulator, self)._extract(*args, **kwargs)
        out[0]['omit_feedback'] = np.array([t['omit_feedback'] for t in self.bpod_trials])
        out[0]['exit_state'] = np.array([t['behavior_data']['States timestamps']['exit_state'][0][0] for t in self.bpod_trials])
        return out
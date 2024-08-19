"""Task QC for samuel_cuedBiasedChoiceWorld.

The task itself is identical to the biasedChoiceWorld except that there is an interactive delay
between stimOn and goCue.  Critically, the tone plays **before** the visual stimulus onset and is
therefore not strictly a 'go cue' tone and the delay is not strictly an 'interactive' one.

Mapping the correct Bpod trials extractor is done in `projects.task_extractor_map.json`. The trial
alignment to the DAQ remains unchanged. The
"""
from inspect import getmembers, ismethod

import numpy as np
from ibllib.qc.task_metrics import TaskQC as BaseTaskQC
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsBpod, ChoiceWorldTrialsTimeline


class TaskQC(BaseTaskQC):
    """Check that samuel_cuedBiasedChoiceWorld runs as intended."""
    def _get_checks(self):
        def is_metric(x):
            return ismethod(x) and x.__name__.startswith('check_')

        checks = super()._get_checks()
        checks.update(dict(getmembers(self, is_metric)))
        return checks

    def check_stimOn_goCue_delays(self, data, audio_output='harp', **_):
        """
        Verify time delay between audio cue and stimOn is ~equal to the INTERACTIVE_DELAY specified.

        The so-called goCue is expected to occur before the stimulus onset.

        Metric: M = goCue_times - stimOn_times - interactive_delay
        Criteria: 0 < M < 0.010 s
        Units: seconds [s]

        Parameters
        ----------
        data : dict
            Map of trial data with keys ('goCue_times', 'stimOn_times', 'intervals').
        audio_output : str
            Audio output device name.

        Returns
        -------
        numpy.array
            An array the length of trials of metric M.
        numpy.array
            An boolean array the length of trials where True indicates the metric passed the
            criterion.

        Notes
        -----
        For non-harp sound card the permissible delay is 0.053s. This was chosen by taking the 99.5th
        percentile of delays over 500 training sessions using the Xonar soundcard.
        """
        # Calculate the difference between stimOn and goCue times.
        # If either are NaN, the result will be Inf to ensure that it crosses the failure threshold.
        threshold = 0.01 if audio_output.lower() == 'harp' else 0.053
        delay = self.extractor.settings['INTERACTIVE_DELAY']
        metric = np.nan_to_num(data['stimOn_times'] - data['goCue_times'] - delay, nan=np.inf)
        passed = (metric < threshold) & (metric > 0)
        assert data['intervals'].shape[0] == len(metric) == len(passed)
        return metric, passed


class CuedBiasedTrialsTimeline(ChoiceWorldTrialsTimeline):
    """Behaviour task for aligning cuedBiased task to Timeline."""

    def run_qc(self, trials_data=None, update=True, QC=TaskQC, **kwargs):
        """
        Run task QC.

        Parameters
        ----------
        trials_data : dict
            The extracted trials datasets.
        update : bool
            If True, updates Alyx with the QC outcomes.
        QC : TaskQC
            The task QC class to instantiate.

        Returns
        -------
        ibllib.qc.base.QC
            The QC object.
        """
        return super().run_qc(trials_data=trials_data, update=update, QC=QC)


class CuedBiasedTrials(ChoiceWorldTrialsBpod):
    """Behaviour task for extracting Bpod-only cuedBiased task."""

    def run_qc(self, trials_data=None, update=True, QC=TaskQC, **kwargs):
        """
        Run task QC.

        Parameters
        ----------
        trials_data : dict
            The extracted trials datasets.
        update : bool
            If True, updates Alyx with the QC outcomes.
        QC : TaskQC
            The task QC class to instantiate.

        Returns
        -------
        ibllib.qc.base.QC
            The QC object.
        """
        return super().run_qc(trials_data=trials_data, update=update, QC=QC)

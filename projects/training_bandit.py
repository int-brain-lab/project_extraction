"""Bpod extractor for alejandro's _bandit_100_0_biasedChoiceWorld, _bandit_biasedChoiceWorld and
_bandit_alllaser_cued_ephysChoiceWorld task protocols

_bandit_100_0_biasedChoiceWorld and _bandit_biasedChoiceWorld
Bandit Choice World task where two Gabor patches appear on the screen. The mouse must move the wheel beyond the threshold
to the left or right to get a reward. The side that is rewarded changes in blocks defined by `_av_trials.probabilityLeft'.
Reward is not related in any way to visual stimulus. For _bandit_100_0_biasedChoiceWorld within a block the probability
of being rewarded for moving the wheel to the correct side is 100% (0% for the other side).
For _bandit_biasedChoiceWorld, the probability of being rewarded is 70%.

Additional datasets:
- '_av_trials.probabilityRewardLeft.npy' - indicates the probability of reward being given on left movement of trial
Datasets that differ from normal tasK:
- '_ibl_trials.choice' - choice is inferred from direction of wheel movement that first reaches a threshold
- '_ibl_trials.contrastLeft' - in this task reward is not related to visual stimulus. For each trial two Gabor patches
of 100% contrast are shown on the left and right of screen that move with the wheel. Therefore value of 1 for all trials
- '_ibl_trials.contrastRight' - in this task reward is not related to visual stimulus. For each trial two Gabor patches
of 100% contrast are shown on the left and right of screen that move with the wheel. Therefore value of 1 for all trials
- '_ibl_trials.probabilityLeft' - the reward is not linked to visual stimulus so all values are nan

_bandit_alllaser_cued_ephysChoiceWorld
Same as _bandit_biasedChoiceWorld above with the addition of laser stimulation. When the task is in an opto block,
defined by '_av_trials.laserProbability.npy', reward is given through laser stimulation to the VTA rather than a water
reward.

Additional datasets:
- '_av_trials.probabilityRewardLeft.npy' - indicates the probability of reward being given on left movement of trial
- '_av_trials.laserProbability.npy' - indicates the block of trials where laser stimulation is active
- '_ibl_trials.laserStimulation.npy' - indicates trials where the laser was actually stimulated, only those trials
within a laser block with correct feedback

Datasets that differ from normal tasK:
- See _bandit_100_0_biasedChoiceWorld and _bandit_biasedChoiceWorld
In addition
- '_ibl_trials.rewardVolume' - where reward is given by the laser instead of water the reward volume is set to zero
"""

import logging
import numpy as np
from one.alf.io import AlfBunch

import ibllib.io.extractors.training_trials as tt
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes

_logger = logging.getLogger('ibllib')


class TrialsLaserBandit(BaseBpodTrialsExtractor):

    var_names = tt.TrainingTrials.var_names + ('probabilityRewardLeft', 'laserStimulation', 'laserProbability')
    save_names = tt.TrainingTrials.save_names + ('_av_trials.probabilityRewardLeft.npy',
                                                 '_ibl_trials.laserStimulation.npy', '_av_trials.laserProbability.npy')

    def _extract(self, extractor_classes=None, **kwargs) -> dict:

        base = [BanditRepNum, tt.GoCueTriggerTimes, tt.StimOnTriggerTimes, tt.ItiInTimes, tt.StimOffTriggerTimes,
                tt.StimFreezeTriggerTimes, tt.ErrorCueTriggerTimes, LaserBanditTrialsTable, tt.PhasePosQuiescence,
                tt.PauseDuration, ProbabilityRewardLeft,  BanditLaserStimulation, BanditLaserProbability]

        # Extract common biased choice world datasets
        out, _ = run_extractor_classes(
            base, session_path=self.session_path, bpod_trials=self.bpod_trials,
            settings=self.settings, save=False, task_collection=self.task_collection)

        return {k: out[k] for k in self.var_names}




class TrialsBandit(BaseBpodTrialsExtractor):
    var_names = tt.TrainingTrials.var_names + ('probabilityRewardLeft',)
    save_names = tt.TrainingTrials.save_names + ('_av_trials.probabilityRewardLeft.npy',)

    def _extract(self, extractor_classes=None, **kwargs) -> dict:

        base = [BanditRepNum, tt.GoCueTriggerTimes, tt.StimOnTriggerTimes, tt.ItiInTimes, tt.StimOffTriggerTimes,
                tt.StimFreezeTriggerTimes, tt.ErrorCueTriggerTimes, BanditTrialsTable, tt.PhasePosQuiescence,
                tt.PauseDuration, ProbabilityRewardLeft]

        # Extract common biased choice world datasets
        out, _ = run_extractor_classes(
            base, session_path=self.session_path, bpod_trials=self.bpod_trials,
            settings=self.settings, save=False, task_collection=self.task_collection)


        return {k: out[k] for k in self.var_names}



class BanditTrialsTable(BaseBpodTrialsExtractor):
    """
    Extracts the following into a table from Bpod raw data:
        intervals, goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight,
        feedback_times, feedbackType, rewardVolume, probabilityLeft, firstMovement_times
    Additionally extracts the following wheel data:
        wheel_timestamps, wheel_position, wheel_moves_intervals, wheel_moves_peak_amplitude
    """

    var_names = tt.TrialsTable.var_names
    save_names = tt.TrialsTable.save_names


    def _extract(self, extractor_classes=None, **kwargs):
        base = [tt.Intervals, tt.GoCueTimes, tt.ResponseTimes, BanditChoice, tt.StimOnOffFreezeTimes, BanditContrastLR,
                tt.FeedbackTimes, tt.FeedbackType, tt.RewardVolume, BanditProbabilityLeft, tt.Wheel]

        out, _ = run_extractor_classes(
            base, session_path=self.session_path, bpod_trials=self.bpod_trials, settings=self.settings, save=False,
            task_collection=self.task_collection)

        table = AlfBunch({k: v for k, v in out.items() if k not in self.var_names})
        assert len(table.keys()) == 12

        return table.to_df(), *(out.pop(x) for x in self.var_names if x != 'table')



class LaserBanditTrialsTable(BaseBpodTrialsExtractor):
    """
    Extracts the following into a table from Bpod raw data:
        intervals, goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight,
        feedback_times, feedbackType, rewardVolume, probabilityLeft, firstMovement_times
    Additionally extracts the following wheel data:
        wheel_timestamps, wheel_position, wheel_moves_intervals, wheel_moves_peak_amplitude
    """

    var_names = tt.TrialsTable.var_names
    save_names = tt.TrialsTable.save_names

    def _extract(self, extractor_classes=None, **kwargs):
        base = [tt.Intervals, tt.GoCueTimes, tt.ResponseTimes, BanditChoice, tt.StimOnOffFreezeTimes, BanditContrastLR,
                tt.FeedbackTimes, tt.FeedbackType, BanditRewardVolume, BanditProbabilityLeft, tt.Wheel]

        out, _ = run_extractor_classes(
            base, session_path=self.session_path, bpod_trials=self.bpod_trials, settings=self.settings, save=False,
            task_collection=self.task_collection)

        table = AlfBunch({k: v for k, v in out.items() if k not in self.var_names})
        assert len(table.keys()) == 12

        return table.to_df(), *(out.pop(x) for x in self.var_names if x != 'table')


class BanditRepNum(tt.RepNum):

    def _extract(self):
        repnum = np.ones(len(self.bpod_trials))

        return repnum


class BanditChoice(tt.Choice):
    """
    Get the subject's choice in every trial.
    **Optional:** saves _ibl_trials.choice.npy to alf folder.

    Extracts choice from side on which first rotary encoder threhsold was detected
    -1 is a CCW turn (towards the left) this is stored in 'RotaryEncoder1_1'
    +1 is a CW turn (towards the right) this is stored in 'RotaryEncoder1_2'
    0 is a no_go trial

    """

    def _extract(self):
        return np.array([self._extract_choice(t) for t in self.bpod_trials]).astype(int)

    def _extract_choice(self, data):
        if (('RotaryEncoder1_2' in data['behavior_data']['Events timestamps']) &
                ('RotaryEncoder1_1' in data['behavior_data']['Events timestamps'])):
            # When both thresholds are passed, take the first crossed threshold to be the choice direction
            choice = -1 if (data['behavior_data']['Events timestamps']['RotaryEncoder1_2'][0] >
                            data['behavior_data']['Events timestamps']['RotaryEncoder1_1'][0]) else 1
        elif 'RotaryEncoder1_1' in data['behavior_data']['Events timestamps']:  # this should be CCW
            choice = -1
        elif 'RotaryEncoder1_2' in data['behavior_data']['Events timestamps']:  # this should be CW
            choice = 1
        else:
            choice = 0

        return choice


class ProbabilityRewardLeft(BaseBpodTrialsExtractor):
    """
    Probability of reward being given on left movement of wheel
    """
    save_names = '_av_trials.probabilityRewardLeft.npy'
    var_names = 'probabilityRewardLeft'

    def _extract(self, **kwargs):
        return np.array([t['stim_probability_left'] for t in self.bpod_trials])


class BanditProbabilityLeft(tt.ProbabilityLeft):
    """
    No visual stimulus related events so probability left is nan for all trials
    """

    def _extract(self, **kwargs):
        return np.ones(len(self.bpod_trials)) * np.nan


class BanditContrastLR(tt.ContrastLR):
    """
    Get left and right contrasts from raw datafile. Optionally, saves
    _ibl_trials.contrastLeft.npy and _ibl_trials.contrastRight.npy to alf folder.

    Uses signed_contrast to create left and right contrast vectors.
    """

    def _extract(self):
        contrastLeft = np.array([t['contrast'] for t in self.bpod_trials])
        contrastRight = np.array([t['contrast'] for t in self.bpod_trials])

        return contrastLeft, contrastRight


class BanditRewardVolume(tt.RewardVolume):
    """
    Load reward volume delivered for each trial. For trials where the reward was given by laser stimulation
    rather than water stimulation set reward volume to 0
    """

    def _extract(self):
        rewards = super(BanditRewardVolume, self)._extract()
        laser = np.array([t['opto_block'] for t in self.bpod_trials]).astype(bool)
        rewards[laser] = 0

        return rewards


class BanditLaserProbability(BaseBpodTrialsExtractor):
    """
    Get the block of trials where laser stimulation is active
    """
    save_names = '_av_trials.laserProbability.npy'
    var_names = 'laserProbability'

    def _extract(self):
        laser = np.array([t['opto_block'] for t in self.bpod_trials]).astype(int)
        return laser


class BanditLaserStimulation(BaseBpodTrialsExtractor):
    """
    Get the trials where laser reward stimulation was given. Laser stimulation given when task was in laser block and feedback
    is correct
    """

    save_names = '_ibl_trials.laserStimulation.npy'
    var_names = 'laserStimulation'

    def _extract(self):
        reward = np.array([~np.isnan(t['behavior_data']['States timestamps']['reward'][0][0]) for t in
                           self.bpod_trials]).astype(bool)
        laser = np.array([t['opto_block'] for t in self.bpod_trials]).astype(int)
        laser[~reward] = 0
        return laser


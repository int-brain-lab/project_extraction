import numpy as np

import one.alf.io as alfio
from one.api import ONE
from ci.tests import base
import unittest


@unittest.skip('old pipeline no longer supported')
class TestTrainingBanditTrials(base.IntegrationTest):

    def setUp(self) -> None:

        self.one_offline = ONE(mode='local')
        self.session_path = self.data_path.joinpath("personal_projects/training_bandit/dop_46/2022-02-21/001/")

    def test_training_bandit(self):
        """
        Alejandro training bandit task
        :return:
        """
        from projects.training_bandit import extract_all as training_extract_all
        trials, _, _ = training_extract_all(self.session_path, save=False)
        trials_orig = alfio.load_object(self.session_path.joinpath('alf'), object='trials')

        assert np.array_equal(trials['probabilityRewardLeft'], trials_orig['probabilityRewardLeft'])
        assert np.array_equal(trials['table']['goCue_times'], trials_orig['goCue_times'])
        assert np.array_equal(trials['table']['choice'], trials_orig['choice'])
        assert np.array_equal(trials['table']['contrastLeft'], trials_orig['contrastLeft'])
        assert np.array_equal(trials['table']['contrastRight'], trials_orig['contrastRight'])


@unittest.skip('old pipeline no longer supported')
class TestEphysBanditTrials(base.IntegrationTest):

    @unittest.skip("Skip FTP upload test")
    def setUp(self) -> None:
        self.one_offline = ONE(mode='local')
        self.session_path = self.data_path.joinpath("personal_projects/ephys_bandit/dop_38/2021-12-13/002/")

    def test_ephys_bandit(self):
        """
        Alejandro ephys bandit task
        :return:
        """
        from projects.ephys_bandit import extract_all as ephys_extract_all
        trials, _ = ephys_extract_all(self.session_path, save=False)
        trials_orig = alfio.load_object(self.session_path.joinpath('alf'), object='trials')

        assert np.array_equal(trials['probabilityRewardLeft'], trials_orig['probabilityRewardLeft'])
        np.testing.assert_equal(trials['table']['goCue_times'], trials_orig['goCue_times'])
        assert np.array_equal(trials['table']['choice'], trials_orig['choice'])
        assert np.array_equal(trials['table']['contrastLeft'], trials_orig['contrastLeft'])
        assert np.array_equal(trials['table']['contrastRight'], trials_orig['contrastRight'])
        assert np.array_equal(trials['laserProbability'], trials_orig['laserProbability'])
        assert np.array_equal(trials['laserStimulation'], trials_orig['laserStimulation'])

        correct_no_water = np.bitwise_and(trials['table']['feedbackType'] == 1, trials['laserProbability'] == 1)
        correct_water = np.bitwise_and(trials['table']['feedbackType'] == 1, trials['laserProbability'] == 0)
        laser_on = np.bitwise_and(trials['table']['feedbackType'] == 1, trials['laserProbability'] == 1)
        assert np.all(trials['table']['rewardVolume'][correct_no_water] == 0)
        assert np.all(trials['table']['rewardVolume'][correct_water] != 0)
        assert np.all(trials['laserStimulation'][laser_on] == 1)
        assert np.all(trials['laserStimulation'][~laser_on] != 1)

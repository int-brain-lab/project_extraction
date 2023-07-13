import shutil

import numpy as np

import one.alf.io as alfio
from one.api import ONE
from ci.tests import base

from projects.ephys_passive_opto import EphysPassiveOptoTrials


class TestEphysPassiveOptoTrials(base.IntegrationTest):

    def setUp(self) -> None:
        self.one_offline = ONE(mode='local')
        self.session_path = self.data_path.joinpath("personal_projects/ephys_passive_opto/KS056/2021-07-18/001")

    def test_ephys_passive_opto(self):
        """
        Karolina's task
        :return:
        """
        session_path = self.session_path
        task = EphysPassiveOptoTrials(session_path=session_path, one=self.one_offline)
        task.run()

        task.assert_expected_outputs()

        trials = alfio.load_object(session_path.joinpath('alf'), object='trials')
        assert alfio.check_dimensions(trials) == 0

        assert np.sum(np.isnan(trials['laserIntervals'])) == 110
        assert set(trials.keys()) == set(['laserIntervals', 'intervals', 'laserProbability'])
        assert np.all(trials['laserProbability'] == .8)

    def tearDown(self) -> None:
        shutil.rmtree(self.session_path.joinpath('alf'), ignore_errors=True)


# one = ONE()
# eid = '51751156-6b97-48f7-8482-b695f8cab732'
# session_path = one.eid2path(eid)
# task = EphysPassiveOptoTrials(session_path=session_path, location='remote', one=one)
# task.run()

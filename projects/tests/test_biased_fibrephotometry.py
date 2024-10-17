import unittest
import shutil

import numpy as np
from projects import biased_fibrephotometry
import one.alf.io as alfio
from one.api import ONE


from ci.tests import base


@unittest.skip('old pipeline no longer supported')
class TestBiasedPhotometry(base.IntegrationTest):

    def setUp(self) -> None:
        """
        For now we'll have only the raw data, extract the ALF data then delete the alf collection
        after checking the output.  In the future we may want to symlink the alf folder and compare
        the two outputs.
        """
        self.session_path = self.data_path.joinpath(
            'personal_projects', 'biased_photometry', 'fip_16', '2021-04-21', '001')
        self.addCleanup(shutil.rmtree, self.session_path / 'alf', ignore_errors=True)
        # Below I started to symlink the files to a temp directory for extraction without overwriting the alf folder
        # self.tempdir = tempfile.TemporaryDirectory()
        # self.addCleanup(self.tempdir.cleanup)
        # self.tmp_session_path = Path(self.tempdir.name).joinpath('fip_16', '2021-04-21', '001')
        # to_copy = filter(lambda x: not x.name.startswith('photometry'), self.session_path.rglob('*.*'))
        # for file in to_copy:
        #     link = alfiles.get_alf_path(self.tmp_session_path)
        #     file.parent.mkdir(parents=True, exist_ok=True)
        #     link.symlink_to(ff)

    def test_extraction(self):
        import ibllib.pipes.training_preprocessing as tpp

        # Extract the trials (required for photometry extraction)
        self.assertEqual(
            0, tpp.TrainingTrials(self.session_path, one=ONE(mode='local')).run()
        )

        # Extract the photometry signal and timestamps
        task = biased_fibrephotometry.FibrePhotometryPreprocess(
            self.session_path, one=ONE(mode='local')
        )
        status = task.run()
        self.assertEqual(0, status)

        photometry = alfio.load_object(self.session_path / 'alf/photometry', 'photometry')['signal']
        self.assertEqual(1, len(list(self.session_path.joinpath('alf/photometry').glob('photometry*'))))
        np.testing.assert_array_equal(
            photometry['wavelength'][:5], np.array([0, 470, 415, 0, 0], dtype=int)
        )
        self.assertEqual((361433, 8), photometry.shape)
        expected = np.array([[0.00392157, 0.00392157, 0.00392157],
                             [0.00574833, 0.0052378, 0.00478918],
                             [0.00395847, 0.00416557, 0.00401466]])
        np.testing.assert_array_almost_equal(expected, photometry[['Region0G', 'Region1G', 'Region2G']][:3])

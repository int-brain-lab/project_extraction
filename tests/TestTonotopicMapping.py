import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from iblrig_custom_tasks.samuel_tonotopicMapping import task


class TestCreateDataframe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary directory and copy the fixture file to it."""
        cls.fixture = Path(__file__).parent / 'TonotopicMapping.jsonable'
        assert cls.fixture.exists(), 'jsonable fixture not found'
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.jsonable = Path(cls.temp_dir.name) / '_iblrig_taskData.raw.jsonable'
        shutil.copy(cls.fixture, cls.jsonable)
        cls.dataframe = task.create_dataframe(cls.jsonable)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary directory after the tests."""
        cls.temp_dir.cleanup()

    def test_file_not_found(self):
        """Test that a non-existent file raises a ValidationError."""
        with self.assertRaises(ValidationError):
            task.create_dataframe('non_existent_file.jsonable')

    def test_wrong_file_name(self):
        """Test that a file with the wrong name raises a ValueError."""
        with self.assertRaises(ValueError):
            task.create_dataframe(self.fixture)

    def test_return_type(self):
        """Test that the function returns a pandas DataFrame."""
        self.assertIsInstance(self.dataframe, pd.DataFrame)

    def test_columns(self):
        """Test that the DataFrame has the correct columns."""
        self.assertListEqual(list(self.dataframe.columns), ['Trial', 'Stimulus', 'Value', 'Frequency', 'Attenuation'])

    def test_index(self):
        """Test that the DataFrame has a TimedeltaIndex."""
        self.assertIsInstance(self.dataframe.index, pd.TimedeltaIndex)
        self.assertEqual(self.dataframe.index.dtype, np.dtype('timedelta64[us]'))

    def test_no_nans(self):
        """Test that the DataFrame does not contain NaN values."""
        self.assertTrue(all(~self.dataframe.isna().any()))

    def test_duration(self):
        """Test that the duration of the fixture pulses is close to 150ms."""
        t0 = self.dataframe[self.dataframe['Value'] == 0].index
        t1 = self.dataframe[self.dataframe['Value'] == 1].index
        durations = pd.arrays.TimedeltaArray(t0 - t1)
        np.testing.assert_allclose(durations.microseconds / 1e3, 150, atol=0.15)


if __name__ == '__main__':
    unittest.main()

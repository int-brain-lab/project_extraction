import unittest
from unittest.mock import Mock
from iblrig_custom_tasks._sp_passiveVideo.task import Session, Player
from iblrig.test.base import TaskArgsMixin


class TestPassiveVideo(TaskArgsMixin, unittest.TestCase):

    def setUp(self):
        self.get_task_kwargs()

    def test_next_trial(self):
        self.assertRaises(NotImplementedError, Session, **self.task_kwargs)
        self.task_kwargs['hardware_settings']['MAIN_SYNC'] = False
        task = Session(log_level='DEBUG', **self.task_kwargs)
        task.video = Mock(auto_spec=Player)
        task.task_params.VIDEO = r'C:\Users\Work\Downloads\ONE\perlin-xyscale2-tscale50-comb08-5min.mp4'
        task.task_params.VIDEO = r'C:\Users\Work\Downloads\SampleVideo_1280x720_1mb.mp4'
        task.next_trial()
        task.video.play.assert_called_once_with(task.task_params.VIDEO)
        task.video.replay.assert_not_called()
        task.video.reset_mock()
        task.next_trial()
        task.video.replay.assert_called_once()
        # task.bpod = MagicMock()
        # with patch.object(task, 'start_mixin_bpod'):
        #     task.run()


if __name__ == '__main__':
    unittest.main()

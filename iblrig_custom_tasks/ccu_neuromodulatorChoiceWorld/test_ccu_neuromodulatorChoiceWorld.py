"""
Here we test for the state machine code and the task to be importable by the GUI
"""
from iblrig_custom_tasks.ccu_neuromodulatorChoiceWorld.task import Session as NeuromodulatorChoiceWorldSession

import datetime
import time

import numpy as np
import pandas as pd

from iblrig.raw_data_loaders import load_task_jsonable
from iblrig.test.base import PATH_FIXTURES, BaseTestCases, IntegrationFullRuns
from iblrig.test.tasks.test_biased_choice_world_family import get_fixtures
from iblrig_tasks._iblrig_tasks_biasedChoiceWorld.task import Session as BiasedChoiceWorldSession
from iblrig_tasks._iblrig_tasks_ephysChoiceWorld.task import Session as EphysChoiceWorldSession
from iblrig_tasks._iblrig_tasks_ImagingChoiceWorld.task import Session as ImagingChoiceWorldSession
from iblrig_tasks._iblrig_tasks_neuroModulatorChoiceWorld.task import Session as NeuroModulatorChoiceWorldSession


class TestCCU(BaseTestCases.CommonTestInstantiateTask):
    def setUp(self) -> None:
        self.get_task_kwargs()
        self.task = NeuromodulatorChoiceWorldSession(**self.task_kwargs)
        np.random.seed(12345)

    def test_task(self, reward_set: np.ndarray | None = None):
        if reward_set is None:
            reward_set = np.array([0, 1.5])
        task = self.task
        task.create_session()
        trial_fixtures = get_fixtures()
        nt = 500
        t = np.zeros(nt)
        for i in np.arange(nt):
            t[i] = time.time()
            task.next_trial()
            # pc = task.psychometric_curve()
            trial_type = np.random.choice(['correct', 'error', 'no_go'], p=[0.9, 0.05, 0.05])
            task.trial_completed(trial_fixtures[trial_type])
            if trial_type == 'correct':
                self.assertTrue(task.trials_table['trial_correct'][task.trial_num])
            else:
                # fixme here we should init the trials table with nan
                self.assertFalse(task.trials_table['trial_correct'][task.trial_num])

            if i == 245:
                task.show_trial_log()
            assert not np.isnan(task.reward_time)
        # test the trial table results
        task.trials_table = task.trials_table[: task.trial_num + 1]
        np.testing.assert_array_equal(task.trials_table['trial_num'].values, np.arange(task.trial_num + 1))
        # makes sure the water reward counts check out
        assert task.trials_table['reward_amount'].sum() == task.session_info.TOTAL_WATER_DELIVERED
        assert np.sum(task.trials_table['reward_amount'] == 0) == task.trial_num + 1 - task.session_info.NTRIALS_CORRECT
        assert np.all(~np.isnan(task.trials_table['reward_valve_time']))


        df_template = task.get_session_template(task.task_params['SESSION_TEMPLATE_ID'])

        # Test the blocks task logic
        df_blocks = task.trials_table.groupby(['reward_probability_left', 'stim_probability_left']).agg(
            count=pd.NamedAgg(column='stim_angle', aggfunc='count'),
            n_stim_probability_left=pd.NamedAgg(column='stim_probability_left', aggfunc='nunique'),
            stim_probability_left=pd.NamedAgg(column='stim_probability_left', aggfunc='first'),
            position=pd.NamedAgg(column='position', aggfunc=lambda x: 1 - (np.mean(np.sign(x)) + 1) / 2),
        )
        # todo check the common columns of template df with the recovered trials table
        # todo modify / adapt the logic tests down here to match the new task requirements
        # test that the first block is 90 trials
        assert df_blocks['count'].values[0] == 90
        # make all first block trials were reset to 0
        assert np.all(df_blocks['first_trial'] == 0)
        # test that the first block has 50/50 probability
        assert df_blocks['stim_probability_left'].values[0] == 0.5
        # make sure that all subsequent blocks alternate between 0.2 and 0.8 left probability
        assert np.all(np.isclose(np.abs(np.diff(df_blocks['stim_probability_left'].values[1:])), 0.6))
        # assert the the trial outcomes are within 0.3 of the generating probability
        np.testing.assert_array_less(np.abs(df_blocks['position'] - df_blocks['stim_probability_left']), 0.4)
        np.testing.assert_array_equal(np.unique(task.trials_table['reward_amount']), reward_set)
        # assert quiescent period
        self.check_quiescent_period()

    def check_quiescent_period(self):
        """
        Check the quiescence period

        From Appendix 2:
            At the beginning of each trial, the mouse must not move the wheel for a fixed, “quiescent” period for the
            trial to continue. The duration of this period is between 400 and 700 milliseconds (computed as 200 ms +
            a duration drawn from an exponential distribution with a mean of 350 milliseconds, min and max of 200-500
            milliseconds).

        Overload this method for a change in quiescent period
        """
        self.assertTrue(np.all(self.task.trials_table['quiescent_period'] >= 0.4))
        self.assertTrue(np.all(self.task.trials_table['quiescent_period'] <= 0.7))
        self.assertAlmostEqual(self.task.trials_table['quiescent_period'].mean() - 0.2, 0.35, delta=0.05)

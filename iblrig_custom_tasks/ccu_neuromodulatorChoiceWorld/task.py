import logging
from pathlib import Path

import pandas as pd

from iblrig.base_choice_world import BiasedChoiceWorldSession, ActiveChoiceWorldSession
from iblrig.hardware import SOFTCODE
from iblrig.misc import get_task_arguments
from pybpodapi.protocol import StateMachine

log = logging.getLogger(__name__)

# TODO: add function annotation/ type hinting
# TODO: come up with a more descriptive protocol name


class Session(ActiveChoiceWorldSession):
    protocol_name = 'ccu_rewardbias'

    def __init__(self, *args, session_template_id=0, **kwargs):
        df_template = self.get_session_template(session_template_id)
        super().__init__(*args, **kwargs)
        self.task_params.SESSION_TEMPLATE_ID = session_template_id
        # TODO: need to check that same session is not repeated for same mouse?
        for c in df_template.columns:
            self.trials_table[c] = df_template[c]
        # TODO populate the omit_feedback in the parquet construction
        # TODO: label the blocks in the trials table at table generation

    @staticmethod
    def get_session_template(session_template_id):
        """
        Returns a pre-generated trials table dataframe.

        Parameters
        ----------
        session_template_id : int
            id of template session trial data to load

        Returns
        -------
        trials_table : pandas.DataFrame
            table containing trial-wise information for the template session
        """
        fpath = Path(__file__).parent.joinpath('neuromodcw_session_templates.pqt')
        df_sessions = pd.read_parquet(fpath)
        trials_table = df_sessions[df_sessions['session_id'] == session_template_id]
        return trials_table

    @staticmethod
    def extra_parser():
        """:return: argparse.parser()"""
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--session_template_id',
            option_strings=['--session_template_id'],
            dest='session_template_id',
            default=0,
            type=int,
            help='pre-generated session template id (zero-based)',
        )
        return parser

    def get_state_machine_trial(self, i):
        # TODO: check Nate's custom task to see how to edit states instead of
        # re-defining them all
        sma = StateMachine(self.bpod)

        if i == 0:  # First trial exception start camera
            session_delay_start = self.task_params.get('SESSION_DELAY_START', 0)
            log.info('First trial initializing, will move to next trial only if:')
            log.info('1. camera is detected')
            log.info(f'2. {session_delay_start} sec have elapsed')
            sma.add_state(
                state_name='trial_start',
                state_timer=0,
                state_change_conditions={'Port1In': 'delay_initiation'},
                output_actions=[('SoftCode', SOFTCODE.TRIGGER_CAMERA), ('BNC1', 255)],
            )  # start camera
            sma.add_state(
                state_name='delay_initiation',
                state_timer=session_delay_start,
                output_actions=[],
                state_change_conditions={'Tup': 'reset_rotary_encoder'},
            )
        else:
            sma.add_state(
                state_name='trial_start',
                state_timer=0,  # ~100µs hardware irreducible delay
                state_change_conditions={'Tup': 'reset_rotary_encoder'},
                output_actions=[self.bpod.actions.stop_sound, ('BNC1', 255)],
            )  # stop all sounds

        sma.add_state(
            state_name='reset_rotary_encoder',
            state_timer=0,
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={'Tup': 'quiescent_period'},
        )

        sma.add_state(  # '>back' | '>reset_timer'
            state_name='quiescent_period',
            state_timer=self.quiescent_period,
            output_actions=[],
            state_change_conditions={
                'Tup': 'stim_on',
                self.movement_left: 'reset_rotary_encoder',
                self.movement_right: 'reset_rotary_encoder',
            },
        )

        sma.add_state(
            state_name='stim_on',
            state_timer=0.1,
            output_actions=[self.bpod.actions.bonsai_show_stim],
            state_change_conditions={'Tup': 'interactive_delay', 'BNC1High': 'interactive_delay', 'BNC1Low': 'interactive_delay'},
        )

        sma.add_state(
            state_name='interactive_delay',
            state_timer=self.task_params.INTERACTIVE_DELAY,
            output_actions=[],
            state_change_conditions={'Tup': 'play_tone'},
        )

        sma.add_state(
            state_name='play_tone',
            state_timer=0.1,
            output_actions=[self.bpod.actions.play_tone],
            state_change_conditions={'Tup': 'reset2_rotary_encoder', 'BNC2High': 'reset2_rotary_encoder'},
        )

        sma.add_state(
            state_name='reset2_rotary_encoder',
            state_timer=0.05,
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={'Tup': 'closed_loop'},
        )

        if self.omit_feedback:
            for state_name in ['omit_error', 'omit_correct', 'omit_no_go']:
                sma.add_state(
                    state_name=state_name,
                    state_timer=(
                                        self.task_params.FEEDBACK_NOGO_DELAY_SECS
                                        + self.task_params.FEEDBACK_ERROR_DELAY_SECS
                                        + self.task_params.FEEDBACK_CORRECT_DELAY_SECS
                                )
                                / 3,
                    # TODO: check if we want to freeze stim here
                    output_actions=[self.bpod.actions.bonsai_freeze_stim],
                    state_change_conditions={'Tup': 'hide_stim'},
                )
            # same as normal closed loop state, but transistions to states that
            # allow to skip feedback states
            sma.add_state(
                state_name='closed_loop',
                state_timer=self.task_params.RESPONSE_WINDOW,
                output_actions=[self.bpod.actions.bonsai_closed_loop],
                state_change_conditions={
                    'Tup': 'omit_no_go',
                    self.event_error: 'omit_error',
                    self.event_reward: 'omit_correct'},
            )

        else:
            sma.add_state(
                state_name='closed_loop',
                state_timer=self.task_params.RESPONSE_WINDOW,
                output_actions=[self.bpod.actions.bonsai_closed_loop],
                state_change_conditions={
                    'Tup': 'no_go',
                    self.event_error: 'freeze_error',
                    self.event_reward: 'freeze_reward',
                },
            )

        sma.add_state(
            state_name='no_go',
            state_timer=self.task_params.FEEDBACK_NOGO_DELAY_SECS,
            output_actions=[self.bpod.actions.bonsai_hide_stim, self.bpod.actions.play_noise],
            state_change_conditions={'Tup': 'exit_state'},
        )

        sma.add_state(
            state_name='freeze_error',
            state_timer=0,
            output_actions=[self.bpod.actions.bonsai_freeze_stim],
            state_change_conditions={'Tup': 'error'},
        )

        sma.add_state(
            state_name='error',
            state_timer=self.task_params.FEEDBACK_ERROR_DELAY_SECS,
            output_actions=[self.bpod.actions.play_noise],
            state_change_conditions={'Tup': 'hide_stim'},
        )

        sma.add_state(
            state_name='freeze_reward',
            state_timer=0,
            output_actions=[self.bpod.actions.bonsai_show_center],
            state_change_conditions={'Tup': 'reward'},
        )

        sma.add_state(
            state_name='reward',
            state_timer=self.reward_time,
            output_actions=[('Valve1', 255), ('BNC1', 255)],
            state_change_conditions={'Tup': 'correct'},
        )

        sma.add_state(
            state_name='correct',
            state_timer=self.task_params.FEEDBACK_CORRECT_DELAY_SECS - self.reward_time,
            output_actions=[],
            state_change_conditions={'Tup': 'hide_stim'},
        )

        sma.add_state(
            state_name='hide_stim',
            state_timer=0.1,
            output_actions=[self.bpod.actions.bonsai_hide_stim],
            state_change_conditions={'Tup': 'exit_state', 'BNC1High': 'exit_state', 'BNC1Low': 'exit_state'},
        )

        sma.add_state(
            state_name='exit_state',
            state_timer=self.task_params.ITI_DELAY_SECS,
            output_actions=[('BNC1', 255)],
            state_change_conditions={'Tup': 'exit'}
        )
        return sma

    def next_trial(self):
        self.trial_num += 1
        # TODO: quiescent period should be overloadable
        self.draw_next_trial_info(
            pleft=self.trials_table.at[self.trial_num, 'stim_probability_left'],
            contrast=self.trials_table.at[self.trial_num, 'contrast'],
            position=self.trials_table.at[self.trial_num, 'position'],
            reward_amount=self.trials_table.at[self.trial_num, 'reward_amount']
        )

    def show_trial_log(self, extra_info=''):
        trial_info = self.trials_table.iloc[self.trial_num]
        extra_info = f"""RICH PROBABILITY:     {trial_info.rich_probability_left}"""
        super().show_trial_log(extra_info=extra_info)

    @property
    def omit_feedback(self):
        # TODO: needs to be a pre-generated parameter
        # return self.trials_table.at[self.trial_num, 'omit_feedback']
        return False


if __name__ == '__main__':  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

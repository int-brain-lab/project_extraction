from pathlib import Path
from typing import Annotated

import yaml
from annotated_types import Interval

import iblrig.misc
from iblrig.base_choice_world import BiasedChoiceWorldSession, BiasedChoiceWorldTrialData
from iblrig.hardware import SOFTCODE
from iblutil.util import setup_logger
from pybpodapi.protocol import StateMachine

log = setup_logger(__name__)

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)


class CuedBiasedChoiceWorldTrialData(BiasedChoiceWorldTrialData):
    """Pydantic Model for Trial Data, extended from :class:`~.iblrig.base_choice_world.BiasedChoiceWorldTrialData`."""

    play_audio_cue: bool


class Session(BiasedChoiceWorldSession):
    protocol_name = 'samuel_cuedBiasedChoiceWorld'
    TrialDataModel = CuedBiasedChoiceWorldTrialData

    def __init__(self, *args, probability_audio_cue: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        # store parameters to task_params
        self.session_info['PROBABILITY_AUDIO_CUE'] = probability_audio_cue

        # loads in the settings in order to determine the main sync and thus the pipeline extractor tasks
        is_main_sync = self.hardware_settings.get('MAIN_SYNC', False)
        trials_task = 'CuedBiasedTrials' if is_main_sync else 'CuedBiasedTrialsTimeline'
        self.extractor_tasks = ['TrialRegisterRaw', trials_task, 'TrainingStatus']

        # Update experiment description which was created by superclass init
        self.experiment_description['tasks'][-1][self.protocol_name]['extractors'] = self.extractor_tasks

    def get_state_machine_trial(self, i):
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
                'Tup': 'play_tone',
                self.movement_left: 'reset_rotary_encoder',
                self.movement_right: 'reset_rotary_encoder',
            },
        )
        # play tone, move on to next state if sound is detected, with a time-out of 0.1s
        # SP how can we make sure the delay between play_tone and stim_on is always exactly 1s?
        sma.add_state(
            state_name='play_tone',
            state_timer=0.1,  # SP is this necessary??
            output_actions=[self.bpod.actions.play_tone],
            state_change_conditions={
                'Tup': 'interactive_delay',
                'BNC2High': 'interactive_delay',
            },
        )
        # this will add a delay between auditory cue and visual stimulus
        # this needs to be precise and accurate based on the parameter
        sma.add_state(
            state_name='interactive_delay',
            state_timer=self.task_params.INTERACTIVE_DELAY,
            output_actions=[],
            state_change_conditions={'Tup': 'stim_on'},
        )
        # show stimulus, move on to next state if a frame2ttl is detected, with a time-out of 0.1s
        sma.add_state(
            state_name='stim_on',
            state_timer=0.1,
            output_actions=[self.bpod.actions.bonsai_show_stim],
            state_change_conditions={
                'Tup': 'reset2_rotary_encoder',
                'BNC1High': 'reset2_rotary_encoder',
                'BNC1Low': 'reset2_rotary_encoder',
            },
        )

        sma.add_state(
            state_name='reset2_rotary_encoder',
            state_timer=0.05,  # the delay here is to avoid race conditions in the bonsai flow
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={'Tup': 'closed_loop'},
        )

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
            output_actions=[self.bpod.actions.bonsai_freeze_stim],
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
            state_timer=self.task_params.FEEDBACK_CORRECT_DELAY_SECS,
            output_actions=[],
            state_change_conditions={'Tup': 'hide_stim'},
        )

        sma.add_state(
            state_name='hide_stim',
            state_timer=0.1,
            output_actions=[self.bpod.actions.bonsai_hide_stim],
            state_change_conditions={
                'Tup': 'exit_state',
                'BNC1High': 'exit_state',
                'BNC1Low': 'exit_state',
            },
        )

        sma.add_state(
            state_name='exit_state',
            state_timer=self.task_params.ITI_DELAY_SECS,
            output_actions=[('BNC1', 255)],
            state_change_conditions={'Tup': 'exit'},
        )
        return sma

    @staticmethod
    def extra_parser():
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--probability_audio_cue',
            option_strings=['--probability_audio_cue'],
            dest='probability_audio_cue',
            default=DEFAULTS.get('PROBABILITY_AUDIO_CUE', 1.0),
            type=float,
            help='defines the probability of the audio cue to be played',
        )
        return parser


if __name__ == '__main__':  # pragma: no cover
    task_kwargs = iblrig.misc.get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**task_kwargs)
    sess.run()

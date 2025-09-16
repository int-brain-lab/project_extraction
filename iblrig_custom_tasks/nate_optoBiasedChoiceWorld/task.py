"""
This task is a replica of BiasedChoiceWorldSession with the addition of optogenetic stimulation
An `opto_stimulation` column is added to the trials_table, which is a boolean array of length NTRIALS_INIT
The PROBABILITY_OPTO_STIMULATION parameter is used to determine the probability of optogenetic stimulation
for each trial

Additionally the state machine is modified to add output TTLs for optogenetic stimulation
"""

import logging
import random
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from iblrig.base_choice_world import SOFTCODE, BiasedChoiceWorldSession
from iblrig.misc import get_task_arguments
from pybpodapi.protocol import StateMachine

stim_location_history = []

log = logging.getLogger('iblrig.task')

INTERACTIVE_DELAY = 1.0
NTRIALS_INIT = 2000
SOFTCODE_STOP_ZAPIT = max(SOFTCODE).value + 1
SOFTCODE_FIRE_ZAPIT = max(SOFTCODE).value + 2

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)


class OptoStateMachine(StateMachine):
    """
    This class just adds output TTL on BNC2 for defined states
    """

    def __init__(
        self,
        bpod,
        is_opto_stimulation=False,
        states_opto_ttls=None,
        states_opto_stop=None,
        states_mask_ttls=None,
        use_zapit=False
    ):
        super().__init__(bpod)
        self.is_opto_stimulation = is_opto_stimulation
        self.states_opto_ttls = states_opto_ttls or []
        self.states_opto_stop = states_opto_stop or []
        self.states_mask_ttls = states_mask_ttls or []
        self.use_zapit = use_zapit

    def add_state(self, **kwargs):
        if self.is_opto_stimulation:
            if kwargs['state_name'] in self.states_opto_ttls:
                kwargs['output_actions'] += [('BNC2', 255)]
                if self.use_zapit:
                    kwargs['output_actions'] += [('SoftCode', SOFTCODE_FIRE_ZAPIT)]
            elif kwargs['state_name'] in self.states_opto_stop and self.use_zapit:
                kwargs['output_actions'] += [('SoftCode', SOFTCODE_STOP_ZAPIT)]
        if kwargs['state_name'] in self.states_mask_ttls:
            kwargs['output_actions'] += [
                ('PWM1', 255),
            ]
        super().add_state(**kwargs)


class Session(BiasedChoiceWorldSession):
    protocol_name = 'nate_optoBiasedChoiceWorld'
    extractor_tasks = ['TrialRegisterRaw', 'ChoiceWorldTrials', 'TrainingStatus']

    def __init__(
        self,
        *args,
        probability_opto_stim: float = DEFAULTS['PROBABILITY_OPTO_STIM'],
        contrast_set_probability_type: Literal['skew_zero', 'uniform'] = DEFAULTS['CONTRAST_SET_PROBABILITY_TYPE'],
        opto_ttl_states: list[str] = DEFAULTS['OPTO_TTL_STATES'],
        opto_stop_states: list[str] = DEFAULTS['OPTO_STOP_STATES'],
        mask_ttl_states: list[str] = DEFAULTS['MASK_TTL_STATES'],
        use_zapit: bool = DEFAULTS['USE_ZAPIT'],
        stim_reverse: float = DEFAULTS['STIM_REVERSE'],
        reward_amount_ul: float = DEFAULTS['REWARD_AMOUNT_UL'],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_params['CONTRAST_SET_PROBABILITY_TYPE'] = contrast_set_probability_type
        self.task_params['OPTO_TTL_STATES'] = opto_ttl_states
        self.task_params['OPTO_STOP_STATES'] = opto_stop_states
        self.task_params['MASK_TTL_STATES'] = mask_ttl_states
        self.task_params['PROBABILITY_OPTO_STIM'] = probability_opto_stim
        self.task_params['STIM_REVERSE'] = stim_reverse
        self.task_params['REWARD_AMOUNT_UL'] = reward_amount_ul
        self.task_params['USE_ZAPIT'] = use_zapit
        self.current_location_idx = None
        self.num_cond = None
        self.client = None

        # generates the opto stimulation for each trial
        self.trials_table['opto_stimulation'] = np.random.choice(
            [0, 1],
            p=[1 - probability_opto_stim, probability_opto_stim],
            size=NTRIALS_INIT,
        ).astype(bool)

    def start_hardware(self):
        if self.task_params['USE_ZAPIT']:
            sys.path.append(r'C:\zapit-tcp-bridge\python')
            if find_spec('TCPclient') is None:
                raise ImportError(f'{__file__} requires zapit-tcp-bridge')
            from TCPclient import TCPclient

            self.client = TCPclient(tcp_port=1488, tcp_ip='127.0.0.1')
            self.client.close()  # need to ensure is closed first; currently nowhere that this is defined at end of task!
            self.client.connect()
            self.num_cond = self.client.get_num_conditions()[1][0]
            log.warning(f'Number of conditions: {self.num_cond}')
        super().start_hardware()
        if self.task_params['USE_ZAPIT']:
            # add the softcodes for the zapit opto stimulation
            soft_code_dict = self.bpod.softcodes
            soft_code_dict.update({SOFTCODE_STOP_ZAPIT: self.zapit_stop_laser})
            soft_code_dict.update({SOFTCODE_FIRE_ZAPIT: self.zapit_fire_laser})
            self.bpod.register_softcodes(soft_code_dict)

    def zapit_arm_laser(self):
        log.warning('Arming laser')
        self.current_location_idx = random.randrange(1, int(self.num_cond))
        response = self.client.send_samples(
            conditionNum=self.current_location_idx, laser_On=True, hardwareTriggered_On=True, logging_On=True
        )
        log.warning(response)
        stim_location_history.append(self.current_location_idx)

    def zapit_fire_laser(self):
        # just logging - actual firing will be triggered by the state machine via TTL
        log.warning('Firing laser')

    def zapit_stop_laser(self):
        log.warning('Stopping laser')
        self.client.stop_optostim()

    def _instantiate_state_machine(self, trial_number=None):
        is_opto_stimulation = self.trials_table.at[trial_number, 'opto_stimulation']
        if is_opto_stimulation and self.task_params['USE_ZAPIT']:
            # we start the laser waiting for a TTL trigger before sending out the state machine on opto trials
            self.zapit_arm_laser()
        return OptoStateMachine(
            self.bpod,
            is_opto_stimulation=is_opto_stimulation,
            states_opto_ttls=self.task_params['OPTO_TTL_STATES'],
            states_opto_stop=self.task_params['OPTO_STOP_STATES'],
            states_mask_ttls=self.task_params['MASK_TTL_STATES'],
            use_zapit=self.task_params['USE_ZAPIT']
        )

    @staticmethod
    def extra_parser():
        """:return: argparse.parser()"""
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--probability_opto_stim',
            option_strings=['--probability_opto_stim'],
            dest='probability_opto_stim',
            default=DEFAULTS['PROBABILITY_OPTO_STIM'],
            type=float,
            help=f'probability of opto-genetic stimulation (default: {DEFAULTS["PROBABILITY_OPTO_STIM"]})',
        )
        parser.add_argument(
            '--reward_amount_ul',
            option_strings=['--reward_amount_ul'],
            dest='reward_amount_ul',
            default=DEFAULTS['REWARD_AMOUNT_UL'],
            type=float,
            help=f'reward amount, ul (default: {DEFAULTS["REWARD_AMOUNT_UL"]})',
        )
        parser.add_argument(
            '--contrast_set_probability_type',
            option_strings=['--contrast_set_probability_type'],
            dest='contrast_set_probability_type',
            default=DEFAULTS['CONTRAST_SET_PROBABILITY_TYPE'],
            type=str,
            choices=['skew_zero', 'uniform'],
            help=f'probability type for contrast set (default: {DEFAULTS["CONTRAST_SET_PROBABILITY_TYPE"]})',
        )
        parser.add_argument(
            '--opto_ttl_states',
            option_strings=['--opto_ttl_states'],
            dest='opto_ttl_states',
            default=DEFAULTS['OPTO_TTL_STATES'],
            nargs='+',
            type=str,
            help='list of the state machine states where opto stim should be delivered',
        )
        parser.add_argument(
            '--opto_stop_states',
            option_strings=['--opto_stop_states'],
            dest='opto_stop_states',
            default=DEFAULTS['OPTO_STOP_STATES'],
            nargs='+',
            type=str,
            help='list of the state machine states where opto stim should be stopped',
        )
        parser.add_argument(
            '--mask_ttl_states',
            option_strings=['--mask_ttl_states'],
            dest='mask_ttl_states',
            default=DEFAULTS['MASK_TTL_STATES'],
            nargs='+',
            type=str,
            help='list of the state machine states where mask stim should be delivered',
        )
        parser.add_argument(
            '--use_zapit',
            option_strings=['--use_zapit'],
            default=DEFAULTS['USE_ZAPIT'],
            action='store_true',
            dest='use_zapit',
            help='use the Zapit TCP client',
        )
        parser.add_argument(
            '--stim_reverse',
            option_strings=['--stim_reverse'],
            default=DEFAULTS['STIM_REVERSE'],
            action='store_true',
            dest='stim_reverse',
            help='Inverse relationship of wheel to stimulus movement',
        )
        return parser


if __name__ == '__main__':  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

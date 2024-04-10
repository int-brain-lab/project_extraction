"""
This task is a replica of BiasedChoiceWorldSession with the addition of optogenetic stimulation
An `opto_stimulation` column is added to the trials_table, which is a boolean array of length NTRIALS_INIT
The PROBABILITY_OPTO_STIMULATION parameter is used to determine the probability of optogenetic stimulation
for each trial

Additionally the state machine is modified to add output TTLs for optogenetic stimulation
"""
import logging
import sys
from argparse import ArgumentTypeError
from pathlib import Path
from typing import Literal
import warnings

import numpy as np
import yaml

import iblrig
from iblrig.base_choice_world import SOFTCODE, BiasedChoiceWorldSession
from pybpodapi.protocol import StateMachine
from importlib import reload
import random

ZAPIT_PYTHON = r'C:\zapit-tcp-bridge\python'

try:
    assert Path(ZAPIT_PYTHON).exists()
    sys.path.append(ZAPIT_PYTHON)
    import Python_TCP_Utils as ptu
    from TCPclient import TCPclient
except (AssertionError, ModuleNotFoundError):
    warnings.warn(
        'Please clone https://github.com/Zapit-Optostim/zapit-tcp-bridge to '
        f'{Path(ZAPIT_PYTHON).parents[1]}', RuntimeWarning)


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
    ):
        super().__init__(bpod)
        self.is_opto_stimulation = is_opto_stimulation
        self.states_opto_ttls = states_opto_ttls or []
        self.states_opto_stop = states_opto_stop or []

    def add_state(self, **kwargs):
        if self.is_opto_stimulation:
            if kwargs['state_name'] in self.states_opto_ttls:
                kwargs['output_actions'] += [
                    ('SoftCode', SOFTCODE_FIRE_ZAPIT),
                    ('BNC2', 255),
                ]
            elif kwargs['state_name'] in self.states_opto_stop:
                kwargs['output_actions'] += [('SoftCode', SOFTCODE_STOP_ZAPIT)]
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_params['CONTRAST_SET_PROBABILITY_TYPE'] = contrast_set_probability_type
        self.task_params['OPTO_TTL_STATES'] = opto_ttl_states
        self.task_params['OPTO_STOP_STATES'] = opto_stop_states
        self.task_params['PROBABILITY_OPTO_STIM'] = probability_opto_stim

        # generates the opto stimulation for each trial
        self.trials_table['opto_stimulation'] = np.random.choice(
            [0, 1],
            p=[1 - probability_opto_stim, probability_opto_stim],
            size=NTRIALS_INIT,
        ).astype(bool)
        self.trials_table['laser_location_idx'] = np.zeros(NTRIALS_INIT, dtype=int)

    def draw_next_trial_info(self, **kwargs):
        """Draw next trial variables.

        This is called by the `next_trial` method before updating the Bpod state machine. This
        subclass method generates the stimulation index which is sent to Zapit when arming the
        laser on stimulation trials.
        """
        if self.trials_table.at[self.trial_num, 'opto_stimulation']:
            N = int(self.task_params.get('NUM_OPTO_COND', 52))
            self.trials_table.at[self.trial_num, 'laser_location_idx'] = random.randrange(1, N)

    def start_hardware(self):
        self.client = TCPclient(tcp_port=1488, tcp_ip='127.0.0.1')

        self.client.close()  # need to ensure is closed first; currently nowhere that this is defined at end of task!
        self.client.connect()
        super().start_hardware()
        # add the softcodes for the zapit opto stimulation
        soft_code_dict = self.bpod.softcodes
        soft_code_dict.update({SOFTCODE_STOP_ZAPIT: self.zapit_stop_laser})
        soft_code_dict.update({SOFTCODE_FIRE_ZAPIT: self.zapit_fire_laser})
        self.bpod.register_softcodes(soft_code_dict)

    def zapit_arm_laser(self):
        log.warning('Arming laser')
        # this is where you define the laser stim (i.e., arm the laser)

        current_location_idx = self.trials_table.at[self.trial_num, 'laser_location_idx']

        #hZP.send_samples(
        #    conditionNum=current_location_idx, hardwareTriggered=True, logging=True
        #)

        zapit_byte_tuple, zapit_int_tuple = ptu.gen_Zapit_byte_tuple(
            trial_state_command=1,
            arg_keys_dict={'conditionNum_channel': True, 'laser_channel': True,
                           'hardwareTriggered_channel': True, 'logging_channel': False,
                           'verbose_channel': False},
            arg_values_dict={'conditionNum': current_location_idx, 'laser_ON': True,
                             'hardwareTriggered_ON': True, 'logging_ON': False,
                             'verbose_ON': False}
        )
        response = self.client.send_receive(zapit_byte_tuple)
        log.warning(response)

    def zapit_fire_laser(self):
        # just logging - actual firing will be triggered by the state machine via TTL
        # this really only triggers a ttl and sends a log entry - no need to plug in code here
        log.warning('Firing laser')

    def zapit_stop_laser(self):
        log.warning('Stopping laser')
        current_location_idx = self.trials_table.at[self.trial_num, 'laser_location_idx']
        zapit_byte_tuple, zapit_int_tuple = ptu.gen_Zapit_byte_tuple(
            trial_state_command=0,
            arg_keys_dict={'conditionNum_channel': True, 'laser_channel': True,
                           'hardwareTriggered_channel': True, 'logging_channel': False,
                           'verbose_channel': False},
            arg_values_dict={'conditionNum': current_location_idx, 'laser_ON': True,
                             'hardwareTriggered_ON': False, 'logging_ON': False,
                             'verbose_ON': False}
        )
        response = self.client.send_receive(zapit_byte_tuple)

    def _instantiate_state_machine(self, trial_number=None):
        """
        We override this using the custom class OptoStateMachine that appends TTLs for optogenetic stimulation where needed
        :param trial_number:
        :return:
        """
        is_opto_stimulation = self.trials_table.at[trial_number, 'opto_stimulation']
        # we start the laser waiting for a TTL trigger before sending out the state machine on opto trials
        if is_opto_stimulation:
            self.zapit_arm_laser()
        return OptoStateMachine(
            self.bpod,
            is_opto_stimulation=is_opto_stimulation,
            states_opto_ttls=self.task_params['OPTO_TTL_STATES'],
            states_opto_stop=self.task_params['OPTO_STOP_STATES'],
        )

    @staticmethod
    def extra_parser():
        """:return: argparse.parser()"""
        def positive_int(value):
            if (value := int(value)) <= 0:
                raise ArgumentTypeError(f'"{value}" is an invalid positive int value')
            return value

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
            '--n_opto_cond',
            default=DEFAULTS['NUM_OPTO_COND'],
            type=positive_int,
            help='the number (N) of preset conditions to draw from, where N > x > 0',
        )
        return parser


if __name__ == '__main__':  # pragma: no cover
    kwargs = iblrig.misc.get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

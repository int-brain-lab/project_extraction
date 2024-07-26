"""
This task is a replica of max_staticTrainingChoiceWorld with the addition of optogenetic stimulation
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

import iblrig
from iblrig.base_choice_world import SOFTCODE 
from pybpodapi.protocol import StateMachine
from iblrig_custom_tasks.max_staticTrainingChoiceWorld.task import Session as StaticTrainingChoiceSession 
from iblrig_custom_tasks.max_optoStaticTrainingChoiceWorld.PulsePal import PulsePalMixin, PulsePalStateMachine

stim_location_history = []

log = logging.getLogger('iblrig.task')

NTRIALS_INIT = 2000
SOFTCODE_FIRE_LED = max(SOFTCODE).value + 1
SOFTCODE_RAMP_DOWN_LED = max(SOFTCODE).value + 2
RAMP_SECONDS = .25 # time to ramp down the opto stim # TODO: make this a parameter

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)

class Session(StaticTrainingChoiceSession, PulsePalMixin):
    protocol_name = 'max_optoStaticTrainingChoiceWorld'
    #extractor_tasks = StaticTrainingChoiceSession.extractor_tasks # TODO: add opto extractor tasks here?

    def __init__(
        self,
        *args,
        probability_opto_stim: float = DEFAULTS['PROBABILITY_OPTO_STIM'],
        opto_ttl_states: list[str] = DEFAULTS['OPTO_TTL_STATES'],
        opto_stop_states: list[str] = DEFAULTS['OPTO_STOP_STATES'],
        max_laser_time: float = DEFAULTS['MAX_LASER_TIME'],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_params['OPTO_TTL_STATES'] = opto_ttl_states
        self.task_params['OPTO_STOP_STATES'] = opto_stop_states
        self.task_params['PROBABILITY_OPTO_STIM'] = probability_opto_stim
        self.task_params['MAX_LASER_TIME'] = max_laser_time

        # generates the opto stimulation for each trial
        self.trials_table['opto_stimulation'] = np.random.choice(
            [0, 1],
            p=[1 - probability_opto_stim, probability_opto_stim],
            size=NTRIALS_INIT,
        ).astype(bool)
        log.warning(self.trials_table['opto_stimulation'])
    
    def _instantiate_state_machine(self, trial_number=None):
        """
        We override this using the custom class PulsePalStateMachine that appends TTLs for optogenetic stimulation where needed
        :param trial_number:
        :return:
        """
        log.warning('Instantiating state machine')
        is_opto_stimulation = self.trials_table.at[trial_number, 'opto_stimulation']
        if is_opto_stimulation:
            self.arm_opto_stim()
            self.arm_ttl_stim()
        return PulsePalStateMachine(
            self.bpod,
            trigger_type='soft', # software trigger
            is_opto_stimulation=is_opto_stimulation,
            states_opto_ttls=self.task_params['OPTO_TTL_STATES'],
            states_opto_stop=self.task_params['OPTO_STOP_STATES'],
        )

    def arm_opto_stim(self):
        # define a contant offset voltage with a ramp down at the end to avoid rebound excitation
        # TODO: set the laser power appropriately based on calibration values!
        log.warning('Arming opto stim')
        ramp = np.linspace(5, 0, 1000) # SET POWER
        t = np.linspace(0, RAMP_SECONDS, 1000)
        v = np.concatenate((np.array([5]), ramp)) # SET POWER
        t = np.concatenate((np.array([0]), t + self.task_params['MAX_LASER_TIME']))

        self.pulsepal_connection.programOutputChannelParam('phase1Duration', 1, self.task_params['MAX_LASER_TIME'])
        self.pulsepal_connection.sendCustomPulseTrain(1, t, v)
        self.pulsepal_connection.programOutputChannelParam('customTrainID', 1, 1)

    @property
    def stim_length_seconds(self):
        return self.task_params['MAX_LASER_TIME']

    def stop_opto_stim(self):
        # we will modify this function to ramp down the opto stim rather than abruptly stopping it
        # send instructions to set the TTL back to 0
        self.pulsepal_connection.programOutputChannelParam('phase1Duration', 2, self.task_params['MAX_LASER_TIME'])
        self.pulsepal_connection.sendCustomPulseTrain(2, [0,], [0,])
        self.pulsepal_connection.programOutputChannelParam('customTrainID', 2, 2)
        
        # send instructions to ramp the opto stim down to 0
        v = np.linspace(5, 0, 1000)
        t = np.linspace(0, RAMP_SECONDS, 1000)
        self.pulsepal_connection.programOutputChannelParam('phase1Duration', 1, self.task_params['MAX_LASER_TIME'])
        self.pulsepal_connection.sendCustomPulseTrain(1, t, v)
        self.pulsepal_connection.programOutputChannelParam('customTrainID', 1, 1)

        # FIXME: THIS IS CURRENTLY RESETING THE LED HIGH IF T_MAX HAS ELAPSED. NEED TO FIX!
        # trigger these instructions
        self.pulsepal_connection.triggerOutputChannels(1, 1, 0, 0)
        log.warning('Stopped opto stim')

    def start_hardware(self):
        super().start_hardware()
        super().start_opto_hardware()
       

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
            '--max_laser_time',
            option_strings=['--max_laser_time'],
            dest='max_laser_time',
            default=DEFAULTS['MAX_LASER_TIME'],
            type=float,
            help='Maximum laser duration in seconds',
        )

        return parser


if __name__ == '__main__':  # pragma: no cover
    kwargs = iblrig.misc.get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

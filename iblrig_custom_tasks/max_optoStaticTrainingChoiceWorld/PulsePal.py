import logging
import sys
from typing import Literal
from abc import ABC, abstractmethod
import numpy as np

from iblrig.base_choice_world import SOFTCODE 
from pybpodapi.protocol import StateMachine, Bpod
from pypulsepal import PulsePalObject
from iblrig.base_tasks import BaseSession

log = logging.getLogger('iblrig.task')

SOFTCODE_FIRE_PULSEPAL = max(SOFTCODE).value + 1
SOFTCODE_STOP_PULSEPAL = max(SOFTCODE).value + 2
V_MAX = 5


class PulsePalStateMachine(StateMachine):
    """
    This class adds:
        1. Hardware or sofware triggering of optogenetic stimulation via a PulsePal (or BPod Analog Output Module)
            EITHER
            - adds soft-codes for starting and stopping the opto stim
            OR
            - sets up a TTL to hardware trigger the PulsePal
        2. (not yet implemented!!!) sets up a TTL channel for recording opto stim times from the PulsePal
    """
    # TODO: define the TTL channel for recording opto stim times?
    def __init__(
        self,
        bpod,
        trigger_type: Literal['soft', 'hardware'] = 'soft',
        is_opto_stimulation=False,
        states_opto_ttls=None,
        states_opto_stop=None,
        opto_t_max_seconds=None,
    ):
        super().__init__(bpod)
        self.trigger_type = trigger_type
        self.is_opto_stimulation = is_opto_stimulation
        self.states_opto_ttls = states_opto_ttls or []
        self.states_opto_stop = states_opto_stop or []

        # Set global timer 1 for T_MAX
        self.set_global_timer(timer_id=1, timer_duration=opto_t_max_seconds)

    def add_state(self, **kwargs):
        if self.is_opto_stimulation:
            if kwargs['state_name'] in self.states_opto_ttls:
                if self.trigger_type == 'soft':
                    kwargs['output_actions'] += [('SoftCode', SOFTCODE_FIRE_PULSEPAL),]
                elif self.trigger_type == 'hardware':
                    kwargs['output_actions'] += [('BNC2', 255),]
                kwargs['output_actions'] += [(Bpod.OutputChannels.GlobalTimerTrig, 1)] # start the global timer when the opto stim comes on
            elif kwargs['state_name'] in self.states_opto_stop:
                if self.trigger_type == 'soft':
                    kwargs['output_actions'] += [('SoftCode', SOFTCODE_STOP_PULSEPAL),]
                elif self.trigger_type == 'hardware':
                    kwargs['output_actions'] += [('BNC2', 0),]
      
        super().add_state(**kwargs)

class PulsePalMixin(ABC):
    """
    A mixin class that adds optogenetic stimulation capabilities to a task via the 
    PulsePal module (or a Analog Output module running PulsePal firmware). It is used 
    in conjunction with the PulsePalStateMachine class rather than the StateMachine class. 

    The user must define the arm_opto_stim method to define the parameters for optogenetic stimulation.
    PulsePalMixin supports soft-code triggering via the start_opto_stim and stop_opto_stim methods.
    Hardware triggering is also supported by defining trigger channels in the arm_opto_stim method.

    The opto stim is currently hard-coded on output channel 1.
    A TTL pulse is hard-coded on output channel 2 for accurately recording trigger times. This TTL
    will rise when the opto stim starts and fall when it stops, thus accurately recording software trigger times.
    """

    def start_opto_hardware(self):
        self.pulsepal_connection = PulsePalObject('COM13') # TODO: get port from hardware params
        log.warning('Connected to PulsePal')
        # TODO: get the calibration value for this specific cannula
        #super().start_hardware() # TODO: move this out

        # add the softcodes for the PulsePal
        soft_code_dict = self.bpod.softcodes
        soft_code_dict.update({SOFTCODE_STOP_PULSEPAL: self.stop_opto_stim})
        soft_code_dict.update({SOFTCODE_FIRE_PULSEPAL: self.start_opto_stim})
        self.bpod.register_softcodes(soft_code_dict)

    @abstractmethod
    def arm_opto_stim(self, ttl_output_channel):
        raise NotImplementedError("User must define the stimulus and trigger type to deliver with pulsepal")
        # Define the pulse sequence and load it to the desired output channel here
        # This method should not fire the pulse train, that is handled by start_opto_stim() (soft-trigger) or a hardware trigger
        # See https://github.com/sanworks/PulsePal/blob/master/Python/Python3/PulsePalExample.py for examples
        # you should also define the max_stim_seconds property here to set the maximum duration of the pulse train

        ##############################
        # Example code to define a sine wave lasting 5 seconds
        voltages = list(range(0, 1000))
        for i in voltages:
            voltages[i] = math.sin(voltages[i]/float(10))*10  # Set 1,000 voltages to create a 20V peak-to-peak sine waveform
        times = np.linspace(0, 5, len(voltages))  # Create a time vector for the waveform
        self.stim_length_seconds = times[-1] # it is essential to get this property right so that the TTL for recording stim pulses is correcty defined
        self.pulsepal_connection.sendCustomPulseTrain(1, times, voltages)
        self.pulsepal_connection.programOutputChannelParam('customTrainID', 1, 1) 
        ##############################

    @property
    @abstractmethod
    def stim_length_seconds():
        # this should be set within the arm_opto_stim method
        pass

    def arm_ttl_stim(self):
        # a TTL pulse from channel 2 that rises when the opto stim starts and falls when it stops
        log.warning('Arming TTL signal')
        self.pulsepal_connection.programOutputChannelParam('phase1Duration', 2, self.stim_length_seconds)
        self.pulsepal_connection.sendCustomPulseTrain(2, [0,], [V_MAX,])
        self.pulsepal_connection.programOutputChannelParam('customTrainID', 2, 2)
        
    def start_opto_stim(self):
        self.pulsepal_connection.triggerOutputChannels(1, 1, 0, 0)
        log.warning('Started opto stim')
    
    def stop_opto_stim(self):
        # this will stop the pulse train instantly (and the corresponding TTL pulse)
        # To avoid rebound spiking in the case of GtACR, a ramp down is recommended
        self.pulsepal_connection.abortPulseTrains()

    def compute_vmax_from_calibration(self, calibration_value):
        # TODO: implement this method to convert the calibration value to a voltage for the opto stim
        pass
    
    def __del__(self):
        del self.pulsepal_connection

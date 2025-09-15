import logging
import time

import numpy as np
import pandas as pd
from pydantic import FilePath, validate_call

from iblrig import sound
from iblrig.base_choice_world import NTRIALS_INIT
from iblrig.base_tasks import BaseSession, BpodMixin
from iblrig.misc import get_task_arguments
from iblrig.pydantic_definitions import TrialDataModel
from iblrig.raw_data_loaders import bpod_session_data_to_dataframe
from iblutil.io import jsonable
from pybpodapi.state_machine import StateMachine

log = logging.getLogger('iblrig')


class TonotopicMappingTrialData(TrialDataModel):
    """Pydantic Model for Trial Data."""

    frequency_sequence: list[int]
    level_sequence: list[int]


class Session(BpodMixin, BaseSession):
    protocol_name = 'samuel_tonotopicMapping'
    TrialDataModel = TonotopicMappingTrialData

    parameters: np.ndarray = np.array([[], []])
    sequence: np.ndarray = np.array([])
    trial_num: int = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trials_table = self.TrialDataModel.preallocate_dataframe(NTRIALS_INIT)

        assert self.hardware_settings.device_sound.OUTPUT == 'harp', 'This task requires a Harp sound-card'
        assert self.task_params['n_freqs'] * len(self.task_params['levels']) <= 30, 'Harp only supports up to 30 waveforms'

        # define frequencies (log spaced from freq_0 to freq_1, rounded to nearest integer)
        frequencies = np.logspace(
            np.log10(self.task_params['freq_0']),
            np.log10(self.task_params['freq_1']),
            num=self.task_params['n_freqs'] - self.task_params['include_white_noise'],
        )
        frequencies = np.round(frequencies).astype(int)
        if self.task_params['include_white_noise']:
            frequencies = np.insert(frequencies, 0, -1, axis=0)

        # get all parameter combinations
        Session.parameters = np.array(np.meshgrid(frequencies, self.task_params['levels'])).T.reshape(-1, 2)

        # get LUT (or create new one based on frequencies)
        attenuation_file = self.get_task_directory().joinpath('attenuation.csv')
        if attenuation_file.exists():
            self.attenuation_lut = pd.read_csv(self.get_task_directory().joinpath('attenuation.csv'))
        else:
            self.attenuation_lut = pd.DataFrame({'frequency_hz': frequencies, 'attenuation_db': np.zeros(len(frequencies))})
            self.attenuation_lut.to_csv(attenuation_file, index=False)

        # calculate repetitions per state machine run (255 states max)
        self.repetitions = []
        max_reps_per_trial = 255 // self.n_stimuli
        reps_remaining = self.task_params['n_reps_per_stim']
        while reps_remaining > 0:
            self.repetitions.append(min(max_reps_per_trial, reps_remaining))
            reps_remaining -= self.repetitions[-1]

        # select channel configuration for playback
        match self.hardware_settings.device_sound.DEFAULT_CHANNELS:
            case 'left':
                channels = 'right'
            case 'right':
                channels = 'left'
            case _:
                channels = 'stereo'

        # generate stimuli
        self.stimuli = []
        for stimulus_index in range(self.n_stimuli):
            frequency = self.parameters[stimulus_index][0]
            level = self.parameters[stimulus_index][1]
            tmp = sound.make_sound(
                rate=self.task_params['fs'],
                frequency=frequency,
                duration=self.task_params['d_sound'],
                amplitude=self.task_params['amplitude'],
                fade=self.task_params['d_ramp'],
                chans=channels,
                gain_db=self.get_corrective_gain(frequency) + level,
            )
            self.stimuli.append(tmp)
        self.harp_indices = [i for i in range(2, self.n_stimuli + 2)]

    @property
    def n_stimuli(self):
        return self.parameters.shape[0]

    @property
    def n_trials(self):
        return len(self.repetitions)

    def get_corrective_gain(self, frequency: int):
        """get corrective gain values from LUT"""
        return np.interp(frequency, self.attenuation_lut['frequency_hz'], self.attenuation_lut['attenuation_db'])

    def start_mixin_sound(self):
        log.info(f'Pushing {len(self.parameters)} stimuli to Harp soundcard')
        sound.configure_sound_card(sounds=self.stimuli, indexes=self.harp_indices, sample_rate=self.task_params['fs'])

        module = self.bpod.sound_card
        module_port = f'Serial{module.serial_port if module is not None else "3"}'
        for stimulus_idx, harp_idx in enumerate(self.harp_indices):
            bpod_message = [ord('P'), harp_idx]
            bpod_action = (module_port, self.bpod._define_message(self.bpod.sound_card, bpod_message))
            self.bpod.actions.update({f'stim_{stimulus_idx}': bpod_action})

        self.bpod.softcode_handler_function = self.softcode_handler

    def start_hardware(self):
        self.start_mixin_bpod()
        self.start_mixin_sound()

    @staticmethod
    def get_state_name(state_idx: int):
        if state_idx < len(Session.sequence):
            stimulus_idx = Session.sequence[state_idx]
            frequency = Session.parameters[stimulus_idx][0]
            gain = Session.parameters[stimulus_idx][1]
            return '{:03d}_{:s}_{:d}dB'.format(state_idx, f'{frequency:d}Hz' if frequency >= 0 else 'WN', gain)
        else:
            return 'exit'

    @staticmethod
    def softcode_handler(softcode: int) -> None:
        """log some information about the current state"""
        state_index = softcode - 1
        stimulus_index = Session.sequence[state_index]
        frequency = Session.parameters[stimulus_index][0]
        gain = Session.parameters[stimulus_index][1]
        n_states = len(Session.sequence)
        if frequency >= 0:
            log.info(f'- {state_index + 1:03d}/{n_states:03d}: {frequency:8d} Hz, {gain:3d} dB')
        else:
            log.info(f'- {state_index + 1:03d}/{n_states:03d}: white noise, {gain:3d} dB')

    def get_state_machine(self, trial_number: int) -> StateMachine:
        # generate sequence, optionally shuffled (seeded with trial number)
        Session.sequence = np.repeat(np.arange(self.n_stimuli), self.repetitions[trial_number])
        if self.task_params['shuffle']:
            np.random.seed(trial_number)
            np.random.shuffle(Session.sequence)

        # build state machine
        sma = StateMachine(self.bpod)
        for state_idx, stimulus_idx in enumerate(self.sequence):
            sma.add_state(
                state_name=self.get_state_name(state_idx),
                state_timer=self.task_params['d_sound'] + self.task_params['d_pause'],
                output_actions=[self.bpod.actions[f'stim_{stimulus_idx}'], ('SoftCode', state_idx + 1)],
                state_change_conditions={'Tup': self.get_state_name(state_idx + 1)},
            )
        return sma

    def _run(self):
        log.info('Sending spacers to BNC ports')
        self.send_spacers()

        for trial_number in range(self.n_trials):
            self.trial_num = trial_number

            # run state machine
            log.info(f'Starting Trial #{trial_number} ({trial_number + 1}/{self.n_trials})')
            sma = self.get_state_machine(trial_number)
            self.bpod.send_state_machine(sma)
            self.bpod.run_state_machine(sma)

            # handle pause event
            if self.paused and trial_number < (self.task_params.NTRIALS - 1):
                log.info(f'Pausing session inbetween trials #{trial_number} and #{trial_number + 1}')
                while self.paused and not self.stopped:
                    time.sleep(1)
                if not self.stopped:
                    log.info('Resuming session')

            # save trial data
            self.trials_table.at[self.trial_num, 'frequency_sequence'] = self.parameters[self.sequence, 0]
            self.trials_table.at[self.trial_num, 'level_sequence'] = self.parameters[self.sequence, 1]
            bpod_data = self.bpod.session.current_trial.export()
            self.save_trial_data_to_json(bpod_data)

            # handle stop event
            if self.stopped:
                log.info('Stopping session after trial #%d', trial_number)
                break


@validate_call
def create_dataframe(jsonable_file: FilePath) -> pd.DataFrame:
    """
    Extract pandas DataFrame with relevant data from _iblrig_taskData.raw.jsonable file.

    Parameters
    ----------
    jsonable_file : str, os.PathLike
        Path to a session's `_iblrig_taskData.raw.jsonable` file.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing event data from the specified trials, with the following columns:

        *  Time : datetime.timedelta
              timestamp of the event (datetime.timedelta)
        *  Trial : int
              index of the trial, zero-based
        *  Stimulus : int
              index of the stimulus, zero-based
        *  Value : int
              value of the event: 1 for onset of waveform, 0 for offset
        *  Frequency : int
              frequency of the stimulus in Hz
        *  Attenuation : int
              attenuation of the stimulus in dB

    Raises
    ------
    ValueError
        If the input file is not named `_iblrig_taskData.raw.jsonable` or if it doesn't contain audio TTLs.
    """

    # check argument
    if jsonable_file.name != '_iblrig_taskData.raw.jsonable':
        raise ValueError('Input file must be named `_iblrig_taskData.raw.jsonable`')

    # load data
    bpod_dicts = jsonable.load_task_jsonable(jsonable_file)[1]
    bpod_data = bpod_session_data_to_dataframe(bpod_dicts)

    # remove frame2ttl data
    output = bpod_data[bpod_data['Channel'].eq('BNC2')].copy()
    if len(output) == 0:
        raise ValueError('No audio TTLs found in the provided file')

    # extract stimulus parameters from state names
    output[['Stimulus', 'Frequency', 'Attenuation']] = output['State'].str.extract(r'^(\d+)_(\d+|WN)[^-\d]+([-\d]+)dB$')
    output.replace({'Frequency': 'WN'}, '-1', inplace=True)
    output[['Stimulus', 'Frequency', 'Attenuation']] = output[['Stimulus', 'Frequency', 'Attenuation']].astype('Int64')

    # remove / reorder columns
    return output[['Trial', 'Stimulus', 'Value', 'Frequency', 'Attenuation']]


if __name__ == '__main__':
    kwargs = get_task_arguments()
    sess = Session(**kwargs)
    sess.run()

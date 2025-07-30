import logging
import time

import numpy as np
import pandas as pd

from iblrig import sound
from iblrig.base_choice_world import NTRIALS_INIT
from iblrig.base_tasks import BaseSession, BpodMixin
from iblrig.misc import get_task_arguments
from iblrig.pydantic_definitions import TrialDataModel
from pybpodapi.state_machine import StateMachine

log = logging.getLogger('iblrig')


class TonotopicMappingTrialData(TrialDataModel):
    """Pydantic Model for Trial Data."""

    frequency_sequence: list[int]


class Session(BpodMixin, BaseSession):
    protocol_name = 'samuel_tonotopicMapping'
    TrialDataModel = TonotopicMappingTrialData

    frequencies: np.ndarray = np.array([])
    sequence: np.ndarray = np.array([])
    trial_num: int = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trials_table = self.TrialDataModel.preallocate_dataframe(NTRIALS_INIT)

        assert self.hardware_settings.device_sound.OUTPUT == 'harp', 'This task requires a Harp sound-card'
        assert self.task_params['n_freqs'] <= 30, 'Harp only supports up to 30 individual sounds'

        # define frequencies (log spaced from freq_0 to freq_1, rounded to nearest integer)
        n_tones = self.task_params['n_freqs']
        if self.task_params['include_white_noise']:
            n_tones -= 1
        Session.frequencies = np.logspace(
            np.log10(self.task_params['freq_0']),
            np.log10(self.task_params['freq_1']),
            num=n_tones,
        )
        Session.frequencies = np.round(self.frequencies).astype(int)
        if self.task_params['include_white_noise']:
            Session.frequencies = np.insert(Session.frequencies, 0, -1, axis=0)

        # get LUT (or create new one based on frequencies)
        attenuation_file = self.get_task_directory().joinpath('attenuation.csv')
        if attenuation_file.exists():
            self.attenuation_lut = pd.read_csv(self.get_task_directory().joinpath('attenuation.csv'))
        else:
            self.attenuation_lut = pd.DataFrame(
                {'frequency_hz': self.frequencies, 'attenuation_db': np.zeros(self.n_frequencies)}
            )
            self.attenuation_lut.to_csv(attenuation_file, index=False)

        # get attenuation values from LUT (linear interpolation for missing values)
        if self.task_params['skip_attenuation']:
            self.attenuation = pd.DataFrame({'frequency_hz': self.frequencies, 'attenuation_db': np.zeros(self.n_frequencies)})
        else:
            self.attenuation = np.interp(
                self.frequencies,
                self.attenuation_lut['frequency_hz'],
                self.attenuation_lut['attenuation_db'],
            )

        # calculate repetitions per state machine run (255 states max)
        self.repetitions = []
        max_reps_per_trial = 255 // self.n_frequencies
        reps_remaining = self.task_params['n_reps_per_freq']
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
        for idx, f in enumerate(self.frequencies):
            tmp = sound.make_sound(
                rate=self.task_params['fs'],
                frequency=f,
                duration=self.task_params['d_sound'],
                amplitude=self.task_params['amplitude'],
                fade=self.task_params['d_ramp'],
                chans=channels,
                gain_db=self.attenuation[idx],
            )
            self.stimuli.append(tmp)
        self.indices = [i for i in range(2, len(self.stimuli) + 2)]

    @property
    def n_frequencies(self):
        return len(self.frequencies)

    @property
    def n_trials(self):
        return len(self.repetitions)

    def start_mixin_sound(self):
        log.info(f'Pushing {len(self.frequencies)} stimuli to Harp soundcard')
        sound.configure_sound_card(sounds=self.stimuli, indexes=self.indices, sample_rate=self.task_params['fs'])

        module = self.bpod.sound_card
        module_port = f'Serial{module.serial_port if module is not None else "3"}'
        for frequency_idx, harp_idx in enumerate(self.indices):
            bpod_message = [ord('P'), harp_idx]
            bpod_action = (module_port, self.bpod._define_message(self.bpod.sound_card, bpod_message))
            self.bpod.actions.update({f'freq_{frequency_idx}': bpod_action})

        self.bpod.softcode_handler_function = self.softcode_handler

    def start_hardware(self):
        self.start_mixin_bpod()
        self.start_mixin_sound()

    @staticmethod
    def get_state_name(state_idx: int):
        if state_idx < len(Session.sequence):
            frequency = Session.frequencies[Session.sequence[state_idx]]
            if frequency >= 0:
                return f'{state_idx + 1:03d}_{Session.frequencies[Session.sequence[state_idx]]}'
            else:
                return f'{state_idx + 1:03d}_white_noise'
        else:
            return 'exit'

    @staticmethod
    def softcode_handler(softcode: int) -> None:
        """log some information about the current state"""
        state_index = softcode - 1
        frequency_index = Session.sequence[state_index]
        frequency = Session.frequencies[frequency_index]
        n_states = len(Session.sequence)
        if frequency >= 0:
            log.info(f'- {state_index + 1:03d}/{n_states}: {frequency:5d} Hz')
        else:
            log.info(f'- {state_index + 1:03d}/{n_states}: white noise')

    def get_state_machine(self, trial_number: int) -> StateMachine:
        # generate sequence, optionally shuffled (seeded with trial number)
        Session.sequence = np.repeat(np.arange(len(self.frequencies)), self.repetitions[trial_number])
        if self.task_params['shuffle']:
            np.random.seed(trial_number)
            np.random.shuffle(Session.sequence)

        # build state machine
        sma = StateMachine(self.bpod)
        for state_idx, frequency_idx in enumerate(self.sequence):
            sma.add_state(
                state_name=self.get_state_name(state_idx),
                state_timer=self.task_params['d_sound'] + self.task_params['d_pause'],
                output_actions=[self.bpod.actions[f'freq_{frequency_idx}'], ('SoftCode', state_idx + 1)],
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
            self.trials_table.at[self.trial_num, 'frequency_sequence'] = self.frequencies[self.sequence]
            bpod_data = self.bpod.session.current_trial.export()
            self.save_trial_data_to_json(bpod_data)

            # handle stop event
            if self.stopped:
                log.info('Stopping session after trial #%d', trial_number)
                break


if __name__ == '__main__':
    kwargs = get_task_arguments()
    sess = Session(**kwargs)
    sess.run()

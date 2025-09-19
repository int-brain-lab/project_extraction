"""Tonotopic mapping task for IBL-Rig.

This module implements a passive tonotopic mapping protocol that plays a
set of pure tones (and optionally white noise) at various levels via the
Harp sound card, while emitting TTLs on BNC2 for synchronization. The
sequence is organized as a Bpod state machine with up to 255 states per
trial. After the session, the raw jsonable file is converted to a compact
Parquet table containing the stimulus on/off events and stimulus
parameters.

Key concepts:
- Frequencies are generated on a log-spaced grid between freq_0 and
  freq_1, optionally prepended by a white-noise pseudo-frequency (-1).
- For each frequency and level combination, a waveform is pre-rendered
  and uploaded to the Harp soundcard.
- Attenuation corrections can be provided via attenuation.csv in the
  task directory; otherwise a zero-attenuation LUT is created.
- Repetitions per trial are split to respect the 255-state limit.

The module also exposes a helper function `create_dataframe` to extract a
Pandas DataFrame from a session's `_iblrig_taskData.raw.jsonable` file,
keeping only the audio TTL channel and parsing the state names to recover
stimulus parameters.
"""

import logging
import time
from pathlib import Path
from typing import cast

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
    """Schema for per-trial metadata recorded by this task.

    Attributes
    ----------
    frequency_sequence : list[int]
        The sequence of frequencies (Hz; -1 encodes white noise) that
        were played within the trial, in the order they occurred.
    level_sequence : list[int]
        The corresponding sequence of level values (attenuation/gain in
        dB) used for each stimulus within the trial.
    """

    frequency_sequence: list[int]
    level_sequence: list[int]


class Session(BpodMixin, BaseSession):
    """Tonotopic mapping session orchestrated via Bpod and Harp.

    This session pre-generates a grid of stimuli (frequency x level),
    uploads them to the Harp sound-card, and then steps through a Bpod
    state machine that triggers each stimulus with a fixed duration and a
    pause between stimuli. The order can be shuffled per trial.
    """

    protocol_name = 'samuel_tonotopicMapping'
    TrialDataModel = TonotopicMappingTrialData

    parameters: np.ndarray = np.array([[], []])
    sequence: np.ndarray = np.array([])
    trial_num: int = -1

    def __init__(self, *args, **kwargs):
        """Initialize the session, stimuli, and repetition plan.

        Steps:
        - Validate that a Harp sound card is used and can hold all
          waveforms (<= 29).
        - Build the frequency grid (log-spaced) and combine with levels.
        - Load or create an attenuation LUT.
        - Split desired repetitions into chunks that fit into 255 states.
        - Pre-render and register all stimuli with appropriate gains.
        """
        super().__init__(*args, **kwargs)
        self.trials_table = self.TrialDataModel.preallocate_dataframe(NTRIALS_INIT)

        # Hardware constraints: Harp output only and waveform count limit
        assert self.hardware_settings.device_sound.OUTPUT == 'harp', 'This task requires a Harp sound-card'
        assert self.task_params['n_freqs'] * len(self.task_params['levels']) <= 29, 'Harp only supports up to 29 waveforms'

        # Define frequencies (log spaced from freq_0 to freq_1, rounded to nearest integer)
        frequencies = np.logspace(
            np.log10(self.task_params['freq_0']),
            np.log10(self.task_params['freq_1']),
            num=self.task_params['n_freqs'] - self.task_params['include_white_noise'],
        )
        frequencies = np.round(frequencies).astype(int)
        if self.task_params['include_white_noise']:
            # Use -1 as a sentinel for white noise to keep arrays numeric
            frequencies = np.insert(frequencies, 0, -1, axis=0)

        # Get all parameter combinations (frequency x level)
        Session.parameters = np.array(np.meshgrid(frequencies, self.task_params['levels'])).T.reshape(-1, 2)

        # Get LUT (or create new one based on frequencies) for corrective gains
        attenuation_file = self.get_task_directory().joinpath('attenuation.csv')
        if attenuation_file.exists():
            self.attenuation_lut = pd.read_csv(self.get_task_directory().joinpath('attenuation.csv'))
        else:
            self.attenuation_lut = pd.DataFrame({'frequency_hz': frequencies, 'attenuation_db': np.zeros(len(frequencies))})
            self.attenuation_lut.to_csv(attenuation_file, index=False)

        # Calculate repetitions per state machine run (255 states max)
        self.repetitions = []
        max_reps_per_trial = 255 // self.n_stimuli
        reps_remaining = self.task_params['n_reps_per_stim']
        while reps_remaining > 0:
            self.repetitions.append(min(max_reps_per_trial, reps_remaining))
            reps_remaining -= self.repetitions[-1]

        # Select channel configuration for playback. We mirror the default
        # so that sound is output on the opposite channel of DEFAULT.
        match self.hardware_settings.device_sound.DEFAULT_CHANNELS:
            case 'left':
                channels = 'right'
            case 'right':
                channels = 'left'
            case _:
                channels = 'stereo'

        # Generate and register stimuli for Harp
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
                # Combine corrective gain (from LUT) with requested level
                gain_db=self.get_corrective_gain(frequency) + level,
            )
            self.stimuli.append(tmp)
        # Harp indexes start at 2 because 1 is reserved by Bpod
        self.harp_indices = [i for i in range(2, self.n_stimuli + 2)]

    @property
    def n_stimuli(self):
        """Total number of distinct stimuli (frequency x level)."""
        return self.parameters.shape[0]

    @property
    def n_trials(self):
        """Number of Bpod runs required given the 255-state constraint."""
        return len(self.repetitions)

    def get_corrective_gain(self, frequency: int):
        """Return corrective gain in dB for a given frequency."""
        return np.interp(frequency, self.attenuation_lut['frequency_hz'], self.attenuation_lut['attenuation_db'])

    def start_mixin_sound(self):
        """Upload waveforms to Harp and register Bpod output actions."""
        log.info(f'Pushing {len(self.parameters)} stimuli to Harp soundcard')
        sound.configure_sound_card(sounds=self.stimuli, indexes=self.harp_indices, sample_rate=self.task_params['fs'])

        module = self.bpod.sound_card
        module_port = f'Serial{module.serial_port if module is not None else "3"}'
        for stimulus_idx, harp_idx in enumerate(self.harp_indices):
            bpod_message = [ord('P'), harp_idx]
            bpod_action = (module_port, self.bpod._define_message(self.bpod.sound_card, bpod_message))
            self.bpod.actions.update({f'stim_{stimulus_idx}': bpod_action})

        # Soft code allows logging per-state when the state is entered
        self.bpod.softcode_handler_function = self.softcode_handler

    def start_hardware(self):
        """Start Bpod and sound-card mixins."""
        self.start_mixin_bpod()
        self.start_mixin_sound()

    @staticmethod
    def get_state_name(state_idx: int):
        """Return human-readable state name including parameters.

        The name is formatted as: "{idx:03d}_{freq_label}_{gain}dB" where
        freq_label is e.g. "8000Hz" or "WN" for white noise. An extra
        "exit" state name is returned past the last stimulus index.
        """
        if state_idx < len(Session.sequence):
            stimulus_idx = Session.sequence[state_idx]
            frequency = Session.parameters[stimulus_idx][0]
            gain = Session.parameters[stimulus_idx][1]
            return '{:03d}_{:s}_{:d}dB'.format(state_idx, f'{frequency:d}Hz' if frequency >= 0 else 'WN', gain)
        else:
            return 'exit'

    @staticmethod
    def softcode_handler(softcode: int) -> None:
        """Log information about the current state entry.

        Parameters
        ----------
        softcode : int
            One-based state index sent by the state machine.
        """
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
        """Construct the Bpod state machine for a given trial.

        The sequence of stimuli is repeated `repetitions[trial_number]`
        times and optionally shuffled deterministically using the trial
        number as seed.
        """
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
        """Run the session across all required Bpod state-machine runs."""
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

            # save trial data: also store the exact sequence for QC and extraction
            self.trials_table.at[self.trial_num, 'frequency_sequence'] = self.parameters[self.sequence, 0]
            self.trials_table.at[self.trial_num, 'level_sequence'] = self.parameters[self.sequence, 1]
            bpod_data = self.bpod.session.current_trial.export()
            self.save_trial_data_to_json(bpod_data)

            # handle stop event
            if self.stopped:
                log.info('Stopping session after trial #%d', trial_number)
                break

        # convert data to parquet and remove jsonable file
        path_jsonable = cast(Path, self.paths['DATA_FILE_PATH'])
        path_parquet = path_jsonable.with_suffix('.pqt')
        data = create_dataframe(path_jsonable)
        data.to_parquet(path_parquet)
        assert path_parquet.exists()


@validate_call
def create_dataframe(jsonable_file: FilePath) -> pd.DataFrame:
    """Create a compact DataFrame of audio TTL events from a jsonable file.

    This utility loads the raw task jsonable, keeps only the audio TTL
    channel (BNC2), and parses the Bpod state names to recover the
    stimulus index, frequency (or white noise), and attenuation. The
    output is suitable for downstream analysis and alignment.

    Parameters
    ----------
    jsonable_file : str | os.PathLike
        Path to a session's `_iblrig_taskData.raw.jsonable` file.

    Returns
    -------
    pd.DataFrame
        Columns: Trial, Stimulus, Value, Frequency, Attenuation.

    Raises
    ------
    ValueError
        If the input file is not named `_iblrig_taskData.raw.jsonable` or
        if it doesn't contain audio TTLs on channel BNC2.
    """

    # check argument
    if jsonable_file.name != '_iblrig_taskData.raw.jsonable':
        raise ValueError('Input file must be named `_iblrig_taskData.raw.jsonable`')

    # load data
    bpod_dicts = jsonable.load_task_jsonable(jsonable_file)[1]
    bpod_data = bpod_session_data_to_dataframe(bpod_dicts)

    # restrict to audio TTL events
    output = bpod_data[bpod_data['Channel'].eq('BNC2')].copy()
    if len(output) == 0:
        raise ValueError('No audio TTLs found in the provided file')

    # extract stimulus parameters from state names
    output[['Stimulus', 'Frequency', 'Attenuation']] = output['State'].str.extract(r'^(\d+)_(\d+|WN)[^-\d]+([-\d]+)dB$')
    output.replace({'Frequency': 'WN'}, '-1', inplace=True)
    output[['Stimulus', 'Frequency', 'Attenuation']] = output[['Stimulus', 'Frequency', 'Attenuation']].astype('Int64')
    output.index.name = 'Nanoseconds'

    # remove / reorder columns
    return output[['Trial', 'Stimulus', 'Value', 'Frequency', 'Attenuation']]


if __name__ == '__main__':
    kwargs = get_task_arguments()
    sess = Session(**kwargs)
    sess.run()

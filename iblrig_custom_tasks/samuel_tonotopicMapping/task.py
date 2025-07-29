
import logging

import numpy as np

from iblrig import sound
from iblrig.base_tasks import BaseSession, BpodMixin
from iblrig.misc import get_task_arguments
from pybpodapi.state_machine import StateMachine

logger = logging.getLogger('iblrig')


class Session(BpodMixin, BaseSession):
    protocol_name = 'samuel_tonotopicMapping'

    frequencies: list[int] = []
    sequence: list[int] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.hardware_settings.device_sound.OUTPUT == 'harp', 'This task requires a Harp sound-card'
        assert self.task_params['n_freqs'] <= 31, 'Harp only supports up to 31 individual sounds'

        # define frequencies (log spaced from freq_0 to freq_1, rounded to nearest integer)
        Session.frequencies = np.logspace(
            np.log10(self.task_params['freq_0']),
            np.log10(self.task_params['freq_1']),
            num=self.task_params['n_freqs'],
        )
        Session.frequencies = np.round(self.frequencies).astype(int)

        # repetitions per state machine run (253 states max)
        self.repetitions = 253 // len(self.frequencies)

        # generate stimuli
        self.stimuli = []
        for f in self.frequencies:
            tmp = sound.make_sound(
                rate=self.task_params['fs'],
                frequency=f,
                duration=self.task_params['d_sound'],
                amplitude=self.task_params['amplitude'],
                fade=self.task_params['d_ramp'],
                chans='stereo',
                gain_db=0,
            )
            self.stimuli.append(tmp)
        self.indices = [i for i in range(2, len(self.stimuli) + 2)]

        # self.attenuation = pd.read_csv(self.get_task_directory().joinpath('attenuation.csv'))

    def start_mixin_sound(self):
        logger.info(f'Pushing {len(self.frequencies)} stimuli to Harp soundcard')
        sound.configure_sound_card(sounds=self.stimuli, indexes=self.indices, sample_rate=self.task_params['fs'])

        module = self.bpod.sound_card
        module_port = f'Serial{module.serial_port if module is not None else "3"}'
        for frequency_idx, harp_idx in enumerate(self.indices):
            bpod_message = [ord('P'), harp_idx]
            bpod_action = (module_port, self.bpod._define_message(self.bpod.sound_card, bpod_message))
            self.bpod.actions.update({f'freq_{frequency_idx}': bpod_action})

        self.bpod.softcode_handler_function = Session.softcode_handler

    def start_hardware(self):
        self.start_mixin_bpod()
        self.start_mixin_sound()

    @staticmethod
    def get_state_name(state_idx: int):
        if state_idx < len(Session.sequence):
            return f'{state_idx + 1:03d}_{Session.frequencies[Session.sequence[state_idx]]}'
        else:
            return 'exit'

    @staticmethod
    def softcode_handler(softcode: int) -> None:
        """log some information about the current state"""
        state_index = softcode - 1
        frequency_index = Session.sequence[state_index]
        frequency = Session.frequencies[frequency_index]
        n_states = len(Session.sequence)
        logger.info(f'{state_index + 1:03d}/{n_states}: {frequency:5d} Hz')

    def get_state_machine(self, seed: int) -> StateMachine:
        # generate shuffled sequence, seeded with state machine number
        Session.sequence = np.repeat(np.arange(len(self.frequencies)), self.repetitions)
        np.random.seed(seed)
        np.random.shuffle(Session.sequence)

        # build state machine
        sma = StateMachine(self.bpod)
        for state_idx, frequency_idx in enumerate(Session.sequence):
            sma.add_state(
                state_name=self.get_state_name(state_idx),
                state_timer=self.task_params['d_sound'] + self.task_params['d_pause'],
                output_actions=[self.bpod.actions[f'freq_{frequency_idx}'], ('SoftCode', state_idx + 1)],
                state_change_conditions={'Tup': self.get_state_name(state_idx + 1)},
            )
        return sma

    def _run(self):
        logger.info('Sending spacers to BNC ports')
        self.send_spacers()

        sma = self.get_state_machine(1)
        self.bpod.send_state_machine(sma)
        self.bpod.run_state_machine(sma)
        self.bpod.session.current_trial.export()


if __name__ == '__main__':  # pragma: no cover
    kwargs = get_task_arguments()
    sess = Session(**kwargs)
    sess.run()

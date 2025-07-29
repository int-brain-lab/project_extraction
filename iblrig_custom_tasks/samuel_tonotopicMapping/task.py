import json

import pandas as pd
import numpy as np
import logging

from pybpodapi.state_machine import StateMachine

from iblrig import sound
from iblrig.base_tasks import BaseSession, BpodMixin
from iblrig.misc import get_task_arguments

logger = logging.getLogger('iblrig')


class Session(BpodMixin, BaseSession):
    protocol_name = 'samuel_tonotopicMapping'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.hardware_settings.device_sound.OUTPUT == 'harp', 'This task requires a Harp sound-card'
        assert self.task_params['n_freqs'] <= 31, 'Harp only supports up to 31 individual sounds'

        # define frequencies (log spaced from freq_0 to freq_1, rounded to nearest integer)
        self.frequencies = np.logspace(
            np.log10(self.task_params['freq_0']),
            np.log10(self.task_params['freq_1']),
            num=self.task_params['n_freqs'],
        )
        self.frequencies = np.round(self.frequencies).astype(int)

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
        for idx, harp_index in enumerate(self.indices):
            bpod_message = [ord("P"), harp_index]
            bpod_action = (module_port, self.bpod._define_message(self.bpod.sound_card, bpod_message))
            self.bpod.actions.update({f'play_{idx}': bpod_action})

    def start_hardware(self):
        self.start_mixin_bpod()
        self.start_mixin_sound()

    def get_state_machine(self, sequence):
        sma = StateMachine(self.bpod)
        # table = self.sequence_table[self.sequence_table['sequence'] == sequence].reindex()

        for i, frequency in enumerate(self.frequencies):
            sma.add_state(
                state_name=f"trigger_sound_{i}",
                state_timer=self.task_params['d_sound'] + self.task_params['d_pause'],
                output_actions=[self.bpod.actions[f'play_{i}']],
                state_change_conditions={
                    "Tup": f"trigger_sound_{i + 1}",
                },
            )
        sma.add_state(
            state_name=f"trigger_sound_{i + 1}",
            state_timer=0.0,
            state_change_conditions={"Tup": "exit"},
        )
        return sma

    def _run(self):
        logger.info("Sending spacers to BNC ports")
        self.send_spacers()

        sma = self.get_state_machine(1)
        self.bpod.send_state_machine(sma)
        self.bpod.run_state_machine(sma)



if __name__ == '__main__':  # pragma: no cover
    kwargs = get_task_arguments()
    sess = Session(**kwargs)
    sess.run()

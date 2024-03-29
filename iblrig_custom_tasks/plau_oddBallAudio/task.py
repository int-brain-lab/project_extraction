from pathlib import Path
import json

import numpy as np
import pandas as pd

from pybpodapi.protocol import StateMachine
from iblrig.base_tasks import BaseSession, BpodMixin
import iblrig.sound
from iblutil.util import Bunch
from iblrig.hardware import sound_device_factory

import iblrig.misc
import iblrig

task_parameter_file = Path(iblrig.__file__).parent.joinpath("base_choice_world_params.yaml")


class Session(BpodMixin, BaseSession):
    protocol_name = "plau_oddBallAudio"

    def __init__(self, *args, **kwargs):
        super(Session, self).__init__(*args, **kwargs)
        self.sound = {}
        self.sound['sd'], self.sound['samplerate'], self.sound['channels'] = sound_device_factory(output='harp')
        self.task_params.SEQUENCE_FILE = self.task_params.SEQUENCE_FILE or Path(__file__).parent.joinpath('sound_sequence.csv')
        self.task_params.FOLDER_SOUNDS = self.task_params.FOLDER_SOUNDS or Path(__file__).parent.joinpath('tones')
        self.sequence_table = pd.read_csv(self.task_params.SEQUENCE_FILE)

        # initialize the sounds table from the csv file
        # loop over all the sounds and 1) load them, 2) define bpod actions
        files_sounds = sorted(Path(self.task_params.FOLDER_SOUNDS).rglob("i*_*.bin"))
        assert len(files_sounds) > 0, f"No sound binary files found in {self.task_params.FOLDER_SOUNDS}"
        self.df_sounds = pd.DataFrame(Bunch({
            'names': [f.stem for f in files_sounds if ' usermeta' not in f.stem],
            'harp_indices': np.arange(len(files_sounds)) + 2,
            'bin_duration': np.zeros(len(files_sounds)),
        }))
        self.df_sounds['bpod_action'] = ""
        self.all_sounds = []
        for i, fsound in enumerate(files_sounds):
            s = np.fromfile(fsound, dtype=np.int32).astype(np.float32)
            s = s.reshape(int(s.size / 2), 2).astype(np.float32) / 2 ** 31
            s = s * self.task_params.HARP_AMPLITUDE / self.task_params.HARP_DUMP_TONE_AMPLITUDE
            self.all_sounds.append(s)
            sn = self.df_sounds.at[i, 'names']
            self.df_sounds.at[i, 'bin_duration'] = s.shape[0] / self.sound['samplerate']
            self.df_sounds.at[i, 'bpod_action'] = f'play_{sn}'
        # now relate the loaded sounds to the sequence table
        self.sequence_table = self.sequence_table.merge(self.df_sounds, left_on='sound_name', right_on='names',
                                                        how='left')
        assert np.sum(np.isnan(self.sequence_table["harp_indices"])) == 0, \
            f"Some sound files are missing from {self.task_params.FOLDER_SOUNDS}"
        #  this contains the actual delay for the state machine: sound duration + delay + sequence change delay
        self.sequence_table['state_delay'] = (
                self.sequence_table['delay'] +
                np.r_[np.diff(self.sequence_table['sequence']), 0] * self.task_params.SEQUENCE_CHANGE_DELAY
        )

    def start_mixin_sound(self):
        """
        We overload the choice world sound configuration to use our custom set of sounds
        :return:
        """
        assert self.sound['samplerate'] == 96000
        # once all sounds are loaded, send them over to the harp sound card
        iblrig.sound.configure_sound_card(
            sounds=self.all_sounds,
            indexes=list(self.df_sounds['harp_indices']),
            sample_rate=self.sound['samplerate'],
        )
        # now we can register the sounds in the bpod actions
        sound_table = self.sequence_table.groupby('sound_name').agg(
            harp_indices=pd.NamedAgg(column='harp_indices', aggfunc='first'),
            bpod_action=pd.NamedAgg(column='bpod_action', aggfunc='first'),
        )
        for i, sound in sound_table.iterrows():
            bpod_message = [ord("P"), sound.harp_indices]
            bpod_action = ('Serial3', self.bpod._define_message(self.bpod.sound_card, bpod_message))
            self.bpod.actions.update({f'play_{sound.name}': bpod_action})

    def start_hardware(self):
        self.start_mixin_bpod()
        self.start_mixin_sound()


    def get_state_machine(self, sequence):
        sma = StateMachine(self.bpod)
        table = self.sequence_table[self.sequence_table['sequence'] == sequence].reindex()
        for i, rec in table.iterrows():
            # the first state triggers the sound and detects the upgoing front to move
            sma.add_state(
                state_name=f"trigger_sound_{i}",
                state_timer=rec.bin_duration,
                output_actions=[self.bpod.actions[rec.bpod_action]],
                state_change_conditions={
                    "BNC2High": f"play_sound_{i}",
                    "Tup": f"delay_{i}",
                },
            )
            # this states marks the begin and end of the sound played
            sma.add_state(
                state_name=f"play_sound_{i}",
                state_timer=0,
                output_actions=[("BNC1", 255)],
                state_change_conditions={
                    "BNC2Low": f"delay_{i}",
                },
            )
            # this state is the delay between the end of the sound and the next sound
            sma.add_state(
                state_name=f"delay_{i}",
                state_timer=rec.state_delay,
                state_change_conditions={
                    "Tup": f"trigger_sound_{i + 1}",
                },
            )

        sma.add_state(
            state_name=f"trigger_sound_{i + 1}",
            state_timer=0.0,
            state_change_conditions={"Tup": "exit"},
        )
        return sma, table

    def _run(self):
        # ideally we should populate the sound duration from the files read
        self.logger.info("Sending spacers to BNC ports")
        self.send_spacers()
        self.logger.info("Start the oddball protocol sound state machine")

        all_tables = []
        for seq in self.sequence_table['sequence'].unique():
            for _ in range(self.task_params.REPEAT_SEQUENCE):
                self.logger.info(f"Running sequence {seq}")
                sma, table = self.get_state_machine(seq)
                self.bpod.send_state_machine(sma)
                self.bpod.run_state_machine(sma)
                bpod_data = self.bpod.session.current_trial.export()
                # this fails when BNC is not connected
                table['bpod_timestamp'] = np.NaN
                bnc_high = bpod_data['Events timestamps'].get('BNC2High', [])
                n_bnc_high = len(bnc_high)
                n_bnc_low = len(bpod_data['Events timestamps'].get('BNC2Low', []))
                if not (n_bnc_high == n_bnc_low == table.shape[0]):
                    self.logger.warning("BNC2High and BNC2Low do not match the number of sounds played, check the BNC connection"
                                "from the sound card to the TTL I/O In2 port on the Bpod")
                table['bpod_timestamp'] = bnc_high
                all_tables.append(table)
                # Dump and save the bpod trial
                with open(self.paths['DATA_FILE_PATH'], 'a') as fp:
                    fp.write(json.dumps(dict(behavior_data=bpod_data)) + '\n')
                if self.paths.SESSION_FOLDER.joinpath('.stop').exists():
                    break
            if self.paths.SESSION_FOLDER.joinpath('.stop').exists():
                self.paths.SESSION_FOLDER.joinpath('.stop').unlink()
                self.logger.critical('Graceful exit')
                break
        self.sequence_table.to_parquet(self.paths.SESSION_RAW_DATA_FOLDER.joinpath('_plau_oddBallSoundsTable.pqt'))


if __name__ == "__main__":  # pragma: no cover
    kwargs = iblrig.misc.get_task_arguments()
    sess = Session(**kwargs)
    sess.run()

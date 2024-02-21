"""The video protocol simply plays a single video file.

TODO saves to parquet; would require changes to raw data loaders and behaviour copier class
TODO Add custom task list to Session class
"""
import time
from pathlib import Path
from collections import defaultdict
import logging

import pandas as pd
from pybpodapi.protocol import Bpod

import iblrig.misc
from iblrig.base_tasks import BpodMixin

_logger = logging.getLogger(__name__)

# this allows the CI and automated tests to import the file and make sure it is valid without having vlc
try:
    import vlc
except ModuleNotFoundError:
    _logger.error(f'VLC not installed. Please install VLC to use this task. {__file__}')


class Player:
    """A VLC player."""
    def __init__(self, rate=1, logger=None):
        self._instance = vlc.Instance(['--video-on-top'])
        self._player = self._instance.media_player_new()
        self._player.set_fullscreen(True)
        self._player.set_rate(rate)
        self._media = None
        self.events = defaultdict(list)
        self.logger = logger or _logger
        em = self._player.event_manager()
        for event in (vlc.EventType.MediaPlayerPlaying, vlc.EventType.MediaPlayerEndReached):
            em.event_attach(event, self._record_event)

    def _record_event(self, event):
        """VLC event callback."""
        self.logger.debug('%s', event.type)
        # Have to convert to str as object pointer may change
        self.events[str(event.type).split('.')[-1]].append(time.time())

    def play(self, path):
        """Play a video.

        Parameters
        ----------
        path : str
            A full path to a video file.
        """
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self._media = self._instance.media_new(path)
        self._player.set_media(self._media)
        self._player.play()

    def replay(self):
        """Replay the same media."""
        self._player.set_media(self._player.get_media())
        self._player.play()

    def stop(self):
        """Stop and close the player."""
        self._player.stop()

    @property
    def is_playing(self):
        """bool: True if media loaded and playing."""
        return bool(self._player.get_media() and self._player.is_playing())

    @property
    def is_started(self):
        """bool: True is media playback was initiated and playback has not ended."""
        starts = self.events.get('MediaPlayerPlaying', [])
        # print(f'len starts: {len(starts)}; dt = {(starts or [0])[-1] > ends[-1]}')
        return bool(len(starts) > 0 and starts[-1] > (self.get_ended_time() or 0))

    def get_ended_time(self, repeat=-1):
        """
        Return the time the video player reached the end of the playing media.

        Parameters
        ----------
        repeat : int
            The end time of the nth repeat. Defaults to last.

        Returns
        -------
        float
            The time at which the MediaPlayerEndReached event was recorded.
        """
        ends = self.events.get('MediaPlayerEndReached', [])
        if not ends:
            return
        elif repeat == -1 or len(ends) > repeat:
            return ends[repeat]


class Session(BpodMixin):
    """Play a single video."""

    protocol_name = '_sp_passiveVideo'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.hardware_settings.get('MAIN_SYNC', False):
            raise NotImplementedError('Recording frame2ttl on Bpod not yet implemented')
        self.paths.DATA_FILE_PATH = self.paths.DATA_FILE_PATH.with_name('_sp_taskData.raw.pqt')
        self.video = None
        self.trial_num = -1
        columns = ['intervals_0', 'intervals_1']
        self.data = pd.DataFrame(pd.NA, index=range(self.task_params.NREPEATS), columns=columns)

    def save(self):
        self.logger.info('Saving data')
        if self.video:
            data = pd.concat([self.data, pd.DataFrame.from_dict(self.video.events)], axis=1)
            data.to_parquet(self.paths.DATA_FILE_PATH)
        self.paths.SESSION_FOLDER.joinpath('transfer_me.flag').touch()

    def start_hardware(self):
        self.start_mixin_bpod()  # used for protocol spacer only
        self.video = Player(logger=self.logger)

    def next_trial(self):
        """Start the next trial."""
        self.trial_num += 1
        self.data.at[self.trial_num, 'intervals_0'] = time.time()
        if self.trial_num == 0:
            self.logger.info('Starting video %s', self.task_params.VIDEO)
            self.video.play(self.task_params.VIDEO)
        else:
            self.logger.debug('Trial #%i: Replaying video', self.trial_num + 1)
            assert self.video
            self.video.replay()

    def _set_bpod_out(self, val):
        """Set Bpod BNC1 output state."""
        if isinstance(val, bool):
            val = 255 if val else 128
        self.bpod.manual_override(Bpod.ChannelTypes.OUTPUT, Bpod.ChannelNames.BNC, channel_number=1, value=val)

    def _run(self):
        """This is the method that runs the video."""
        # make the bpod send spacer signals to the main sync clock for protocol discovery
        self.send_spacers()
        for rep in range(self.task_params.NREPEATS):  # Main loop
            self.next_trial()
            self._set_bpod_out(True)
            # TODO c.f. MediaListPlayerPlayed event
            while not self.video.is_started:
                ...  # takes time to actually start playback
            while self.video.is_playing or (end_time := self.video.get_ended_time(rep)) is None:
                time.sleep(0.05)
            # trial finishes when playback finishes
            self._set_bpod_out(False)
            self.session_info.NTRIALS += 1
            self.data.at[self.trial_num, 'intervals_1'] = time.time()
            dt = self.task_params.ITI_DELAY_SECS - (time.time() - end_time)
            self.logger.debug(f'dt = {dt}')
            # wait to achieve the desired ITI duration
            if dt > 0:
                time.sleep(dt)

        self.video.stop()  # close player
        self.save()


if __name__ == '__main__':  # pragma: no cover
    # python .\iblrig_tasks\_sp_passiveVideo\task.py --subject mysubject
    kwargs = iblrig.misc.get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

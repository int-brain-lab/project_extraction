import logging

import cv2
import numpy as np
import pandas as pd
import one.alf.io as alfio
import ibldsp.utils
from iblutil.spacer import Spacer

from ibllib.pipes.base_tasks import BehaviourTask
from ibllib.exceptions import SyncBpodFpgaException
from ibllib.io.video import get_video_meta
from ibllib.io.extractors.ephys_fpga import get_protocol_period, get_sync_fronts
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap
from ibllib.io.extractors.mesoscope import plot_timeline

_logger = logging.getLogger('ibllib').getChild(__name__)
_logger.setLevel(logging.DEBUG)


class PassiveVideoTimeline(BehaviourTask):
    """Extraction task for _sp_passiveVideo protocol."""
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {}
        signature['input_files'] = [
            # NB: _sp_taskData.raw is currently saved under the _iblrig_taskData.raw dataset type in Alyx
            ('_sp_taskData.raw.*', self.collection, True, True),
            ('_sp_video.raw.*', self.collection, False, True),
            ('_iblrig_taskSettings.raw.*', self.collection, True, True),
            (f'_{self.sync_namespace}_DAQdata.raw.npy', self.sync_collection, True),
            (f'_{self.sync_namespace}_DAQdata.timestamps.npy', self.sync_collection, True),
            (f'_{self.sync_namespace}_DAQdata.meta.json', self.sync_collection, True),
        ]
        signature['output_files'] = [('_sp_video.times.npy', self.output_collection, True),]
        return signature

    def generate_sync_sequence(seed=1234, ns=3600, res=8):
        """Generate the sync square frame colour sequence.

        Instead of changing each frame, the video sync square flips between black and white
        in a particular sequence defined within this function (in random multiples of res).

        Parameters
        ----------
        ns : int
            Related to the length in frames of the sequence (n_frames = ns * res).
        res : int
            The minimum number of sequential frames in each colour state. The N sequential frames
            is a multiple of this number.
        seed : int, optional
            The numpy random seed integer, by default 1234

        Returns
        -------
        numpy.array
            An integer array of sync square states (one per frame) where 0 represents black and 1
            represents white.
        """
        state = np.random.get_state()
        try:
            np.random.seed(1234)
            seq = np.tile(np.random.random(ns), (res, 1)).T.flatten()
            return (seq > .5).astype(np.int8)
        finally:
            np.random.set_state(state)

    def load_sync_sequence_from_video(self, video_file, location='bottom right', size=(5, 5)):
        cap = cv2.VideoCapture(str(video_file))
        sequence = []
        location = location.casefold().split()
        loc_map = {
            'top': slice(0, size[1]), 'bottom': slice(-size[1], None),
            'left': slice(0, size[0]), 'right': slice(-size[0], None)}
        idx = tuple(loc_map[x] for x in reversed(location))  # h, w
        success = True
        while success:
            success, frame = cap.read()
            if success:
                # Find the sync square in the video frame
                pixel = np.mean(frame[idx])
                sequence.append(int(pixel > 128))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert len(sequence) == length, 'sequence length does not match video length'
        return np.array(sequence)

    def extract_frame_times(self, save=True, frame_rate=None, display=False, **kwargs):
        """Extract the Bpod trials data and Timeline acquired signals.

        Sync requires three steps:
            1. Find protocol period using spacers
            2. Find each video repeat with Bpod out
            3. Find frame times with frame2ttl

        Parameters
        ----------
        save : bool, optional
            Whether to save the video frame times to file, by default True.
        frame_rate : int, optional
            The frame rate of the video presented, by default 30.
        display : bool, optional
            When true, plot the aligned frame times. By default False.

        Returns
        -------
        numpy.array
            The extracted frame times where N rows represent the number of frames and M columns
            represent the number of video repeats. The exact number of frames is not known and
            NaN values represent shorter video repeats.
        pathlib.Path
            The file path of the saved video times, or None if save=False.

        Raises
        ------
        ValueError
            The `protocol_number` property is None and no `tmin` or `tmax` values were passed as
            keyword arguments.
        SyncBpodFpgaException
            The synchronization of frame times was likely unsuccessful.
        """
        DEFAULT_FRAME_RATE = 30
        _, (p,), _ = self.input_files[0].find_files(self.session_path)
        # Load raw data
        proc_data = pd.read_parquet(p)
        sync_path = self.session_path / self.sync_collection
        self.timeline = alfio.load_object(sync_path, 'DAQdata', namespace='timeline')
        sync, chmap = load_timeline_sync_and_chmap(sync_path, timeline=self.timeline)

        # Attempt to get the frame rate from the video file if not provided
        if video_file := next(self.session_path.joinpath(self.collection).glob('_sp_video.raw.*'), None):
            video_meta = get_video_meta(video_file)
            if frame_rate is not None and frame_rate != video_meta.fps:
                _logger.warning(
                    'Frame rate mismatch: %.2f Hz (video) vs %.2f Hz (provided). Using %.2f Hz',
                    video_meta.fps, frame_rate, video_meta.fps)
            else:
                _logger.debug('Video frame rate: %.2f Hz', video_meta.fps)
            frame_rate = video_meta.fps
        else:
            video_meta = None
            frame_rate = frame_rate or DEFAULT_FRAME_RATE
            _logger.warning('Video not found. Assumed video frame rate: %.2f Hz', frame_rate)
        Fs = self.timeline['meta']['daqSampleRate']
        assert Fs > frame_rate * 1.5, 'DAQ sample rate must be higher than video frame rate'

        bpod = get_sync_fronts(sync, chmap['bpod'])
        # Get the spacer times for this protocol
        if any(arg in kwargs for arg in ('tmin', 'tmax')):
            tmin, tmax = kwargs.get('tmin'), kwargs.get('tmax')
        elif self.protocol_number is None:
            raise ValueError('Protocol number not defined')
        else:
            # The spacers are TTLs generated by Bpod at the start of each protocol
            tmin, tmax = get_protocol_period(self.session_path, self.protocol_number, bpod)
            tmin += (Spacer().times[-1] + Spacer().tup + 0.05)  # exclude spacer itself

        # Remove unnecessary data from sync
        selection = np.logical_and(
            sync['times'] <= (tmax if tmax is not None else sync['times'][-1]),
            sync['times'] >= (tmin if tmin is not None else sync['times'][0]),
        )
        sync = alfio.AlfBunch({k: v[selection] for k, v in sync.items()})
        bpod = get_sync_fronts(sync, chmap['bpod'])
        _logger.debug('Protocol period from %.2fs to %.2fs (~%.0f min duration)',
                      *sync['times'][[0, -1]], np.diff(sync['times'][[0, -1]]) / 60)

        # For each period of video playback the Bpod should output voltage HIGH
        bpod_rep_starts, = np.where(bpod['polarities'] == 1)
        _logger.info('N video repeats: %i; N Bpod pulses: %i', len(proc_data), len(bpod_rep_starts))
        assert len(bpod_rep_starts) == len(proc_data)

        # These durations are longer than video actually played and will be cut down after
        durations = (proc_data['intervals_1'] - proc_data['intervals_0']).values
        max_n_frames = np.max(np.ceil(durations * frame_rate).astype(int))
        n_frames = video_meta.length if video_meta else max_n_frames
        frame_times = np.full((n_frames, len(proc_data)), np.nan)

        sync_sequence = kwargs.get('sync_sequence', self.generate_sync_sequence())
        for i, rep in proc_data.iterrows():
            # Get the frame2ttl times for the video presentation
            idx = bpod_rep_starts[i]
            start = bpod['times'][idx]
            try:
                end = bpod['times'][idx + 1]
            except IndexError:
                _logger.warning('Final Bpod LOW missing')
                end = start + (rep['intervals_1'] - rep['intervals_0'])
            f2ttl = get_sync_fronts(sync, chmap['frame2ttl'])
            ts = f2ttl['times'][np.logical_and(f2ttl['times'] >= start, f2ttl['times'] < end)]
            if video_meta:
                _logger.debug('Repeat %i: video duration: %.2fs, f2ttl duration: %.2f',
                              i, video_meta.duration.seconds, ts[-1] - ts[0])

            # video_runtime is the video length reported by VLC.
            # As it was added later, the less accurate media player timestamps may be used if the former is not available
            duration = rep.get('video_runtime') or (rep['MediaPlayerEndReached'] - rep['MediaPlayerPlaying'])
            # Start the sync sequence times at the start of the first frame2ttl flip (ts[0]) as this makes syncing more
            # performant because the offset is small
            sequence_times = np.arange(0, duration, 1 / frame_rate)
            sequence_times += ts[0]
            # The below assertion could be caused by an incorrect frame rate or sync sequence
            assert sequence_times.size <= sync_sequence.size, 'video duration appears longer than sync sequence'
            if len(sequence_times) > n_frames:
                # Duration rounding error may lead to 1 frame time too many
                sequence_times = sequence_times[:n_frames]
            # Keep only the part of the sequence that was shown
            x = sync_sequence[:len(sequence_times)]
            # Find change points (black <-> white indices)
            x, = np.where(np.abs(np.diff(x)))
            # Include first frame as change point
            x = np.r_[0, x + 1]
            # Synchronize the two by aligning flip times
            DRIFT_THRESHOLD_PPM = 50
            fcn, drift = ibldsp.utils.sync_timestamps(sequence_times[x], ts, linear=True)
            # Log any major drift or raise if too large
            if np.abs(drift) > DRIFT_THRESHOLD_PPM * 2 and x.size - ts.size > 100:
                raise SyncBpodFpgaException(
                    f'sync cluster f*ck: drift = {drift:.2f}, changepoint difference = {x.size - ts.size}')
            elif np.abs(drift) > DRIFT_THRESHOLD_PPM:
                _logger.warning('Frame synchronization shows values greater than %.2g ppm', DRIFT_THRESHOLD_PPM)
            _logger.debug('Frame synchronization drift: %.2f ppm', drift)

            # Get the frame times in timeline time
            frame_times[:len(sequence_times), i] = fcn(sequence_times)

        # Trim down to length of repeat with most frames
        if np.any(empty := np.all(np.isnan(frame_times), axis=1)):
            frame_times = frame_times[:np.where(empty)[0][0], :]

        if display:
            import matplotlib.pyplot as plt
            from matplotlib import colormaps
            from ibllib.plots import squares
            plot_timeline(self.timeline, channels=['bpod', 'frame2ttl'])
            _, ax = plt.subplots(2, 1, sharex=True)
            squares(f2ttl['times'], f2ttl['polarities'], ax=ax[0])
            ax[0].set_yticks((-1, 1))
            ax[0].title.set_text('frame2ttl')
            cmap = colormaps['plasma']
            for i, times in enumerate(frame_times.T):
                # Plot the sync sequence and sync'd frame times
                rgba = cmap(i / frame_times.shape[1])
                ax[1].plot(times, sync_sequence[:len(times)], c=rgba, label=f'{i}')
                # Plot the f2ttl values
                idx = bpod_rep_starts[i]
                start = bpod['times'][idx]
                try:
                    end = bpod['times'][idx + 1]
                except IndexError:
                    end = start + (rep['intervals_1'] - rep['intervals_0'])
                mask = np.logical_and(f2ttl['times'] >= start, f2ttl['times'] < end)
                squares(f2ttl['times'][mask], f2ttl['polarities'][mask],
                        yrange=[0, 1], ax=ax[1], linestyle=':', color='k')
            ax[1].title.set_text('aligned sync square sequence')
            ax[1].set_yticks((0, 1))
            ax[1].set_yticklabels([-1, 1])
            plt.legend(markerfirst=False, title='repeat #', loc='upper right', facecolor='white')

            # Check the sync sequence from the video
            if video_file:
                observed = self.load_sync_sequence_from_video(video_file)
                _, ax = plt.subplots(2, 1, sharex=True)
                ax[0].title.set_text('generated sync square sequence')
                ax[0].plot(sync_sequence[:observed.size])
                ax[1].title.set_text('observed sync square sequence')
                ax[1].plot(observed)

                # resample the f2ttl sequence to the frame times
                # tts = ts-ts[0]
                # from scipy import interpolate
                # interp = interpolate.interp1d(tts, pol, kind = "nearest")
                # ttts = np.arange(0, tts[-1], 1/frame_rate)
                # ax[2].plot(interp(ttts))
                # squares(ts-ts[0], pol, ax=ax[2])

            plt.show()

        if save:
            filename = self.session_path.joinpath(self.output_collection, '_sp_video.times.npy')
            filename.parent.mkdir(exist_ok=True, parents=True)
            np.save(filename, frame_times)
            out_files = [filename]
        else:
            out_files = []

        return {'video_times': frame_times}, out_files

    def run_qc(self, **_):
        raise NotImplementedError

    def _run(self, save=True, **kwargs):
        _, output_files = self.extract_frame_times(save=save, **kwargs)
        return output_files

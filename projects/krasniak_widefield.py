from ibllib.io.extractors.widefield import Widefield as BaseWidefield
from pathlib import Path
from ibllib.io.extractors.ephys_fpga import get_main_probe_sync, get_sync_fronts
from ibllib.plots.snapshot import ReportSnapshot
import numpy as np
import ibllib.exceptions as err
from ibllib.pipes.base_tasks import WidefieldTask
import ibldsp as dsp
import logging
from one.api import ONE
from ibllib.io.extractors.camera import get_video_length

_logger = logging.getLogger('ibllib')

try:
    from labcams.io import parse_cam_log
except ImportError:
    _logger.warning('labcams not installed')

try:
    import wfield.cli as wfield_cli
except ImportError:
    _logger.warning('wfield not installed')


DEFAULT_WIRING_MAP = {
    3: 470,
    2: 405
}


class Widefield(BaseWidefield):

    def preprocess(self, fs=30, functional_channel=1, nbaseline_frames=30, k=200, nchannels=2):

        # No motion correction for Krasniak
        # wfield_cli._motion(str(self.data_path), nchannels=nchannels, plot_ext='.png')

        # COMPUTE AVERAGE FOR BASELINE
        wfield_cli._baseline(str(self.data_path), nbaseline_frames, nchannels=nchannels)
        # DATA REDUCTION
        wfield_cli._decompose(str(self.data_path), k=k, nchannels=nchannels)
        # HAEMODYNAMIC CORRECTION
        # check if it is 2 channel
        dat = wfield_cli.load_stack(str(self.data_path), nchannels=nchannels)
        if dat.shape[1] == 2:
            del dat
            wfield_cli._hemocorrect(str(self.data_path), fs=fs, functional_channel=functional_channel, plot_ext='.png')

    def sync_timestamps(self, save=False, save_paths=None, **kwargs):

        if save and save_paths:
            assert len(save_paths) == 3, 'Must provide save_path as list with 3 paths'
            for save_path in save_paths:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        filepath = next(self.data_path.glob('*.camlog'))
        fpga_sync, chmap = get_main_probe_sync(self.session_path)
        bpod = get_sync_fronts(fpga_sync, chmap['bpod'])
        logdata, led, sync, ncomm = parse_cam_log(filepath, readTeensy=True)
        if bpod.times.size == 0:
            raise err.SyncBpodFpgaException('No Bpod event found in FPGA. No behaviour extraction. '
                                            'Check channel maps.')

        # Check that the no. of syncs from bpod and teensy match
        try:
            assert len(bpod['times']) == len(sync), 'Number of detected sync pulses on bpod and teensy do not match,' \
                                                    'trying laser ttl'
        except AssertionError:
            bpod = get_sync_fronts(fpga_sync, chmap['laser_ttl'])
            assert len(bpod['times']) == len(sync), 'Number of detected sync pulses on laser and teensy do not match'

        # Check the number of led frames matched the number of video frames
        assert led.frame.is_monotonic_increasing
        video_path = next(self.data_path.glob('imaging.frames.mov'), None)
        if not video_path:
            one = ONE()
            datasets = one.list_datasets(one.path2eid(self.session_path), filename='imaging.frames.mov')
            url = one.path2url(self.session_path / datasets[0])
            video_length = get_video_length(url)
        else:
            video_length = get_video_length(video_path)

        diff = len(led) - video_length
        if diff < 0:
            raise ValueError('More frames than timestamps detected')
        if diff > 2:
            raise ValueError('Timestamps and frames differ by more than 2')

        led = led[0:video_length]

        def sync_times(bpod, sync, led, diff=0):
            # convert to seconds
            fcn, drift, iteensy, ifpga = dsp.utils.sync_timestamps(sync.timestamp.values / 1e3 - diff, bpod['times'],
                                                                   return_indices=True)

            _logger.debug(f'Widefield-FPGA clock drift: {drift} ppm')

            # Find led times that are outside of the sync pulses
            led_times = np.copy(led.timestamp.values)
            pre_times = led_times < np.min(sync.timestamp)
            post_times = led_times > np.max(sync.timestamp)
            led_times[pre_times] = np.nan
            led_times[post_times] = np.nan

            # Interpolate frames that lie within sync pulses timeframe
            widefield_times = fcn(led_times / 1e3)
            kp = ~np.isnan(widefield_times)
            # Extrapolate times that lie outside sync pulses timeframe (i.e before or after)
            pol = np.polyfit(led_times[kp] / 1e3, widefield_times[kp], 1)
            extrap_vals = np.polyval(pol, led.timestamp.values / 1e3)
            widefield_times[~kp] = extrap_vals[~kp]

            return widefield_times

        widefield_times = sync_times(bpod, sync, led)

        try:
            assert np.all(np.diff(widefield_times) > 0)
        except AssertionError:
            _logger.warning('Difference between timestamps not all > 0, trying to extract by first subtracting offset')
            diff = sync.timestamp.values[0] / 1e3 - bpod['times'][0]
            widefield_times = sync_times(bpod, sync, led, diff=diff)
            widefield_times = widefield_times - diff
            assert np.all(np.diff(widefield_times) > 0)

        # Now extract the LED channels and meta data
        # Load channel meta and wiring map
        channel_meta_map = self._channel_meta(kwargs.get('light_source_map'))
        channel_wiring = self._channel_wiring()
        channel_id = np.empty_like(led.led.values)

        for _, d in channel_wiring.iterrows():
            mask = led.led.values == d['LED']
            if np.sum(mask) == 0:
                raise err.WidefieldWiringException
            channel_id[mask] = channel_meta_map.get(channel_meta_map['wavelength'] == d['wavelength']).index[0]

        if save:
            save_time = save_paths[0] if save_paths else self.data_path.joinpath('timestamps.npy')
            save_led = save_paths[1] if save_paths else self.data_path.joinpath('led.npy')
            save_meta = save_paths[2] if save_paths else self.data_path.joinpath('led_properties.csv')
            save_paths = [save_time, save_led, save_meta]
            np.save(save_time, widefield_times)
            np.save(save_led, channel_id)
            channel_meta_map.to_csv(save_meta)

            return save_paths
        else:
            return widefield_times, channel_id, channel_meta_map


class WidefieldSyncKrasniak(WidefieldTask):
    priority = 60
    level = 1
    force = False
    job_size = 'small'
    signature = {
        'input_files': [('imaging.frames.mov', 'raw_widefield_data', True),
                        ('widefieldEvents.raw.camlog', 'raw_widefield_data', True),
                        ('_spikeglx_sync.channels.npy', 'raw_ephys_data', True),
                        ('_spikeglx_sync.polarities.npy', 'raw_ephys_data', True),
                        ('_spikeglx_sync.times.npy', 'raw_ephys_data', True)],
        'output_files': [('imaging.times.npy', 'alf/widefield', True),
                         ('imaging.imagingLightSource.npy', 'alf/widefield', True),
                         ('imagingLightSource.properties.htsv', 'alf/widefield', True)]
    }

    def _run(self):

        self.wf = Widefield(self.session_path)
        save_paths = [self.session_path.joinpath(sig[1], sig[0]) for sig in self.signature['output_files']]
        out_files = self.wf.sync_timestamps(save=True, save_paths=save_paths)

        # TODO QC

        return out_files


class WidefieldPreprocessKrasniak(WidefieldTask):

    priority = 80
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [('*.dat', self.device_collection, True),
                            ('widefieldEvents.raw.*', self.device_collection, True)],
            'output_files': [('widefieldChannels.frameAverage.npy', 'alf/widefield', True),
                             ('widefieldU.images.npy', 'alf/widefield', True),
                             ('widefieldSVT.uncorrected.npy', 'alf/widefield', True),
                             ('widefieldSVT.haemoCorrected.npy', 'alf/widefield', True)]
        }
        return signature

    def _run(self, upload_plots=True, **kwargs):
        self.wf = Widefield(self.session_path)
        _, out_files = self.wf.extract(save=True, extract_timestamps=False)

        if upload_plots:
            output_plots = []
            if self.wf.data_path.joinpath('hemodynamic_correction.png').exists():
                output_plots.append(self.wf.data_path.joinpath('hemodynamic_correction.png'))

            if len(output_plots) > 0:
                eid = self.one.path2eid(self.session_path)
                snp = ReportSnapshot(self.session_path, eid, one=self.one)
                snp.outputs = output_plots
                snp.register_images(widths=['orig'], function='wfield')

        return out_files

    def tearDown(self):
        super(WidefieldPreprocessKrasniak, self).tearDown()
        self.wf.remove_files()

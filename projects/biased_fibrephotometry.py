"""Extraction pipeline for Alejandro's learning_witten_dop project, task protocol _iblrig_tasks_FPChoiceWorld6.4.2"""
from inspect import getmembers, isfunction
import logging

import pandas as pd
import numpy as np
import scipy.interpolate

import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from one.alf.spec import is_session_path

from ibllib.io.extractors.base import BaseExtractor
from ibllib.io.raw_daq_loaders import load_channels_tdms, load_raw_daq_tdms
from ibllib.io.extractors.training_trials import GoCueTriggerTimes
from ibldsp.utils import rises, sync_timestamps
from iblutil.util import Bunch
from ibllib.io import raw_daq_loaders
from ibllib.qc.base import QC
from ibllib.pipes import base_tasks

_logger = logging.getLogger('ibllib').getChild(__name__.split('.')[-1])

"""Data extraction from fibrephotometry DAQ files.

Below is the expected folder structure for a fibrephotometry session:

    subject/
    ├─ 2021-06-30/
    │  ├─ 001/
    │  │  ├─ raw_photometry_data/
    │  │  │  │  ├─ _neurophotometrics_fpData.raw.pqt
    │  │  │  │  ├─ _neurophotometrics_fpData.channels.csv
    │  │  │  │  ├─ _mcc_DAQdata.raw.tdms

fpData.raw.pqt is a copy of the 'FPdata' file, the output of the Neuophotometrics Bonsai workflow.
fpData.channels.csv is table of frame flags for deciphering LED and GPIO states. The default table,
copied from the Neurophotometrics manual can be found in iblscripts/deploy/fppc/
_mcc_DAQdata.raw.tdms is the DAQ tdms file, containing the pulses from bpod and from the neurophotometrics system

Neurophotometrics FP3002 specific information.
The light source map refers to the available LEDs on the system.
The flags refers to the byte encoding of led states in the system.
"""
LIGHT_SOURCE_MAP = {
    'color': ['None', 'Violet', 'Blue', 'Green'],
    'wavelength': [0, 415, 470, 560],
    'name': ['None', 'Isosbestic', 'GCaMP', 'RCaMP'],
}

NEUROPHOTOMETRICS_LED_STATES = {
    'Condition': {
        0: 'No additional signal',
        1: 'Output 1 signal HIGH',
        2: 'Output 0 signal HIGH',
        3: 'Stimulation ON',
        4: 'GPIO Line 2 HIGH',
        5: 'GPIO Line 3 HIGH',
        6: 'Input 1 HIGH',
        7: 'Input 0 HIGH',
        8: 'Output 0 signal HIGH + Stimulation',
        9: 'Output 0 signal HIGH + Input 0 signal HIGH',
        10: 'Input 0 signal HIGH + Stimulation',
        11: 'Output 0 HIGH + Input 0 HIGH + Stimulation',
    },
    'No LED ON': {0: 0, 1: 8, 2: 16, 3: 32, 4: 64, 5: 128, 6: 256, 7: 512, 8: 48, 9: 528, 10: 544, 11: 560},
    'L415': {0: 1, 1: 9, 2: 17, 3: 33, 4: 65, 5: 129, 6: 257, 7: 513, 8: 49, 9: 529, 10: 545, 11: 561},
    'L470': {0: 2, 1: 10, 2: 18, 3: 34, 4: 66, 5: 130, 6: 258, 7: 514, 8: 50, 9: 530, 10: 546, 11: 562},
    'L560': {0: 4, 1: 12, 2: 20, 3: 36, 4: 68, 5: 132, 6: 260, 7: 516, 8: 52, 9: 532, 10: 548, 11: 564}
}

CHANNELS = pd.DataFrame.from_dict(NEUROPHOTOMETRICS_LED_STATES)
DAQ_CHMAP = {"photometry": 'AI0', 'bpod': 'AI1'}
V_THRESHOLD = 3


def sync_photometry_to_daq(vdaq, fs, df_photometry, chmap=DAQ_CHMAP, v_threshold=V_THRESHOLD):
    """
    :param vdaq: dictionary of daq traces.
    :param fs: sampling frequency
    :param df_photometry:
    :param chmap:
    :param v_threshold:
    :return:
    """
    # here we take the flag that is the most common
    daq_frames, tag_daq_frames = read_daq_timestamps(vdaq=vdaq, v_threshold=v_threshold)
    nf = np.minimum(tag_daq_frames.size, df_photometry['Input0'].size)

    # we compute the framecounter for the DAQ, and match the bpod up state frame by frame for different shifts
    # the shift that minimizes the mismatch is usually good
    df = np.median(np.diff(df_photometry['Timestamp']))
    fc = np.cumsum(np.round(np.diff(daq_frames) / fs / df).astype(np.int32)) - 1  # this is a daq frame counter
    fc = fc[fc < (nf - 1)]
    max_shift = 300
    error = np.zeros(max_shift * 2 + 1)
    shifts = np.arange(-max_shift, max_shift + 1)
    for i, shift in enumerate(shifts):
        rolled_fp = np.roll(df_photometry['Input0'].values[fc], shift)
        error[i] = np.sum(np.abs(rolled_fp - tag_daq_frames[:fc.size]))
    # a negative shift means that the DAQ is ahead of the photometry and that the DAQ misses frame at the beginning
    frame_shift = shifts[np.argmax(-error)]
    if np.sign(frame_shift) == -1:
        ifp = fc[np.abs(frame_shift):]
    elif np.sign(frame_shift) == 0:
        ifp = fc
    elif np.sign(frame_shift) == 1:
        ifp = fc[:-np.abs(frame_shift)]
    t_photometry = df_photometry['Timestamp'].values[ifp]
    t_daq = daq_frames[:ifp.size] / fs
    # import matplotlib.pyplot as plt
    # plt.plot(shifts, -error)
    fcn_fp2daq = scipy.interpolate.interp1d(t_photometry, t_daq, fill_value='extrapolate')
    drift_ppm = (np.polyfit(t_daq, t_photometry, 1)[0] - 1) * 1e6
    if drift_ppm > 120:
        _logger.warning(f"drift photometry to DAQ PPM: {drift_ppm}")
    else:
        _logger.info(f"drift photometry to DAQ PPM: {drift_ppm}")
    # here is a bunch of safeguards
    assert np.unique(np.diff(df_photometry['FrameCounter'])).size == 1  # checks that there are no missed frames on photo
    assert np.abs(frame_shift) <= 5  # it's always the end frames that are missing
    assert np.abs(drift_ppm) < 60
    ts_daq = fcn_fp2daq(df_photometry['Timestamp'].values)  # those are the timestamps in daq time
    return ts_daq, fcn_fp2daq, drift_ppm


def read_daq_voltage(daq_file, chmap=DAQ_CHMAP):
    channel_names = [c.name for c in load_raw_daq_tdms(daq_file)['Analog'].channels()]
    assert all([v in channel_names for v in chmap.values()]), "Missing channel"
    vdaq, fs = load_channels_tdms(daq_file, chmap=chmap)
    vdaq = {k: v - np.median(v) for k, v in vdaq.items()}
    return vdaq, fs


def read_daq_timestamps(vdaq, v_threshold=V_THRESHOLD):
    """
    From a tdms daq file, extracts the photometry frames and their tagging.
    :param vsaq: dictionary of the voltage traces from the DAQ. Each item has a key describing
    the channel as per the channel map, and contains a single voltage trace.
    :param v_threshold:
    :return:
    """
    daq_frames = rises(vdaq['photometry'], step=v_threshold, analog=True)
    if daq_frames.size == 0:
        daq_frames = rises(-vdaq['photometry'], step=v_threshold, analog=True)
        _logger.warning(f'No photometry pulses detected, attempting to reverse voltage and detect again,'
                        f'found {daq_frames.size} in reverse voltage. CHECK YOUR FP WIRING TO THE DAQ !!')
    tagged_frames = vdaq['bpod'][daq_frames] > v_threshold
    return daq_frames, tagged_frames


def check_timestamps(daq_file, photometry_file, tolerance=20, chmap=DAQ_CHMAP, v_threshold=V_THRESHOLD):
    """
    Reads data file and checks that the number of timestamps check out with a tolerance of n_frames
    :param daq_file:
    :param photometry_file:
    :param tolerance: number of acceptable missing frames between the daq and the photometry file
    :param chmap:
    :param v_threshold:
    :return: None
    """
    df_photometry = pd.read_csv(photometry_file)
    v, fs = read_daq_voltage(daq_file=daq_file, chmap=chmap)
    daq_frames, _ = read_daq_timestamps(vdaq=v, v_threshold=v_threshold)
    assert (daq_frames.shape[0] - df_photometry.shape[0]) < tolerance
    _logger.info(f"{daq_frames.shape[0] - df_photometry.shape[0]} frames difference, "
                 f"{'/'.join(daq_file.parts[-2:])}: {daq_frames.shape[0]} frames, "
                 f"{'/'.join(photometry_file.parts[-2:])}: {df_photometry.shape[0]}")


class BaseFibrePhotometry(BaseExtractor):
    """
        FibrePhotometry(self.session_path, collection=self.collection)
    """
    save_names = ('photometry.signal.pqt')
    var_names = ('df_out')

    def __init__(self, *args, collection='raw_photometry_data', **kwargs):
        """An extractor for all Neurophotometrics fibrephotometry data"""
        self.collection = collection
        super().__init__(*args, **kwargs)

    @staticmethod
    def _channel_meta(light_source_map=None):
        """
        Return table of light source wavelengths and corresponding colour labels.

        Parameters
        ----------
        light_source_map : dict
            An optional map of light source wavelengths (nm) used and their corresponding colour name.

        Returns
        -------
        pandas.DataFrame
            A sorted table of wavelength and colour name.
        """
        light_source_map = light_source_map or LIGHT_SOURCE_MAP
        meta = pd.DataFrame.from_dict(light_source_map)
        meta.index.rename('channel_id', inplace=True)
        return meta

    def _extract(self, light_source_map=None, collection=None, regions=None, **kwargs):
        """

        Parameters
        ----------
        regions: list of str
            The list of regions to extract. If None extracts all columns containing "Region". Defaults to None.
        light_source_map : dict
            An optional map of light source wavelengths (nm) used and their corresponding colour name.
        collection: str / pathlib.Path
            An optional relative path from the session root folder to find the raw photometry data.
            Defaults to `raw_photometry_data`

        Returns
        -------
        numpy.ndarray
            A 1D array of signal values.
        numpy.ndarray
            A 1D array of ints corresponding to the active light source during a given frame.
        pandas.DataFrame
            A table of intensity for each region, with associated times, wavelengths, names and colors
        """
        collection = collection or self.collection
        fp_data = alfio.load_object(self.session_path / collection, 'fpData')
        ts = self.extract_timestamps(fp_data['raw'], **kwargs)

        # Load channels and
        channel_meta_map = self._channel_meta(kwargs.get('light_source_map'))
        led_states = fp_data.get('channels', pd.DataFrame(NEUROPHOTOMETRICS_LED_STATES))
        led_states = led_states.set_index('Condition')
        # Extract signal columns into 2D array
        regions = regions or [k for k in fp_data['raw'].keys() if 'Region' in k]
        out_df = fp_data['raw'].filter(items=regions, axis=1).sort_index(axis=1)
        out_df['times'] = ts
        out_df['wavelength'] = np.nan
        out_df['name'] = ''
        out_df['color'] = ''
        # Extract channel index
        states = fp_data['raw'].get('LedState', fp_data['raw'].get('Flags', None))
        for state in states.unique():
            ir, ic = np.where(led_states == state)
            if ic.size == 0:
                continue
            for cn in ['name', 'color', 'wavelength']:
                out_df.loc[states == state, cn] = channel_meta_map.iloc[ic[0]][cn]
        return out_df

    def extract_timestamps(self, fp_data, **kwargs):
        """Extract the photometry.timestamps array.

        This depends on the DAQ and task synchronization protocol.

        Parameters
        ----------
        fp_data : dict
            A Bunch of raw fibrephotometry data, with the keys ('raw', 'channels').

        Returns
        -------
        numpy.ndarray
            An array of timestamps, one per frame.
        """
        daq_file = next(self.session_path.joinpath(self.collection).glob('*.tdms'))
        vdaq, fs = read_daq_voltage(daq_file, chmap=DAQ_CHMAP)
        ts, fcn_daq2_, drift_ppm = sync_photometry_to_daq(
            vdaq=vdaq, fs=fs, df_photometry=fp_data, v_threshold=V_THRESHOLD)
        gc_bpod, _ = GoCueTriggerTimes(session_path=self.session_path).extract(task_collection='raw_behavior_data', save=False)
        gc_daq = rises(vdaq['bpod'])

        fcn_daq2_bpod, drift_ppm, idaq, ibp = sync_timestamps(
            rises(vdaq['bpod']) / fs, gc_bpod, return_indices=True)
        assert drift_ppm < 100, f"Drift between bpod and daq is above 100 ppm: {drift_ppm}"
        assert (gc_daq.size - idaq.size) < 5, "Bpod and daq synchronisation failed as too few" \
                                              "events could be matched"
        ts = fcn_daq2_bpod(ts)
        return ts


# upload to the session endpoint, qc per regions
class FibrePhotometryQC(QC):

    def __init__(self, session_path_or_eid, **kwargs):
        """
        :param session_path_or_eid: A session eid or path
        :param log: A logging.Logger instance, if None the 'ibllib' logger is used
        :param one: An ONE instance for fetching and setting the QC on Alyx
        """
        # When an eid is provided, we will download the required data by default (if necessary)
        download_data = not is_session_path(session_path_or_eid)
        self.download_data = kwargs.pop('download_data', download_data)
        super().__init__(session_path_or_eid, **kwargs)
        self.data = Bunch()

        # QC outcomes map
        self.metrics = None

    def load_data(self, download_data: bool = None) -> None:
        """Extract the data from data files
        Extracts all the required task data from the data files.

        :param download_data: if True, any missing raw data is downloaded via ONE.
        """
        if download_data is not None:
            self.download_data = download_data
        if self.one and not self.one.offline:
            self._ensure_required_data()

        alf_path = self.session_path.joinpath('alf')

        # Load times
        self.data = alfio.load_object(alf_path, '_ibl_photometry')

    def _ensure_required_data(self):
        """
        Ensures the datasets required for QC are local.  If the download_data attribute is True,
        any missing data are downloaded.  If all the data are not present locally at the end of
        it an exception is raised.
        :return:
        """
        for ds in self.dstypes:
            # Check if data available locally
            if not next(self.session_path.rglob(ds), None):
                # If download is allowed, try to download
                if self.download_data is True:
                    assert self.one is not None, 'ONE required to download data'
                    try:
                        self.one.load_dataset(self.eid, ds, download_only=True)
                    except ALFObjectNotFound:
                        raise AssertionError(f'Dataset {ds} not found locally and failed to download')
                else:
                    raise AssertionError(f'Dataset {ds} not found locally and download_data is False')

    def run(self, update: bool = False, **kwargs) -> (str, dict):

        _logger.info(f'Running Fibre Photometry QC for session {self.eid}')
        if all(x is None for x in self.data.values()):
            self.load_data(**kwargs)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(FibrePhotometryQC, is_metric)
        self.metrics = {}
        for k, ch in checks:
            output = ch(self)
            metrics = {f'_roi{i}_' + k[6:]: out for i, out in output}
            self.metrics.update(metrics)

    def check_photobleach(self, n_frames=1000, threshold=0.8):

        qcs = []
        for iD, d in enumerate(self.data['green'].T):
            first = np.mean(d[100:100 + n_frames])
            last = np.mean(d[-100 + n_frames:-100])
            qcs.append((last / first) > threshold)

        return qcs

    def check_dff_qc(self,):
        # need to do photobleaching correction
        # and then run the qc on this
        ...

    def bleach_correct(nacc, avg_window=60, fr=25):
        '''
        Correct for bleaching of gcamp across the session. Calculates
        DF/F
        Parameters
        ----------
        nacc: series with fluorescence data
        avg_window: time for sliding window to calculate F value in seconds
        fr = frame_rate
        '''
        # First calculate sliding window
        avg_window = int(avg_window * fr)
        F = nacc.rolling(avg_window, center=True).mean()
        nacc_corrected = (nacc - F) / F
        return nacc_corrected


def check_photobleaching_qc(raw_signal, frame_window=1000):
    init = raw_signal[100:1100].mean()
    last = raw_signal[-1100:-100].mean()  # 100 is to avoid init and end artifacts
    qc = 1 * ((last / init) > 0.8)
    return qc


def dff_qc(dff, thres=0.05, frame_interval=40):
    separation_min = 2000 / frame_interval  # 2 seconds separation (10 min)
    peaks = np.where(dff > 0.05)[0]
    qc = 1 * (len(np.where(np.diff(peaks) > separation_min)[0]) > 5)
    return qc


class FibrePhotometry(BaseFibrePhotometry):
    """Unlike base extractor, this uses the Bpod times as the main clock and syncs to MCC DAQ"""

    def __init__(self, *args, **kwargs):
        """An extractor for all biased_fibrephotometry task data"""
        super().__init__(*args, **kwargs)

    def _extract(self, light_source_map=None, collection=None, regions=None, **kwargs):
        """

        Parameters
        ----------
        regions: list of str
            The list of regions to extract. If None extracts all columns containing "Region". Defaults to None.
        light_source_map : dict
            An optional map of light source wavelengths (nm) used and their corresponding colour name.
        collection: str / pathlib.Path
            An optional relative path from the session root folder to find the raw photometry data.
            Defaults to `raw_photometry_data`

        Returns
        -------
        numpy.ndarray
            A 1D array of signal values.
        numpy.ndarray
            A 1D array of ints corresponding to the active light source during a given frame.
        pandas.DataFrame
            A table of intensity for each region, with associated times, wavelengths, names and colors
        """
        out_df = super()._extract(light_source_map=None, collection=None, regions=None, **kwargs)

        fp_path = self.session_path.joinpath('raw_fp_data')
        if not fp_path.exists():
            fp_path = self.session_path.joinpath('alf', 'fp_data')

        fp_data = alfio.load_object(self.session_path.joinpath(self.collection), 'fpData')['raw']
        processed = pd.read_csv(fp_path.joinpath('FP470_processed.csv'))
        assert all(processed['FrameCounter'].diff()[1:] == processed['FrameCounter'].diff().median())
        include = np.zeros_like(out_df['times'].values, dtype=bool)
        state = fp_data.get('LedState', fp_data.get('Flags', None))
        mask = state.isin(CHANNELS['L470'])
        frame_470 = np.where(mask)[0]
        first_470 = frame_470[0]
        diff_470 = np.diff(frame_470)[0]
        fr_start = np.where(fp_data['FrameCounter'].values == processed.iloc[0]['FrameCounter'])[0][0] - first_470
        fr_end = (np.where(fp_data['FrameCounter'].values == processed.iloc[-1]['FrameCounter'])[0][0] +
                  (diff_470 - first_470))
        include[fr_start:fr_end] = 1
        out_df['include'] = include

        return out_df

    def extract_timestamps(self, fp_data, chmap=None, **kwargs):
        """Extract and sync the fibrephotometry timestamps acquired through the MCC DAQ.

        This function is called by the _extract method of the super class.

        Parameters
        ----------
        fp_data : dict
            A Bunch of raw fibrephotometry data, with the keys ('raw', 'channels').
        chmap : dict
            An optional dict of DAQ channel IDs for 'bpod' and 'fp' channel inputs.

        Returns
        -------
        numpy.ndarray
            A 1D array of timestamps, one per frame
        """
        chmap = kwargs.get('chmap', DAQ_CHMAP)
        daq_data = raw_daq_loaders.load_channels_tdms(self.session_path.joinpath(self.collection), chmap=chmap)[0]
        trials_data = alfio.load_object(self.session_path / 'alf', 'trials')
        return self.sync_timestamps(daq_data, fp_data, trials_data)

    @staticmethod
    def sync_timestamps(daq_data, fp_data, trials):
        """
        Converts the Neurophotometrics frame timestamps in Bpod time using the Bpod feedback times.

        Both the Neurophotometrics and Bpod devices are connected to a common DAQ. The Bpod sends
        a TTL to the DAQ when feedback is delivered (among other things).  The Neurophotometrics
        sends TTLs when a frame in the 470 channel is acquired.  The Bpod is used as the master
        clock.  First the feedback TTLs are identified on the DAQ based on length. The Bpod times
        for these TTLs are taken from the trials data and assigned to a DAQ sample.  Times for the
        other DAQ samples are calculated using linear interpolation.  Those timestamps for the
        samples where a Neurophotometrics TTL was received are sync'd to the raw Neurophotometrics
        clock timestamps.  Using the determined interpolation function, the raw Neurophotometrics
        timestamps for each frame are converted to Bpod time.

        Parameters
        ----------
        daq_data : dict
            Dictionary with keys ('bpod', 'fp') containing voltage values acquired from the DAQ
        fp_data : dict
            A Bunch containing the raw fibrephotometry data output from Neurophotometrics Bonsai
            workflow and a table of channel values for interpreting the frame state.
        trials : dict
            An ALF trials object containing Bpod trials events.  Only `feedback_times` and `choice`
            keys are required

        Returns
        -------
        numpy.ndarray
            An array of frame timestamps the length of fp_data
        """

        daq_data = pd.DataFrame.from_dict(daq_data)
        daq_data['photometry'] = 1 * (daq_data['photometry'] >= 4)
        daq_data['bpod'] = 1 * (daq_data['bpod'] >= 2)

        # Find if the session has been started more than twice
        daq_data.loc[np.where(daq_data['photometry'].diff() == 1)[0], 'photometry_ttl'] = 1
        ttl_interval = np.median(np.diff(daq_data.loc[daq_data['photometry_ttl'] == 1].index))
        state = fp_data.get('LedState', fp_data.get('Flags', None))
        mask = state.isin(CHANNELS['L470'])
        if ttl_interval == 40:
            frame_number = fp_data['FrameCounter'][mask]
        elif ttl_interval == 20:
            frame_number = fp_data['FrameCounter'][mask]
        elif ttl_interval == 10:
            frame_number = fp_data['FrameCounter']
        elif ttl_interval == 25:
            frame_number = fp_data['FrameCounter']
        elif ttl_interval == 50:
            frame_number = fp_data['FrameCounter'][mask]

        ttl_diff = np.diff(daq_data.loc[daq_data['photometry_ttl'] == 1].index)
        while ttl_diff.max() > ttl_interval * 3:
            print(f'Big gaps: {ttl_diff.max()}')
            ttl_id = np.where(ttl_diff == ttl_diff.max())[0][0]
            real_id = daq_data.loc[daq_data['photometry_ttl'] == 1].index[ttl_id]
            daq_data.iloc[:int(real_id + ttl_diff.max() - ttl_interval), :] = 0
            daq_data.loc[np.where(daq_data['photometry'].diff() == 1)[0], 'photometry_ttl'] = 1
            ttl_diff = np.diff(daq_data.loc[daq_data['photometry_ttl'] == 1].index)

        photometry_ttl = np.where(daq_data['photometry_ttl'] == 1)[0].astype(int)
        assert (np.abs(photometry_ttl.size - frame_number.size) <= 15)
        fp_data.loc[frame_number, 'daq_timestamp'] = photometry_ttl[:len(frame_number)]

        daq_data.loc[np.where(daq_data['bpod'].diff() == 1)[0], 'ttl_on'] = 1
        daq_data.loc[np.where(daq_data['bpod'].diff() == -1)[0], 'ttl_off'] = 1
        daq_data.loc[np.where(daq_data['bpod'].diff() == 1)[0], 'ttl_duration'] = \
            daq_data.loc[daq_data['ttl_off'] == 1].index - daq_data.loc[daq_data['ttl_on'] == 1].index
        # Valve and error tones have pulses > 105
        daq_data.loc[daq_data['ttl_duration'] > 105, 'feedback_times'] = 1

        n_daq_trials = (daq_data['feedback_times'] == 1).sum()
        if n_daq_trials == trials['feedback_times'].size:
            daq_data.loc[daq_data['feedback_times'] == 1, 'bpod_times'] = trials['feedback_times']
        elif n_daq_trials - trials['feedback_times'].size == 1:
            daq_data.loc[daq_data['feedback_times'] == 1, 'bpod_times'] = np.r_[trials['feedback_times'], np.nan]
        else:
            trial_diff = n_daq_trials - trials['feedback_times'].size
            nan_trials = np.where(trials.choice == 0)[0]
            if len(nan_trials) == -1 * trial_diff:
                feedback_times = np.delete(trials['feedback_times'], nan_trials)
                assert len(feedback_times) == n_daq_trials, "Trials don't match up"
                daq_data.loc[daq_data['feedback_times'] == 1, 'bpod_times'] = feedback_times
            else:
                assert n_daq_trials == trials['feedback_times'].size, "Trials don't match up"

        daq_data['bpod_times'].interpolate(inplace=True)
        # Set values after last pulse to nan
        bpod_column = daq_data.columns.get_loc("bpod_times")
        daq_data.iloc[np.where(daq_data['bpod_times'] == daq_data['bpod_times'].max())[0][1]:, bpod_column] = np.nan

        fp_data.loc[frame_number, 'bpod_times'] = daq_data['bpod_times'].values[
            fp_data.loc[frame_number, 'daq_timestamp'].values.astype(int)]

        use_times = ~fp_data['bpod_times'].isna()

        fcn = scipy.interpolate.interp1d(
            fp_data['Timestamp'][use_times].values, fp_data['bpod_times'][use_times].values, fill_value="extrapolate")

        ts = fcn(fp_data['Timestamp'].values)

        return ts


class FibrePhotometryPreprocess(base_tasks.DynamicTask):
    @property
    def signature(self):
        signature = {
            'input_files': [('_mcc_DAQdata.raw.tdms', self.device_collection, True),
                            ('_neurophotometrics_fpData.raw.pqt', self.device_collection, True)],
            'output_files': [('photometry.signal.pqt', 'alf/photometry', True)]
        }
        return signature

    priority = 90
    level = 1

    def __init__(self, session_path, regions=None, **kwargs):
        super().__init__(session_path, **kwargs)
        # Task collection (this needs to be specified in the task kwargs)
        self.collection = self.get_task_collection(kwargs.get('collection', None))
        self.device_collection = self.get_device_collection('photometry', device_collection='raw_photometry_data')
        self.regions = regions

    def _run(self, **kwargs):
        _, out_files = FibrePhotometry(self.session_path, collection=self.device_collection).extract(
            regions=self.regions, path_out=self.session_path.joinpath('alf', 'photometry'), save=True)
        return out_files

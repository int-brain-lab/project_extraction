import logging
import re
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from pandas.core.dtypes.concat import union_categoricals

from ibldsp.utils import sync_timestamps
from ibllib.exceptions import SyncBpodFpgaException
from ibllib.io.extractors.ephys_fpga import BPOD_FPGA_DRIFT_THRESHOLD_PPM, get_protocol_period, get_sync_fronts
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap
from ibllib.pipes.base_tasks import BehaviourTask
from iblutil.io import jsonable
from one.alf import io as alfio

logger = logging.getLogger('ibllib.' + __name__)


RE_PATTERN_EVENT = re.compile(r'^(?P<Channel>\D+\d?)_?(?P<Value>.*)$')


def bpod_session_data_to_dataframe(bpod_data: list[dict[str, Any]], existing_data: pd.DataFrame | None = None) -> pd.DataFrame:
    trials = np.arange(len(bpod_data))
    if existing_data is not None and 'Trial' in existing_data:
        trials += existing_data.iloc[-1].Trial + 1
    dataframes = [] if existing_data is None or len(existing_data) == 0 else [existing_data]
    for index, trial in enumerate(trials):
        dataframes.append(bpod_trial_data_to_dataframe(bpod_data[index], trial))
    return concat_bpod_dataframes(dataframes)


def concat_bpod_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    categories_type = union_categoricals([df['Type'] for df in dataframes])
    categories_state = union_categoricals([df['State'] for df in dataframes])
    categories_event = union_categoricals([df['Event'] for df in dataframes])
    categories_channel = union_categoricals([df['Channel'] for df in dataframes])
    for df in dataframes:
        df['Type'] = df['Type'].cat.set_categories(categories_type.categories)
        df['State'] = df['State'].cat.set_categories(categories_state.categories)
        df['Event'] = df['Event'].cat.set_categories(categories_event.categories)
        df['Channel'] = df['Channel'].cat.set_categories(categories_channel.categories)
    return pd.concat(dataframes)


def bpod_trial_data_to_dataframes(
    bpod_trial_data: list[dict[str, Any]], existing_data: list[pd.DataFrame] | None = None
) -> list[pd.DataFrame]:
    dataframes = existing_data if existing_data is not None else list()
    trial_number = len(dataframes)
    for single_trial_data in bpod_trial_data:
        dataframes.append(bpod_trial_data_to_dataframe(bpod_trial_data=single_trial_data, trial=trial_number))
        trial_number += 1
    return dataframes


def bpod_trial_data_to_dataframe(bpod_trial_data: dict[str, Any], trial: int) -> pd.DataFrame:
    trial_start = bpod_trial_data['Trial start timestamp']
    trial_end = bpod_trial_data['Trial end timestamp']
    state_times = bpod_trial_data['States timestamps'].items()
    event_times = bpod_trial_data['Events timestamps'].items()
    event_list = [(0, 'TrialStart', pd.NA, pd.NA)]
    event_list += [(t, 'StateStart', state, pd.NA) for state, times in state_times for t, _ in times if not np.isnan(t)]
    event_list += [(t, 'InputEvent', pd.NA, event) for event, times in event_times for t in times]
    event_list += [(t, 'StateEnd', state, pd.NA) for state, times in state_times for _, t in times if not np.isnan(t)]
    event_list += [(trial_end - trial_start, 'TrialEnd', pd.NA, pd.NA)]
    event_list = sorted(event_list)
    df = pd.DataFrame(data=event_list, columns=['Time', 'Type', 'State', 'Event'])
    df.Time = np.array((df.Time + trial_start) * 1e6, dtype='timedelta64[us]')
    df.set_index('Time', inplace=True)
    df['Type'] = df['Type'].astype('category')
    df['State'] = df['State'].astype('category').ffill()
    df['Event'] = df['Event'].astype('category')
    df.insert(2, 'Trial', pd.to_numeric([trial], downcast='unsigned')[0])
    mappings = df['Event'].cat.categories.to_series().str.extract(RE_PATTERN_EVENT, expand=True)
    mappings['Channel'] = mappings['Channel'].astype('category')
    mappings['Value'] = mappings['Value'].replace({'Low': '0', 'High': '1', 'Out': '0', 'In': '1'})
    mappings['Value'] = pd.to_numeric(mappings['Value'], errors='coerce', downcast='unsigned', dtype_backend='numpy_nullable')
    df['Channel'] = df['Event'].map(mappings['Channel'])
    df['Value'] = df['Event'].map(mappings['Value'])
    return df


def create_dataframe(jsonable_file: Path) -> pd.DataFrame:
    if jsonable_file.name != '_iblrig_taskData.raw.jsonable':
        raise ValueError('Input file must be named `_iblrig_taskData.raw.jsonable`')
    bpod_dicts = jsonable.load_task_jsonable(jsonable_file)[1]
    bpod_data = bpod_session_data_to_dataframe(bpod_dicts)
    output = bpod_data[bpod_data['Channel'].eq('BNC2')].copy()
    if len(output) == 0:
        raise ValueError('No audio TTLs found in the provided file')
    output[['Stimulus', 'Frequency', 'Attenuation']] = output['State'].str.extract(r'^(\d+)_(\d+|WN)[^-\d]+([-\d]+)dB$')
    output.replace({'Frequency': 'WN'}, '-1', inplace=True)
    output[['Stimulus', 'Frequency', 'Attenuation']] = output[['Stimulus', 'Frequency', 'Attenuation']].astype('Int64')
    return output[['Trial', 'Stimulus', 'Value', 'Frequency', 'Attenuation']]


class TonotopicMappingBpod(BehaviourTask):
    """Extract data from tonotopic mapping task - bpod time."""

    @property
    def signature(self):
        signature = super().signature
        signature['input_files'] = [
            ('_iblrig_taskData.raw.pqt', self.collection, True, True),
            ('_iblrig_taskSettings.raw.json', self.collection, True, True),
        ]
        signature['output_files'] = [('_sp_tonotopic.trials.pqt', self.output_collection, True)]
        return signature

    @overload
    def extract_behaviour(self, save: bool = Literal[True]) -> tuple[pd.DataFrame, list[Path]]: ...

    @overload
    def extract_behaviour(self, save: bool = Literal[False]) -> tuple[pd.DataFrame, None]: ...

    def extract_behaviour(self, save: bool = True) -> tuple[pd.DataFrame, list[Path] | None]:
        filename_in = self.session_path.joinpath(self.collection, '_iblrig_taskData.raw.jsonable').absolute()
        filename_out = []
        data = create_dataframe(filename_in)
        if save:
            filename_out.append(self.session_path / self.output_files[0].glob_pattern)
            filename_out[0].parent.mkdir(exist_ok=True, parents=True)
            data.to_parquet(filename_out[0])
        return data, filename_out

    def _run(self, overwrite: bool = False, save: bool = True) -> list[Path]:
        trials, output_files = self.extract_behaviour(save=save)
        return output_files


class TonotopicMappingTimeline(TonotopicMappingBpod):
    """Extract data from tonotopic mapping task - timeline time."""

    @property
    def signature(self):
        signature = super().signature
        signature['input_files'].extend(
            [
                (f'_{self.sync_namespace}_DAQdata.raw.npy', self.sync_collection, True),
                (f'_{self.sync_namespace}_DAQdata.timestamps.npy', self.sync_collection, True),
                (f'_{self.sync_namespace}_DAQdata.meta.json', self.sync_collection, True),
            ]
        )
        return signature

    def extract_behaviour(self, save=True):
        # get times of bpod TTL event in seconds
        bpod_data, filenames = super().extract_behaviour(save=False)
        bpod_seconds = bpod_data.index.total_seconds().to_numpy(dtype='f')

        # get timeline data
        timeline = alfio.load_object(self.session_path / self.sync_collection, 'DAQdata', namespace='timeline')
        timeline_sync, chmap = load_timeline_sync_and_chmap(self.session_path / self.sync_collection, timeline=timeline)

        # get protocol start and end times (via spacers)
        bpod = get_sync_fronts(timeline_sync, chmap['bpod'])  # spacers
        if self.protocol_number is None:
            raise NotImplementedError
        t_start, t_end = get_protocol_period(self.session_path, self.protocol_number, bpod)

        # get audio sync fronts
        audio_seconds = get_sync_fronts(timeline_sync, chmap['audio'], t_start, t_end)['times']

        # Sync bpod/timeline timestamps
        fcn, drift, ibpod, ifpga = sync_timestamps(bpod_seconds, audio_seconds, return_indices=True)
        logger.info(
            'N trials: %i bpod, %i FPGA, %i merged, sync %.5f ppm', len(bpod_seconds), len(audio_seconds), len(ibpod), drift
        )
        if (drift > 200) and (bpod_seconds.size != audio_seconds.size):
            raise SyncBpodFpgaException('sync cluster f*ck')
        elif drift > BPOD_FPGA_DRIFT_THRESHOLD_PPM:
            logger.warning('BPOD/FPGA synchronization shows values greater than %.2f ppm', BPOD_FPGA_DRIFT_THRESHOLD_PPM)

        # convert timestamps to timeline time
        timeline_seconds = fcn(bpod_seconds)
        timeline_data = bpod_data.copy()
        timeline_data.index = pd.to_timedelta(timeline_seconds, unit='s')

        # save to disk and return as tuple
        filenames = []
        if save:
            filenames.append(self.session_path / self.output_files[0].glob_pattern)
            filenames[0].parent.mkdir(exist_ok=True, parents=True)
            logger.info('Saving timeline data to %s', filenames[0])
            timeline_data.to_parquet(filenames[0])
        return timeline_data, filenames

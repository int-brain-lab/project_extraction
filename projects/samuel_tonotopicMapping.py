import logging
import re
import shutil
from pathlib import Path
from typing import Literal, overload

import pandas as pd

from ibldsp.utils import sync_timestamps
from ibllib.exceptions import SyncBpodFpgaException
from ibllib.io.extractors.ephys_fpga import BPOD_FPGA_DRIFT_THRESHOLD_PPM, get_protocol_period, get_sync_fronts
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap
from ibllib.pipes.base_tasks import BehaviourTask
from one.alf import io as alfio

logger = logging.getLogger('ibllib.' + __name__)


RE_PATTERN_EVENT = re.compile(r'^(?P<Channel>\D+\d?)_?(?P<Value>.*)$')


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
        filename_in = self.session_path.joinpath(self.collection, '_iblrig_taskData.raw.pqt').absolute()
        data = pd.read_parquet(filename_in)
        filename_out = []
        if save:
            filename_out.append(self.session_path / self.output_files[0].glob_pattern)
            filename_out[0].parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(filename_in, filename_out[0])
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

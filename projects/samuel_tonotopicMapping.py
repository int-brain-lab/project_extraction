from pathlib import Path
from typing import Literal, overload

import pandas as pd

from ibllib.pipes.base_tasks import BehaviourTask
from iblrig_custom_tasks.samuel_tonotopicMapping.task import create_dataframe


class TonotopicMappingBpod(BehaviourTask):
    """Extract data from tonotopic mapping task - bpod time."""

    @property
    def signature(self):
        signature = super().signature
        signature['input_files'] = [('_iblrig_taskData.raw.jsonable', self.collection, True)]
        signature['output_files'] = [('tonotopic.bpod.pqt', self.output_collection, True)]
        return signature

    @overload
    def extract_behaviour(self, save: bool = Literal[True]) -> tuple[pd.DataFrame, list[Path]]: ...

    @overload
    def extract_behaviour(self, save: bool = Literal[False]) -> tuple[pd.DataFrame, None]: ...

    def extract_behaviour(self, save: bool = True) -> tuple[pd.DataFrame, list[Path] | None]:
        filename_in = self.session_path.joinpath(self.collection, '_iblrig_taskData.raw.jsonable').absolute()
        filename_out = None
        data = create_dataframe(filename_in)
        if save:
            filename_out = self.session_path.joinpath(self.output_collection, 'tonotopic.bpod.pqt')
            data.to_parquet(filename_out)
        return data, [filename_out]

    def _run(self, overwrite: bool = False, save: bool = True) -> list[Path]:
        trials, output_files = self.extract_behaviour(save=save)
        return output_files


class TonotopicMappingTimeline(TonotopicMappingBpod):
    """Extract data from tonotopic mapping task - timeline time."""

    @property
    def signature(self):
        signature = super().signature
        signature['output_files'].append(('tonotopic.timeline.pqt', self.output_collection, True))
        return signature

    def extract_behaviour(self, save=True):
        bpod_data, _ = super().extract_behaviour(save=False)
        filename_out = None

        timeline_data = bpod_data  # TODO: implementation

        if save:
            filename_out = self.session_path.joinpath(self.output_collection, 'tonotopic.timeline.pqt')
            timeline_data.to_parquet(filename_out)
        return timeline_data, [filename_out]

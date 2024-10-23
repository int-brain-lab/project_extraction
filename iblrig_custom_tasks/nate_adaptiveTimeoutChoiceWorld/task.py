"""
This task inherits TrainingChoiceWorldSession with the addition of configurable, adaptive timeouts for incorrect
choices depending on the stimulus contrast.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import NonNegativeFloat

from iblrig.misc import get_task_arguments
from iblrig_tasks._iblrig_tasks_trainingChoiceWorld.task import Session as TrainingCWSession

log = logging.getLogger('iblrig.task')


# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)


class AdaptiveTimeoutChoiceWorldTrialData(TrainingCWSession.TrialDataModel):
    adaptive_delay_nogo: NonNegativeFloat
    adaptive_delay_error: NonNegativeFloat


class Session(TrainingCWSession):
    protocol_name = 'nate_adaptiveTimeoutChoiceWorld'
    TrialDataModel = AdaptiveTimeoutChoiceWorldTrialData

    def __init__(
        self,
        *args,
        adaptive_delay_nogo=DEFAULTS['ADAPTIVE_FEEDBACK_NOGO_DELAY_SECS'],
        adaptive_delay_error=DEFAULTS['ADAPTIVE_FEEDBACK_ERROR_DELAY_SECS'],
        **kwargs,
    ):
        self._adaptive_delay_nogo = adaptive_delay_nogo
        self._adaptive_delay_error = adaptive_delay_error
        super().__init__(*args, **kwargs)
        assert len(self._adaptive_delay_nogo) == len(self.task_params.CONTRAST_SET)
        assert len(self._adaptive_delay_error) == len(self.task_params.CONTRAST_SET)

    def draw_next_trial_info(self, **kwargs):
        super().draw_next_trial_info(**kwargs)
        contrast = self.trials_table.at[self.trial_num, 'contrast']
        index = np.flatnonzero(np.array(self.task_params['CONTRAST_SET']) == contrast)[0]
        self.trials_table.at[self.trial_num, 'adaptive_delay_nogo'] = self._adaptive_delay_nogo[index]
        self.trials_table.at[self.trial_num, 'adaptive_delay_error'] = self._adaptive_delay_error[index]

    @property
    def feedback_nogo_delay(self):
        return self.trials_table.at[self.trial_num, 'adaptive_delay_nogo']

    @property
    def feedback_error_delay(self):
        return self.trials_table.at[self.trial_num, 'adaptive_delay_error']

    def show_trial_log(self, extra_info: dict[str, Any] | None = None, log_level: int = logging.INFO):
        trial_info = self.trials_table.iloc[self.trial_num]
        info_dict = {
            'Adaptive no-go delay': f'{trial_info.adaptive_delay_nogo:.2f} s',
            'Adaptive error delay': f'{trial_info.adaptive_delay_error:.2f} s',
        }
        if isinstance(extra_info, dict):
            info_dict.update(extra_info)
        super().show_trial_log(extra_info=info_dict, log_level=log_level)

    @staticmethod
    def extra_parser():
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--adaptive_delay_nogo',
            option_strings=['--adaptive_delay_nogo'],
            dest='adaptive_delay_nogo',
            default=DEFAULTS['ADAPTIVE_FEEDBACK_NOGO_DELAY_SECS'],
            nargs='+',
            type=float,
            help='list of delays for no-go condition (contrasts: 1.0, 0.5, 0.25, 0.125, 0.0625, 0.0)',
        )
        parser.add_argument(
            '--adaptive_delay_error',
            option_strings=['--adaptive_delay_error'],
            dest='adaptive_delay_error',
            default=DEFAULTS['ADAPTIVE_FEEDBACK_ERROR_DELAY_SECS'],
            nargs='+',
            type=float,
            help='list of delays for error condition (contrasts: 1.0, 0.5, 0.25, 0.125, 0.0625, 0.0)',
        )
        return parser


if __name__ == '__main__':  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

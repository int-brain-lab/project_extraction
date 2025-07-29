import pandas as pd

import iblrig
import iblrig.misc
import iblrig.sound
from iblrig.base_tasks import BaseSession, BpodMixin


class Session(BpodMixin, BaseSession):
    protocol_name = 'samuel_tonotopicMapping'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attenuation = pd.read_csv(self.get_task_directory().joinpath('attenuation.csv'))


if __name__ == '__main__':  # pragma: no cover
    kwargs = iblrig.misc.get_task_arguments()
    sess = Session(**kwargs)
    sess.run()

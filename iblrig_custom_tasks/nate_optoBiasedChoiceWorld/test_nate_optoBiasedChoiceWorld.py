"""
Here we test for the state machine code and the task to be importable by the GUI
"""

from iblrig_custom_tasks.nate_optoBiasedChoiceWorld.task import Session

task = Session(subject='toto')


assert task.task_params.get('PROBABILITY_OPTO_STIM', None) is not None
assert any(t.startswith('OptoTrials') for t in task.extractor_tasks or [])

"""
Bpod extractor for ccc_neuromodulatorChoiceWorld task.

This is the same as ephysChoiceWorld with the addition of reward manipulation blocks in the trials table.

The pipeline task subclasses, OptoTrialsBpod and OptoTrialsNidq, aren't strictly necessary.
They simply assert that the laserStimulation datasets were indeed saved and registered by
the Bpod extractor class.
"""
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.bpod_trials import BiasedTrials
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsBpod


class NeuromodulatorChoiceWorldTrialsBpod(ChoiceWorldTrialsBpod):
    """Extract bpod only trials."""
    pass

# class NeuromodulatorChoiceWorldTrials(BaseBpodTrialsExtractor):
#     """
#     Extract like BiasedTrials/EphysTrials but handle reward manipulations
#     """

#     save_names = ('_ibl_trials.goCueTrigger_times.npy', '_ibl_trials.stimOnTrigger_times.npy', None,
#                   '_ibl_trials.stimOffTrigger_times.npy', None, None,
#                   '_ibl_trials.table.pqt', '_ibl_trials.stimOff_times.npy', None, '_ibl_wheel.timestamps.npy',
#                   '_ibl_wheel.position.npy', '_ibl_wheelMoves.intervals.npy', '_ibl_wheelMoves.peakAmplitude.npy', None, None,
#                   '_ibl_trials.included.npy', None, None, '_ibl_trials.quiescencePeriod.npy')
#     var_names = ('goCueTrigger_times', 'stimOnTrigger_times', 'itiIn_times', 'stimOffTrigger_times', 'stimFreezeTrigger_times',
#                  'errorCueTrigger_times', 'table', 'stimOff_times', 'stimFreeze_times', 'wheel_timestamps', 'wheel_position',
#                  'wheelMoves_intervals', 'wheelMoves_peakAmplitude', 'peakVelocity_times', 'is_final_movement', 'included',
#                  'phase', 'position', 'quiescence')

#     def _extract(self, extractor_classes=None, **kwargs) -> dict:
#         extractor_classes = extractor_classes or []

#         # For iblrig v8 we use the biased trials table instead. ContrastLeft, ContrastRight and ProbabilityLeft are
#         # filled from the values in the bpod data itself rather than using the pregenerated session number
#         iblrig_version = self.settings.get('IBLRIG_VERSION', self.settings.get('IBLRIG_VERSION_TAG', '0'))
#         if version.parse(iblrig_version) >= version.parse('8.0.0'):
#             TrialsTable = TrialsTableBiased
#         else:
#             TrialsTable = TrialsTableEphys

#         base = [GoCueTriggerTimes, StimOnTriggerTimes, ItiInTimes, StimOffTriggerTimes, StimFreezeTriggerTimes,
#                 ErrorCueTriggerTimes, TrialsTable, IncludedTrials, PhasePosQuiescence]
#         # Get all detected TTLs. These are stored for QC purposes
#         self.frame2ttl, self.audio = raw.load_bpod_fronts(self.session_path, data=self.bpod_trials)
#         # Exclude from trials table
#         out, _ = run_extractor_classes(
#             base + extractor_classes, session_path=self.session_path, bpod_trials=self.bpod_trials,
#             settings=self.settings, save=False, task_collection=self.task_collection)
#         return {k: out[k] for k in self.var_names}

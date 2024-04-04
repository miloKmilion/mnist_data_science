"""
This file now contains the interface of the experiment tracker
Stage = ExperimentTracker
"""

from typing import Union, Protocol
import numpy as np
from enum import Enum, auto


# 1st change, instead of making a frozen data class. An Enum method is more efficient. This is also for tracking.
class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class ExperimentTracker(Protocol):

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""

    def add_epoch_confusion_matrix(self, y_true: np.array, y_pred: np.array, step: int):
        """Implements logging a confusion matrix at epoch-level."""

    def add_hparams(self, hparams: dict[str, Union[str, float]], metrics: dict[str, float]):
        """Implements logging hyperparameters."""

    def set_stage(self, stage: Stage):
        """ Set the stage of the experiment """

    def flush(self):
        """ Flushes the experiment """

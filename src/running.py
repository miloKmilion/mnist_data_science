from typing import Any, Optional

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.metrics import Metric
from src.tracking import Stage, ExperimentTracker


class Runner:
    def __init__(self,
                 loader: DataLoader[Any],
                 model: torch.nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        self.run_count = 0
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.accuracy_metric = Metric()
        self.compute_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        # As the runner is taking care of the parameters, the batches need to be emptied before each epoch.
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []

    @property
    def avg_accuracy(self) -> float:
        return self.accuracy_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):
        # Training Loop
        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            batch_accuracy = self._run_single(x, y)
            experiment.add_batch_metric('accuracy', batch_accuracy, self.run_count)

    def _run_single(self, x: Any, y: Any):
        self.run_count += 1
        batch_size = x.shape[0]
        prediction = self.model(x)
        loss = self.compute_loss(prediction, y)

        # Compute Batch Metrics
        y_np = y.detach().numpy()
        y_prediction_np = np.argmax(prediction.detach().numpy(), axis=1)
        batch_accuracy: float = accuracy_score(y_np, y_prediction_np)
        self.accuracy_metric.update(batch_accuracy, batch_size)

        # Reverse-mode AutoDiff (backpropagation)
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return batch_accuracy

    def reset(self):
        self.accuracy_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []


def run_epoch(test_runner: Runner,
              train_runner: Runner,
              experiment: ExperimentTracker,
              epoch_id: int,
              epoch_total: int):

    experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment)

    # Log training Epoch Metrics
    experiment.add_epoch_metric("Accuracy", train_runner.avg_accuracy, epoch_id)

    experiment.set_stage(Stage.VAL)
    test_runner.run("Validation Batches", experiment)

    # Log validation Epoch Metrics
    experiment.add_epoch_metric("Accuracy", test_runner.avg_accuracy, epoch_id)
    #experiment.add_epoch_confusion_matrix(y_true=test_runner.y_true_batches,
                                          #y_pred=test_runner.y_pred_batches,
                                          #step=epoch_id)

    # Compute Average Epoch Metrics
    summary = ', '.join([
        f"[Epoch: {epoch_id + 1}/{epoch_total}]",
        f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
        f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
    ])
    print('\n' + summary + '\n')

    test_runner.reset()
    train_runner.reset()

import torch

from src.dataset import get_train_dataloader, get_test_dataloader
from src.models import LinearNet
from src.running import Runner
from src.tracking import Stage
from src.tensorboard import TensorboardExperiment
from src.utils import generate_tensorboard_experiment_directory

# Hyperparameters
hparams = {
    'EPOCHS': 20,
    'LR': 5e-5,
    'OPTIMIZER': 'Adam',
    'BATCH_SIZE': 128
}


def main():
    # Data
    train_loader = get_train_dataloader(batch_size=hparams.get('BATCH_SIZE'))
    test_loader = get_test_dataloader(batch_size=hparams.get('BATCH_SIZE'))

    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.get('LR'))

    # Create the runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root='./runs')
    experiment = TensorboardExperiment(log_dir=log_dir)

    for epoch in range(hparams.get('EPOCHS')):
        experiment.set_stage(Stage.TRAIN)
        train_runner.run("Train Batches", experiment)

        # Log training Epoch Metrics
        experiment.add_epoch_metric("Accuracy", train_runner.avg_accuracy, epoch)

        experiment.set_stage(Stage.VAL)
        test_runner.run("Validation Batches", experiment)

        # Log validation Epoch Metrics
        experiment.add_epoch_metric("Accuracy", test_runner.avg_accuracy, epoch)
        experiment.add_epoch_confusion_matrix(y_true=test_runner.y_true_batches,
                                              y_pred=test_runner.y_pred_batches,
                                              step=epoch)

        # Compute Average Epoch Metrics
        summary = ', '.join([
            f"[Epoch: {epoch + 1}/{hparams.get('EPOCHS')}]",
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
        ])
        print('\n' + summary + '\n')

        test_runner.reset()
        train_runner.reset()

    experiment.flush()


if __name__ == '__main__':
    main()

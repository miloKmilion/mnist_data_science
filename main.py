import pathlib

import torch

from src.dataset import create_dataloader
from src.models import LinearNet
from src.running import Runner, run_epoch
from src.tensorboard import TensorboardExperiment
from src.utils import generate_tensorboard_experiment_directory

# Hyperparameters
EPOCH_COUNT = 20
LR = 5e-5
BATCH_SIZE = 128
LOG_PATH = './runs'

# Data Configuration
DATA_DIR = "./data"
TEST_DATA = pathlib.Path(f"{DATA_DIR}/t10k-images-idx3-ubyte.gz")
TEST_LABELS = pathlib.Path(f"{DATA_DIR}/t10k-labels-idx1-ubyte.gz")
TRAIN_DATA = pathlib.Path(f"{DATA_DIR}/train-images-idx3-ubyte.gz")
TRAIN_LABELS = pathlib.Path(f"{DATA_DIR}/train-labels-idx1-ubyte.gz")


def main():
    # Data
    train_loader = create_dataloader(BATCH_SIZE, TRAIN_DATA, TRAIN_LABELS)
    test_loader = create_dataloader(BATCH_SIZE, TEST_DATA, TEST_LABELS)

    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Create the runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root=LOG_PATH)
    experiment = TensorboardExperiment(log_dir=log_dir)

    for epoch_id in range(EPOCH_COUNT):
        run_epoch(test_runner, train_runner, experiment, epoch_id, EPOCH_COUNT)

    experiment.flush()


if __name__ == '__main__':
    main()

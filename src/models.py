import torch


class LinearNet(torch.nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()

        """By using function composition is possible to build a sequential network."""
        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=28 * 28, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)

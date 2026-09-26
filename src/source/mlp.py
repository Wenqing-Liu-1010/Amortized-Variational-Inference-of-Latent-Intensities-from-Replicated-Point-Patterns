from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_sizes):
        super().__init__()
        sizes = [input_size] + output_sizes
        layers = []
        for left, right in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([nn.Linear(left, right), nn.SiLU()])
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

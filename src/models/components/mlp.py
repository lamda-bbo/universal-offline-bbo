from typing import Sequence, Union

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[Sequence[int], int],
        output_dim: int = 1,
        require_batch_norm: bool = False,
    ) -> None:
        super(SimpleMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.require_batch_norm = require_batch_norm
        if require_batch_norm:
            self.batch_norm = nn.BatchNorm1d(input_dim)

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.require_batch_norm:
            x = self.batch_norm(x)
        return self.layers(x)

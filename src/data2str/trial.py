from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class DataTrial(ABC):
    x: str
    y: float
    ss_str: str
    mask: bool

    def get_data(self) -> Dict[str, torch.Tensor]:
        return self()

    def __call__(self) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(self.x),
            "y": torch.tensor(self.y, dtype=torch.float32),
            "metadata": torch.tensor(self.ss_str),
            "mask": torch.tensor(self.mask, dtype=torch.bool),
        }

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np


@dataclass
class TaskData(ABC):
    data: np.ndarray

    @abstractmethod
    def to_string(self) -> List[str]:
        """Convert data to list of string descriptions"""
        pass

    @abstractmethod
    def get_variable_values(self) -> Dict:
        """Return dictionary of variable values"""
        pass


@dataclass
class ContinuousTaskData(TaskData):
    def to_string(self) -> List[List[str]]:
        return [
            [f"x{i}: {value.item():.4f}" for i, value in enumerate(data)]
            for data in self.data
        ]

    def get_variable_values(self) -> Dict:
        return {f"x{i}": float(value) for i, value in enumerate(self.data)}


@dataclass
class IntegerTaskData(TaskData):
    def to_string(self) -> List[List[str]]:
        return [
            [f"x{i}: {int(value)}" for i, value in enumerate(data)]
            for data in self.data
        ]

    def get_variable_values(self) -> Dict:
        return {f"x{i}": int(value) for i, value in enumerate(self.data)}


@dataclass
class CategoricalTaskData(IntegerTaskData):
    def to_string(self) -> List[List[str]]:
        return [
            [f"x{i}: '{int(value)}'" for i, value in enumerate(data)]
            for data in self.data
        ]


@dataclass
class PermutationTaskData(IntegerTaskData):
    pass

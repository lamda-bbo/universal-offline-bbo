from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class TaskMetadata(ABC):
    name: str
    input_dim: int
    objective: str
    description: str

    @abstractmethod
    def get_variable_metadata(self) -> Dict:
        """Return metadata for each input variable"""
        pass

    @abstractmethod
    def to_string(self) -> List[str]:
        """Convert metadata to list of string descriptions in format 'x{i}:TYPE,[bounds/size]'"""
        pass


@dataclass
class ContinuousTaskMetadata(TaskMetadata):
    bounds: List[Tuple[float, float]]

    def get_variable_metadata(self) -> Dict:
        return {
            f"x{i}": {"type": "DOUBLE", "bounds": self.bounds[i]}
            for i in range(self.input_dim)
        }

    def to_string(self) -> str:
        data_str = "; ".join(
            [f"x{i}:DOUBLE, {list(bound)}" for i, bound in enumerate(self.bounds)]
        )
        task_str = f"name: '{self.name}'"
        if self.description:
            task_str = f"{task_str}, description: '{self.description}'"
        task_str = f"{task_str}, objective: '{self.objective}'"
        # return f"{task_str}. Data info: {data_str}"
        return task_str


@dataclass
class CategoricalTaskMetadata(TaskMetadata):
    n_categories: int

    def get_variable_metadata(self) -> Dict:
        return {
            f"x{i}": {"type": "CATEGORICAL", "size": self.n_categories}
            for i in range(self.input_dim)
        }

    def to_string(self) -> str:
        data_str = "; ".join(
            [f"x{i}:CATEGORICAL, {self.n_categories}" for i in range(self.input_dim)]
        )
        task_str = f"name: '{self.name}'"
        if self.description:
            task_str = f"{task_str}, description: '{self.description}'"
        task_str = f"{task_str}, objective: '{self.objective}'"
        # return f"{task_str}. Data info: {data_str}"
        return task_str


@dataclass
class IntegerTaskMetadata(TaskMetadata):
    bounds: List[Tuple[int, int]]

    def get_variable_metadata(self) -> Dict:
        return {
            f"x{i}": {"type": "INTEGER", "bounds": self.bounds[i]}
            for i in range(self.input_dim)
        }

    def to_string(self) -> str:
        data_str = "; ".join(
            [f"x{i}:INTEGER, {list(bound)}" for i, bound in enumerate(self.bounds)]
        )
        task_str = f"name: '{self.name}'"
        if self.description:
            task_str = f"{task_str}, description: '{self.description}'"
        task_str = f"{task_str}, objective: '{self.objective}'"
        # return f"{task_str}. Data info: {data_str}"
        return task_str


@dataclass
class PermutationTaskMetadata(TaskMetadata):

    def get_variable_metadata(self) -> Dict:
        return {
            f"x{i}": {"task": "PERMUTATION", "size": self.input_dim}
            for i in range(self.input_dim)
        }

    def to_string(self) -> str:
        data_str = f"task:PERMUTATION, size:{self.input_dim}"
        task_str = f"name: '{self.name}'"
        if self.description:
            task_str = f"{task_str}, description: '{self.description}'"
        task_str = f"{task_str}, objective: '{self.objective}'"
        # return f"{task_str}. Data info: {data_str}"
        return task_str

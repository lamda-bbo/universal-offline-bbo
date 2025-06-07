import abc
import time
from typing import Callable, Tuple

import numpy as np

from src.tasks.base import OfflineBBOTask


class BaseSearcher:
    def __init__(
        self,
        task: OfflineBBOTask,
        score_fn: Callable[[np.ndarray], np.ndarray],
        num_solutions: int,
    ) -> None:
        self.task = task
        self.score_fn = score_fn
        self.num_solutions = num_solutions

    @staticmethod
    def get_initial_designs(
        x: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.argsort(y.squeeze())[-k:]
        return x[indices], y[indices]

    @abc.abstractmethod
    def run(self) -> np.ndarray:
        pass


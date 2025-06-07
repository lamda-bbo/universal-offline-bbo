import abc
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

DESIGN_TYPE = Sequence[Union[int, float]]


class BaseTask:
    @abc.abstractmethod
    def _evaluate(self, x: Union[Sequence[DESIGN_TYPE], np.ndarray]) -> np.ndarray:
        pass


class OfflineBBOTask(BaseTask):
    def __init__(
        self,
        task_name: str,
        *,
        task_type: str,
        x_np: np.ndarray,
        y_np: np.ndarray,
        full_y_min: Optional[Union[float, np.ndarray]] = None,
        full_y_max: Optional[Union[float, np.ndarray]] = None,
        x_ood_np: Optional[np.ndarray] = None,
        y_ood_np: Optional[np.ndarray] = None,
    ) -> None:
        assert task_type in (
            "Continuous",
            "Integer",
            "Categorical",
            "Permutation",
        ), f"Unsupported task_type: {task_type}"
        self.task_type = task_type
        self.task_name = task_name
        self.x_np = x_np
        self.y_np = y_np

        self.full_y_min = full_y_min
        self.full_y_max = full_y_max

        self.x_ood_np = x_ood_np
        self.y_ood_np = y_ood_np

    @property
    @abc.abstractmethod
    def eval_stability(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    def ndim_problem(self) -> int:
        return self.x_np.shape[1]

    def evaluate(
        self, x: np.ndarray, return_normalized_y: bool = True
    ) -> Dict[str, np.ndarray]:
        if self.task_type == "Continuous":
            assert x.dtype in [
                np.float32,
                np.float64,
            ], f"Input dtype must be float32 or float64, but got {x.dtype}"
        elif self.task_type in ("Categorical", "Integer", "Permutation"):
            assert x.dtype in [
                np.int32,
                np.int64,
            ], f"Input dtype must be int32 or int64, but got {x.dtype}"
        else:
            raise NotImplementedError

        def get_percentile_score(
            score: np.ndarray, prefix: str = ""
        ) -> Dict[str, float]:
            prefix = f"{prefix}-" if prefix != "" else prefix
            return {
                f"{prefix}score-100th": np.max(score).item(),
                f"{prefix}score-75th": np.percentile(score, 75).item(),
                f"{prefix}score-50th": np.median(score).item(),
                f"{prefix}score-25th": np.percentile(score, 25).item(),
            }

        score = self._evaluate(x)
        score_dict = get_percentile_score(score)

        if return_normalized_y:
            normalized_score = (score - self.full_y_min) / (
                self.full_y_max - self.full_y_min
            )
            score_dict.update(
                get_percentile_score(normalized_score, prefix="normalized")
            )

        return score_dict

    def evaluate_stability(self, x: List[np.ndarray]) -> float:
        y = [self._evaluate(x0) for x0 in x]
        y = np.array([y0.max().item() for y0 in y])
        d_best = self.y_np.max()
        y_max = y.max()
        if y_max < d_best:
            return float("-inf")
        elif y_max == d_best:
            return 0

        opt_steps = len(y)
        r = np.array([i for i in range(opt_steps)])
        s_d = d_best * opt_steps
        s_0 = y_max * opt_steps
        s = np.trapz(y.squeeze(), np.array([i for i in range(opt_steps)]))
        return (s - s_d) / (s_0 - s_d)

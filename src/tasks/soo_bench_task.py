from typing import Dict, Optional, Tuple

import numpy as np
from soo_bench.Taskdata import OfflineTask, set_use_cache

from src.tasks.base import OfflineBBOTask


class SOOBenchTask(OfflineBBOTask):

    def __init__(
        self,
        task_name: str,
        benchmark_id: int,
        seed: int = 1,
        *,
        low: int = 0,
        high: int = 100,
        num_data: Optional[int] = None,
    ) -> None:
        assert low < high, f"low percentile must be lower than high percentile"
        set_use_cache(True)
        self.task = OfflineTask(task_name, benchmark=benchmark_id, seed=seed)
        task_desc = f"{task_name}_{benchmark_id}_{seed}"
        if num_data is None:
            num_data = (
                0  # will be reset to dimension of the problem * 1000 in SOO-Bench
            )

        self.task.sample_bound(num=num_data, low=low, high=high)
        task_x, task_y = self.task.x.copy(), self.task.y.copy()
        task_y = task_y

        dic2y = np.load("src/tasks/dic2y_sb.npy", allow_pickle=True).item()
        full_y_min, full_y_max = dic2y[task_desc]

        super(SOOBenchTask, self).__init__(
            task_desc,
            task_type="Continuous",
            x_np=task_x,
            y_np=task_y,
            full_y_min=full_y_min,
            full_y_max=full_y_max,
        )

    @property
    def eval_stability(self) -> bool:
        return True

    def _evaluate(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        x = x.reshape(-1, self.x_np.shape[1])
        return self.task.predict(x)[0]

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.task.xl, self.task.xu

    @property
    def ndim_problem(self) -> int:
        return self.x_np.shape[1]

    @property
    def num_classes(self) -> int:
        if self.task_type == "Continuous":
            raise ValueError("continuous task does not support num_classes attribute")
        return self.task.num_classes

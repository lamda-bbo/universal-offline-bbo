from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import design_bench as db
import numpy as np
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import (
    DKittyMorphologyDataset,
)
from design_bench.datasets.continuous.hopper_controller_dataset import (
    HopperControllerDataset,
)
from design_bench.datasets.continuous.superconductor_dataset import (
    SuperconductorDataset,
)
from design_bench.datasets.discrete.nas_bench_dataset import NASBenchDataset
from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.task import Task

from src.tasks.base import OfflineBBOTask

_task_kwargs = {
    "AntMorphology-Exact-v0": {"relabel": True},
    "TFBind10-Exact-v0": {
        "dataset_kwargs": {
            "max_samples": 30000,
        }
    },
}

_taskname2datafunc = {
    "AntMorphology-Exact-v0": lambda: AntMorphologyDataset(),
    "DKittyMorphology-Exact-v0": lambda: DKittyMorphologyDataset(),
    "Superconductor-RandomForest-v0": lambda: SuperconductorDataset(),
    "TFBind8-Exact-v0": lambda: TFBind8Dataset(),
    "TFBind10-Exact-v0": lambda: TFBind10Dataset(),
    "CIFARNAS-Exact-v0": lambda: NASBenchDataset(),
    "HopperController-Exact-v0": lambda: HopperControllerDataset(),
}


def _load_ood_data(task_name: str, task_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    full_dataset = _taskname2datafunc[task_name]()
    full_x: np.ndarray = full_dataset.x.copy()
    full_y: np.ndarray = full_dataset.y.copy()

    def create_mask(full_x: np.ndarray, x: np.ndarray) -> np.ndarray:

        dtype = [("", full_x.dtype)] * full_x.shape[1]

        full_x_struct = full_x.view(dtype)
        task_x_struct = x.view(dtype)

        mask = np.in1d(full_x_struct, task_x_struct)
        return mask

    if task_name == "TFBind10-Exact-v0":
        index = np.random.choice(full_y.shape[0], 30000, replace=False)
        full_x = full_x[index]
        full_y = full_y[index]

    mask = create_mask(full_x, task_x)
    diff_x = full_x[~mask]
    diff_x, unique_indices = np.unique(diff_x, axis=0, return_index=True)

    diff_y = full_y[~mask][unique_indices]

    indices = np.arange(diff_x.shape[0])
    np.random.shuffle(indices)
    diff_x = diff_x[indices]
    diff_y = diff_y[indices]

    return diff_x, diff_y


class DesignBenchTask(OfflineBBOTask):

    def __init__(self, task_name: str, scale_up_ratio: float = 1.0) -> None:
        self.task: Task = db.make(task_name, **_task_kwargs.get(task_name, {}))
        dic2y = np.load("src/tasks/dic2y_db.npy", allow_pickle=True).item()
        full_y_min, full_y_max = dic2y[task_name]

        task_x, task_y = self.task.x.copy(), self.task.y.copy()
        x_ood, y_ood = _load_ood_data(task_name, task_x)

        super(DesignBenchTask, self).__init__(
            task_name,
            task_type="Categorical" if self.task.is_discrete else "Continuous",
            x_np=task_x,
            y_np=task_y,
            full_y_min=full_y_min,
            full_y_max=full_y_max,
            x_ood_np=x_ood,
            y_ood_np=y_ood,
        )

        self.scale_up_ratio = scale_up_ratio if self.task_type == "Continuous" else 1.0

    @property
    def eval_stability(self) -> bool:
        return self.task_name in (
            "TFBind8-Exact-v0",
            "TFBind10-Exact-v0",
            "Superconductor-RandomForest-v0",
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, self.x_np.shape[1])
        return self.task.predict(x)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        x_min = self.x_np.min(axis=0)
        x_max = self.x_np.max(axis=0)

        if self.task_type == "Categorical":
            return x_min, x_max

        x_center = (x_max + x_min) / 2

        x_width = x_max - x_min + 1e-6
        x_width_scaled = x_width * self.scale_up_ratio

        new_x_min = x_center - x_width_scaled / 2
        new_x_max = x_center + x_width_scaled / 2

        return new_x_min, new_x_max

    @property
    def ndim_problem(self) -> int:
        return self.x_np.shape[1]

    @property
    def num_classes(self) -> int:
        if self.task_type == "Continuous":
            raise ValueError("continuous task does not support num_classes attribute")
        return self.task.num_classes


if __name__ == "__main__":
    task = DesignBenchTask("Superconductor-RandomForest-v0")
    x = task.x_np[:5]
    print(task.evaluate(x))
    print(task.y_np[:5])

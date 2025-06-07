from typing import Tuple

from src.data2str.task_data import CategoricalTaskData, ContinuousTaskData, TaskData
from src.data2str.task_metadata import (
    CategoricalTaskMetadata,
    ContinuousTaskMetadata,
    TaskMetadata,
)
from src.tasks.base import OfflineBBOTask
from src.tasks.design_bench_task import DesignBenchTask

TASKNAMES = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    # Below are tasks that are not usually used in Design-Bench
    "HopperController-Exact-v0",
]

CONTINUOUS_TASKS = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    # Below are tasks that are not usually used in Design-Bench
    "HopperController-Exact-v0",
]

CATEGORICAL_TASKS = [
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
]

TEXT_DESCRIPTIONS = {
    "AntMorphology-Exact-v0": {
        "name": "Ant Morphology",
        "description": "a quadruped robot morphology optimization",
        "objective": "to run as fast as possible",
    },
    "DKittyMorphology-Exact-v0": {
        "name": "D'Kitty Morphology",
        "description": "D'Kitty robot morphology optimization",
        "objective": "to navigate the robot to a fixed location",
    },
    "Superconductor-RandomForest-v0": {
        "name": "Superconducor",
        "description": "critical temperature maximization",
        "objective": "to design the chemical formula for a superconducting material that has a high critical temperature",
    },
    "TFBind8-Exact-v0": {
        "name": "TF Bind 8",
        "description": "DNA sequence optimization",
        "objective": "to find the length-8 DNA sequence with maximum binding affinity with SIX6_REF_R1 transcription factor",
    },
    "TFBind10-Exact-v0": {
        "name": "TF Bind 10",
        "description": "DNA sequence optimization",
        "objective": "to find the length-10 DNA sequence with maximum binding affinity with SIX6_REF_R1 transcription factor",
    },
    # Below are tasks that are not usually used in Design-Bench
    "HopperController-Exact-v0": {
        "name": "Hopper Controller",
        "description": "Hopper robot neural network controller optimization",
        "objective": "to optimize the weights of a neural network policy so as to maximize the expected discounted return on the Hopper-v2 locomotion task in OpenAI Gym",
    },
}


def create_task(task_name: str) -> Tuple[OfflineBBOTask, TaskMetadata, TaskData]:
    assert task_name in TASKNAMES

    # TODO: fix flags here
    is_continuous = task_name in CONTINUOUS_TASKS
    is_categorical = not is_continuous

    task = DesignBenchTask(
        task_name=task_name, scale_up_ratio=1.5 if is_continuous else 1.0
    )

    if is_continuous:
        xl, xu = task.bounds
        metadata = ContinuousTaskMetadata(
            input_dim=task.x_np.shape[1],
            bounds=[(l, u) for l, u in zip(xl, xu)],
            **TEXT_DESCRIPTIONS[task_name]
        )
        data = ContinuousTaskData(task.x_np)

    elif is_categorical:
        metadata = CategoricalTaskMetadata(
            input_dim=task.x_np.shape[1],
            n_categories=task.num_classes,
            **TEXT_DESCRIPTIONS[task_name]
        )
        data = CategoricalTaskData(task.x_np)

    else:
        raise NotImplementedError("Not supported")

    return task, metadata, data


if __name__ == "__main__":
    task, metadata, data = create_task("AntMorphology-Exact-v0")
    print(task, metadata, data)
    string_data = data.to_string()
    print(string_data[0])

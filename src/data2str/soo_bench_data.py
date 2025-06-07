from typing import Optional, Tuple

from src.data2str.task_data import ContinuousTaskData, TaskData
from src.data2str.task_metadata import ContinuousTaskMetadata, TaskMetadata
from src.tasks.base import OfflineBBOTask
from src.tasks.soo_bench_task import SOOBenchTask

TASKNAMES = [
    "gtopx_data_2_1",
    "gtopx_data_3_1",
    "gtopx_data_4_1",
    "gtopx_data_6_1",
]

CONTINUOUS_TASKS = [
    "gtopx_data_2_1",
    "gtopx_data_3_1",
    "gtopx_data_4_1",
    "gtopx_data_6_1",
]

TEXT_DESCRIPTIONS = {
    "gtopx_data_2_1": {
        "name": "Cassini 2",
        "description": "Complex interplanetary missions to Saturn",
        "objective": "to achieve a rendezvous with Saturn, aiming to minimize the total velocity change",
    },
    "gtopx_data_3_1": {
        "name": "Messenger (reduced)",
        "description": "Simulation of interplanetary missions to Mercury",
        "objective": "to minimize the total velocity change over the course of the mission",
    },
    "gtopx_data_4_1": {
        "name": "Messenger (full)",
        "description": "Interplanetary missions to Mercury, with resonant flybys of the planet",
        "objective": "to minimize the total velocity change incurred throughout the mission",
    },
    "gtopx_data_6_1": {
        "name": "Rosetta",
        "description": "Simulation of multi-gravity-assisted space missions to Comet 67P/Churyumov-Gerasimenko",
        "objective": "to minimize the total velocity change required throughout the mission",
    },
}


def create_task(
    task_name: str,
    benchmark_id: int = 2,
    seed: int = 1,
    *,
    low: int = 0,
    high: int = 100,
    num_data: Optional[int] = None,
) -> Tuple[OfflineBBOTask, TaskMetadata, TaskData]:
    task_desc = f"{task_name}_{benchmark_id}_{seed}"
    assert task_desc in TASKNAMES

    is_continuous = task_desc in CONTINUOUS_TASKS
    assert is_continuous, "Our work in SOO-Bench only support continuous tasks"

    task = SOOBenchTask(
        task_name=task_name,
        benchmark_id=benchmark_id,
        seed=seed,
        low=low,
        high=high,
        num_data=num_data,
    )

    xl, xu = task.bounds
    metadata = ContinuousTaskMetadata(
        input_dim=task.x_np.shape[1],
        bounds=[(l, u) for l, u in zip(xl, xu)],
        **TEXT_DESCRIPTIONS[task_desc],
    )
    data = ContinuousTaskData(task.x_np)

    return task, metadata, data

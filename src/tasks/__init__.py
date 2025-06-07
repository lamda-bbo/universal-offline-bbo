import os
from pathlib import Path
from typing import List, Optional, Tuple

from src.tasks.base import OfflineBBOTask

DESIGN_BENCH_TASKS = [
    "AntMorphology-Exact-v0",
    "DKittyMorphology-Exact-v0",
    "Superconductor-RandomForest-v0",
    "TFBind8-Exact-v0",
    "TFBind10-Exact-v0",
    # # Below are tasks that are not usually used in Design-Bench
    # "HopperController-Exact-v0",
]

SOO_BENCH_TASKS = [
    "gtopx_data_2_1",
    "gtopx_data_3_1",
    "gtopx_data_4_1",
    "gtopx_data_6_1",
]

def get_tasks(task_names: List[str], root_dir: Path) -> List[OfflineBBOTask]:
    tasks = []
    for task_entry in task_names:
        try:
            if task_entry in DESIGN_BENCH_TASKS:
                from src.tasks.design_bench_task import DesignBenchTask

                tasks.append(DesignBenchTask(task_name=task_entry, scale_up_ratio=2.0))
            elif task_entry in SOO_BENCH_TASKS:
                from src.tasks.soo_bench_task import SOOBenchTask

                task_name = task_entry[:10]
                benchmark_id, seed = task_entry[11:].split("_")
                benchmark_id = int(benchmark_id)
                seed = int(seed)
                tasks.append(
                    SOOBenchTask(
                        task_name=task_name,
                        benchmark_id=benchmark_id,
                        seed=seed,
                        low=25,
                        high=75,
                    )
                )
        except:
            raise ValueError(f"Unknown task entry: {task_entry}")
    return tasks


def get_tasks_from_suites(
    task_suites: str, root_dir: Path
) -> Tuple[List[str], List[OfflineBBOTask]]:
    assert task_suites.lower() in [
        "design_bench",
        "soo_bench",
    ]
    if task_suites.lower() == "design_bench":
        task_names = DESIGN_BENCH_TASKS
    elif task_suites.lower() == "soo_bench":
        task_names = SOO_BENCH_TASKS 
    else:
        raise ValueError(f"Unknown task suites {task_suites}")
    return task_names, get_tasks(task_names, root_dir)

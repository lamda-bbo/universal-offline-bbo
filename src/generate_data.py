import json
import os

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data2str.design_bench_data import TASKNAMES as TASKNAMES_DB
from src.data2str.design_bench_data import create_task as create_task_db
from src.data2str.soo_bench_data import TASKNAMES as TASKNAMES_SB
from src.data2str.soo_bench_data import create_task as create_task_sb

data_dir = root / "data"
os.makedirs(data_dir, exist_ok=True)

# Generate Design-Bench data
for task_name in TASKNAMES_DB:
    print(task_name)
    task, metadata, data = create_task_db(task_name)

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": round(y.item(), 4)})

    output_file = f"{data_dir}/{task_name}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)

    metadata_file = f"{data_dir}/{task_name}.metadata"
    with open(metadata_file, "w") as f:
        f.write(metadata.to_string())

# Generate SOO-Bench data
for benchmark_id in [2, 3, 4, 6]:
    task_desc = f"gtopx_data_{benchmark_id}_1"
    assert task_desc in TASKNAMES_SB
    print(task_desc)

    task, metadata, data = create_task_sb(
        "gtopx_data", benchmark_id, 1, low=25, high=75
    )
    print(task._evaluate(task.x_np[:5]), task.y_np[:5])
    print(task.full_y_min, task.full_y_max)
    print(task.y_np.min(), task.y_np.max())
    # such dataset settings follow Section 6 Para 1 in SOO-Bench paper
    # Link: https://openreview.net/pdf?id=bqf0aCF3Dd

    task_data = []
    for x, y in zip(data.to_string(), task.y_np):
        task_data.append({"x": x, "y": round(y.item(), 4)})

    output_file = f"{data_dir}/{task_desc}.json"
    with open(output_file, "w") as f:
        json.dump(task_data, f, indent=2)

    metadata_file = f"{data_dir}/{task_desc}.metadata"
    with open(metadata_file, "w") as f:
        f.write(metadata.to_string())

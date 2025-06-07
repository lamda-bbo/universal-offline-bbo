import os
from pathlib import Path
from typing import List, Union

import pandas as pd


def load_task_names(task_names: Union[str, List[str]], data_dir: Path) -> List[str]:
    if isinstance(task_names, list):
        return task_names
    if "," in task_names:
        task_names = list(task_names.split(","))
    else:
        task_names = [task_names]
    return task_names


def check_if_evaluated(
    results_dir: Path,
    task_name: str,
    model_name: str,
    seed: int,
    metric_name: str
) -> bool:
    csv_path = results_dir / f"{seed}-{metric_name}.csv"
    if not os.path.exists(csv_path):
        return False
    
    existing_df = pd.read_csv(csv_path, header=0, index_col=0)
    
    if task_name not in existing_df.index:
        return False
    
    if model_name not in existing_df.columns:
        return False
    
    return not pd.isna(existing_df.loc[task_name, model_name])



def save_metric_to_csv(
    results_dir: Path,
    task_name: str,
    model_name: str,
    seed: int,
    metric_value: float,
    metric_name: str,
) -> None:
    csv_path = results_dir / f"{seed}-{metric_name}.csv"
    result = {"task": task_name, f"{model_name}": metric_value}

    if not os.path.exists(csv_path):
        new_df = pd.DataFrame([result])
        new_df.to_csv(csv_path, index=False)
    else:
        existing_df = pd.read_csv(csv_path, header=0, index_col=0)
        updated_df = existing_df.copy()
        updated_df.loc[task_name, f"{model_name}"] = metric_value
        updated_df.to_csv(csv_path, index=True, mode="w")

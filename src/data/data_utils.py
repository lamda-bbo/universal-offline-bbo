from typing import List

import numpy as np
import torch
from scipy import stats


def get_good_std(y_unique: torch.Tensor, y_median: torch.Tensor) -> torch.Tensor:
    good_half = y_unique[y_unique >= y_median]
    if len(good_half) > 0:
        std = torch.sqrt(torch.mean((good_half - y_median) ** 2))
        if std > 0:
            return std

    std = torch.sqrt(torch.mean((y_unique - y_median) ** 2))
    if torch.isfinite(std):
        return std

    return torch.mean(torch.abs(y_unique - y_median))


def zero_mean_standardize(
    y: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor
) -> torch.Tensor:
    return torch.where(y_std != 0, (y - y_mean) / y_std, torch.zeros_like(y))


def min_max_y(y: torch.Tensor) -> torch.Tensor:
    y_min = y.min()
    y_max = y.max()
    return torch.where(
        y_max != y_min, (y - y_min) / (y_max - y_min), torch.full_like(y, 0.5)
    )


def log_warping(y: torch.Tensor) -> torch.Tensor:
    s = 1.5
    return 0.5 - torch.log(1 + (s - 1) * y) / torch.log(torch.tensor(s))


def handle_nan_values(y: torch.Tensor) -> torch.Tensor:
    y_no_nan = torch.nan_to_num(y, nan=float("inf"))
    y_min = torch.min(y_no_nan)
    y_no_nan = torch.nan_to_num(y, nan=float("-inf"))
    y_max = torch.max(y_no_nan)
    replacement_value = y_min - 0.5 * (y_max - y_min)
    return torch.where(torch.isnan(y), replacement_value, y)


def normalize_ys_from_different_tasks(
    values: List[float],
    task_names: List[str],
) -> torch.Tensor:
    assert len(values) == len(task_names)
    y_values = torch.tensor(values, dtype=torch.float32)

    unique_tasks = list(set(task_names))

    normalized_values = torch.zeros_like(y_values)

    for task in unique_tasks:
        task_idx = [i for i, t in enumerate(task_names) if t == task]

        task_values = y_values[task_idx]

        task_values = handle_nan_values(task_values)

        task_mean = torch.mean(task_values)
        task_unique = torch.unique(task_values)
        task_std = get_good_std(task_unique, task_mean)

        task_values = zero_mean_standardize(task_values, task_mean, task_std)

        bad_idx = task_values < task_mean
        if torch.any(bad_idx):
            ranks = torch.tensor(
                stats.rankdata(task_values.numpy()), dtype=torch.float32
            )
            bad_rank_percentiles = ranks[bad_idx] / len(task_values)
            bad_rank_percentiles = torch.clamp(bad_rank_percentiles, 0.001, 0.999)
            bad_z_scores = torch.tensor(
                stats.norm.ppf(bad_rank_percentiles.numpy()), dtype=torch.float32
            )
            task_values[bad_idx] = bad_z_scores

        task_values = min_max_y(task_values)
        task_values = log_warping(task_values)

        task_mean = torch.mean(task_values)
        task_values = task_values - task_mean

        normalized_values[task_idx] = task_values

    return normalized_values

import numpy as np
import torch
from lightning import LightningModule
from torch.optim import Adam

from src.searcher.base import BaseSearcher
from src.tasks.design_bench_task import DesignBenchTask


def inverse_batch_norm(
    normalized_x: torch.Tensor, batch_norm_layer: torch.nn.BatchNorm1d
):
    running_mean = batch_norm_layer.running_mean
    running_var = batch_norm_layer.running_var
    gamma = batch_norm_layer.weight
    beta = batch_norm_layer.bias
    eps = batch_norm_layer.eps

    std = torch.sqrt(running_var + eps)

    original_x = std * (normalized_x - beta) / gamma + running_mean

    return original_x


class AdamSearcher(BaseSearcher):
    def __init__(
        self,
        model: LightningModule,
        n_steps: int = 200,
        search_step_size: float = 1e-3,
        MAXIMIZE: bool = True,
        EVAL_STABILITY: bool = False,
        *args,
        **kwargs
    ) -> None:
        self.n_steps = n_steps
        self.search_step_size = search_step_size
        self.model = model
        super(AdamSearcher, self).__init__(*args, **kwargs)
        if self.task.task_type == "Categorical":
            self.n_steps = 100
            self.search_step_size = 1e-1
        self.EVAL_STABILITY = EVAL_STABILITY
        self.MAXIMIZE = MAXIMIZE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xl, self.xu = self.task.bounds
        if EVAL_STABILITY:
            self.X_all = []

    def _decode_x(self, x_res: torch.Tensor) -> np.ndarray:
        x_res = inverse_batch_norm(x_res, self.model.batch_norm).detach().cpu().numpy()
        if not isinstance(self.task, DesignBenchTask) and self.task.task_type in [
            "Continuous",
            "Integer",
        ]:
            x_res = np.clip(x_res, self.xl, self.xu)
        if self.task.task_type == "Categorical":
            x_res = x_res.reshape((-1,) + tuple(self.logits_shape))
            x_res = self.task.task.to_integers(x_res)
        elif self.task.task_type == "Integer":
            x_res = x_res.astype(np.int64)
        elif self.task.task_type == "Permutation":
            x_res = x_res.argsort().argsort()
        return x_res

    def run(self) -> np.ndarray:
        for p in self.model.parameters():
            p.requires_grad = False

        self.model = self.model.to(self.device)

        x_init, y_init = self.get_initial_designs(
            x=self.task.x_np, y=self.task.y_np, k=self.num_solutions
        )

        if self.task.task_type == "Categorical":
            x_init = self.task.task.to_logits(x_init)
            self.logits_shape = x_init.shape[1:]
            x_init = x_init.reshape(x_init.shape[0], -1)
        elif self.task.task_type in ["Integer", "Permutation"]:
            x_init = x_init.astype(np.float32)

        x_init = torch.from_numpy(x_init).to(self.device)
        y_init = torch.from_numpy(y_init).to(self.device)

        try:
            x_init = self.model.batch_norm(x_init)
        except:
            x_init = self.model.batch_norm(x_init.to(dtype=torch.float32))

        x_res = x_init.clone()
        x_res.requires_grad = True
        x_opt = Adam([x_res], lr=self.search_step_size)

        for _ in range(self.n_steps):
            x_opt.zero_grad()
            y_pred = torch.sum(self.model.layers(x_res))
            if self.MAXIMIZE:
                y_pred = y_pred * (-1)
            y_pred.backward()
            x_opt.step()
            if self.EVAL_STABILITY:
                self.X_all.append(self._decode_x(x_res))

        x_res = self._decode_x(x_res)
        return x_res

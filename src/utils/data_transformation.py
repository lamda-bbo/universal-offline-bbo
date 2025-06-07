import numpy as np
import torch
from lightning import LightningDataModule, LightningModule

from src.models. import OmniPredModule
from src.tasks.base import OfflineBBOTask


@torch.no_grad()
def omnipred_fitness_function_string(
    x: np.ndarray,
    m: str,
    model: OmniPredModule,
    task_name: str,
) -> np.ndarray:
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    assert len(x.shape) == 1 or len(x.shape) == 2
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    batch_size, n_var = x.shape
    ms = tuple([m for _ in range(batch_size)])

    def sol2str(single_solution):
        res_str = ", ".join(
            f"x{i}: {val.item():.4f}" if not task_name.startswith('TFBind') else f"x{i}: '{int(val.item())}'" for i, val in enumerate(single_solution)
        )
        return res_str

    x_str = [sol2str(x0) for x0 in x]
    input_str = [f"{m0}. {x0}" for x0, m0 in zip(x_str, ms)]
    input_tokens = model.input_tokenizer(
        input_str, padding="max_length", truncation=True, return_tensors="pt"
    )

    preds = model.generate_numbers(
        input_ids=input_tokens["input_ids"].to(model.device),
        attention_mask=input_tokens["attention_mask"].to(model.device),
    )

    preds = preds.cpu().numpy()
    assert len(preds) == batch_size
    return preds


@torch.no_grad()
def model_fitness_function_string(
    x: np.ndarray, m: str, model: LightningModule, datamodule: LightningDataModule,
    task_name: str,
) -> np.ndarray:
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    assert len(x.shape) == 1 or len(x.shape) == 2
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    batch_size, n_var = x.shape
    ms = tuple([m for _ in range(batch_size)])

    def sol2str(single_solution):
        res_str = ", ".join(
            f"x{i}: {val.item():.4f}" if not task_name.startswith('TFBind') else f"x{i}: '{int(val.item())}'" for i, val in enumerate(single_solution)
        )
        return res_str

    x_str = [sol2str(x0) for x0 in x]
    if datamodule.hparams.cat_metadata:
        x_str = [f"{m0}. {x0}" for x0, m0 in zip(x_str, ms)]
    x_tokens = datamodule.hparams.tokenizer(
        x_str,
        padding="max_length",
        max_length=datamodule.hparams.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    for k, v in x_tokens.items():
        x_tokens[k] = v.squeeze()

    m_tokens = datamodule.hparams.tokenizer(
        ms,
        padding="max_length",
        max_length=datamodule.hparams.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    for k, v in m_tokens.items():
        m_tokens[k] = v.squeeze()
    y_np = model(x_tokens).cpu().numpy()
    assert len(y_np) == batch_size

    return y_np
    
@torch.no_grad()
def model_fitness_function(
    x: np.ndarray, model: LightningModule, task: OfflineBBOTask
) -> np.ndarray:
    assert len(x.shape) == 1 or len(x.shape) == 2
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    batch_size, n_var = x.shape

    if task.task_type == "Categorical":
        x = task.task.to_logits(x)
        x = x.reshape(x.shape[0], -1)
    elif task.task_type in ["Integer", "Permutation"]:
        x = x.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.from_numpy(x).to(device, dtype=torch.float32)
    model = model.to(device)
    y_np = model(x_torch).cpu().numpy()
    assert len(y_np) == batch_size

    return y_np

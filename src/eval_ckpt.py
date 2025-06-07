import os
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.searcher.base import BaseSearcher
from src.tasks import get_tasks
from src.tasks.base import OfflineBBOTask
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    model_fitness_function_string,
    task_wrapper,
)
from src.utils.io_utils import load_task_names

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating task <{cfg.task._target_}>")
    task: OfflineBBOTask = hydra.utils.instantiate(cfg.task)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "task": task,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # exit()

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        assert cfg.ckpt_path
        ckpt_path = cfg.ckpt_path

        log.info(f"ckpt_path is {ckpt_path}")

        if ckpt_path is not None:
            log.info(f"loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            try:
                model.load_state_dict(checkpoint["state_dict"])
            except:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_key = k.replace("_orig_mod.", "")
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict)

        task_names = load_task_names(cfg.task_names, data_dir=root_dir / "data")
        tasks = get_tasks(task_names, root_dir=root_dir)
        score_dict = {}

        for task_name, task_instance in zip(task_names, tasks):
            log.info(f"Instantiating searcher <{cfg.searcher._target_}>")
            with open(f"./data/{task_name}.metadata", "r") as f:
                m = f.read()
            searcher: BaseSearcher = hydra.utils.instantiate(
                cfg.searcher,
                task=task_instance,
                score_fn=lambda x: model_fitness_function_string(
                    x, m=m, model=model, datamodule=datamodule
                ),
            )

            x_res = searcher.run()
            tmp_dict = task_instance.evaluate(x_res, return_normalized_y=True)
            res_dict = {}
            for k, v in tmp_dict.items():
                res_dict[f"{task_name}-{k}"] = v
            score_dict.update(res_dict)
            log.info("Final score statistics:")
            for score_desc, score in res_dict.items():
                log.info(f"{score_desc}: {score}")
                for logger0 in logger:
                    logger0.log_metrics({score_desc: score}, step=1)

    return score_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_ckpt.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = test(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
# logs/embed_regress_multitask_m_emb_from_scratch/runs/2025-01-10_00-05-41_seed42/checkpoints/last.ckpt
# logs/omnipred_24m/runs/2025-01-10_01-29-12_seed42/checkpoints/last.ckpt

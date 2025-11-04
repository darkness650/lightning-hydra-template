from typing import Any, Dict, List, Tuple, Optional

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# Additional deps: local file search and temp dir
import os
import glob
import tempfile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _get_cfg_value(cfg: DictConfig, path: List[str], default=None):
    """Safely get nested value from Hydra config."""
    cur = cfg
    try:
        for p in path:
            if cur is None:
                return default
            cur = cur.get(p)  # type: ignore[attr-defined]
        return cur if cur is not None else default
    except Exception:
        return default


def _to_py(value):
    """Convert OmegaConf containers (DictConfig/ListConfig) into plain Python types."""
    try:
        return OmegaConf.to_container(value, resolve=True)  # handles DictConfig/ListConfig
    except Exception:
        return value


def _resolve_ckpt_from_wandb(cfg: DictConfig) -> Optional[str]:
    """Select the latest finished W&B run by tags/group and resolve its checkpoint.

    Priority:
    1) Download a model Artifact (prefer alias 'best') and pick a .ckpt file.
    2) Fallback to run.summary['best_ckpt_path'].
    Return None if both are unavailable.
    """
    try:
        import wandb  # type: ignore
    except Exception:
        log.warning("WandB is not installed or unavailable; cannot resolve ckpt by tags. Please `pip install wandb.yaml`.")
        return None

    # Convert OmegaConf containers into native Python types for JSON filters
    # Note: query tags should be independent from this eval run's logger tags
    tags_val = (
        _get_cfg_value(cfg, ["search_tags"])              # explicit tags for querying
        or _get_cfg_value(cfg, ["tags"])                  # fallback to global tags
        or []
    )
    tags_py = _to_py(tags_val)
    if isinstance(tags_py, (tuple, set)):
        tags: List[str] = list(tags_py)
    elif isinstance(tags_py, list):
        tags = tags_py
    else:
        tags = [str(tags_py)] if tags_py else []


    # Resolve project/entity
    # Prefer cfg.logger.wandb.yaml.*, fallback to cfg.logger.wandb_eval.* for backward compatibility,
    # then to standalone overrides (wandb_project/wandb_entity)
    project = (
        _get_cfg_value(cfg, ["project"]) or
        _get_cfg_value(cfg, ["wandb_project"])
    )
    project = _to_py(project)

    entity = (
        _get_cfg_value(cfg, ["entity"]) or
        _get_cfg_value(cfg, ["wandb_entity"]) or
        None
    )
    entity = _to_py(entity)
    log.info(f"project: {project}, entity: {entity}")
    # Restrict to a specific group; default to training group 'train'
    search_group = _to_py(_get_cfg_value(cfg, ["search_group"]))

    use_artifacts: bool = bool(_to_py(_get_cfg_value(cfg, ["use_artifacts"], True)))

    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    filters: Dict[str, Any] = {"state": {"$in": ["finished"]}}
    if tags:
        # run must include all given tags
        filters["tags"] = {"$all": list(tags)}
    if search_group:
        filters["group"] = str(search_group)

    log.info(f"Query W&B with: project={project}, entity={entity}, group={search_group}, tags={tags}")

    try:
        runs = api.runs(project_path, filters=filters)
    except Exception as e:
        log.warning(f"Failed to fetch runs from W&B: {e}")
        return None

    if not runs:
        log.warning(f"No matching runs found in W&B project {project_path} for group={search_group}, tags={tags}.")
        return None

    def run_time(r):
        t = getattr(r, "created_at", None)
        if t is None:
            t = getattr(r, "updated_at", None)
        return t if t is not None else 0

    # Pick the latest run by time
    runs_sorted = sorted(runs, key=run_time, reverse=True)
    log.info(
        f"Found {len(runs_sorted)} matching runs. Selecting the latest one (run={getattr(runs_sorted[0], 'id', None) or getattr(runs_sorted[0], 'name', '')}) to resolve ckpt."
    )
    latest = runs_sorted[0]

    # Prefer resolving from model Artifacts
    if use_artifacts:
        try:
            model_arts = [a for a in latest.logged_artifacts() if getattr(a, "type", "") == "model"]
            chosen = None
            # Prefer alias 'best'
            for a in model_arts:
                aliases = [al if isinstance(al, str) else getattr(al, "name", str(al)) for al in getattr(a, "aliases", [])]
                if "best" in aliases:
                    chosen = a
                    break
            if chosen is None and model_arts:
                # Fallback to the most recent model artifact
                log.warning("Latest run has no model artifact with alias 'best'; falling back to the most recent model artifact.")
                chosen = model_arts[-1]
            if chosen is not None:
                tmpdir = tempfile.mkdtemp(prefix="wandb_ckpt_")
                local_dir = chosen.download(root=tmpdir)
                ckpts = glob.glob(os.path.join(local_dir, "**", "*.ckpt"), recursive=True)
                if ckpts:
                    ckpt_path = max(ckpts, key=lambda p: os.path.getsize(p))
                    log.info(
                        f"Downloaded ckpt from W&B Artifact (run={getattr(latest, 'id', None) or getattr(latest, 'name', '')}): {ckpt_path}"
                    )
                    return ckpt_path
        except Exception as e:
            log.warning(
                f"Failed to resolve ckpt from W&B Artifact (run={getattr(latest, 'id', None) or getattr(latest, 'name', '')}): {e}"
            )

    # Fallback: use summary.best_ckpt_path
    try:
        ckpt_path = latest.summary.get("best_ckpt_path")
        if ckpt_path:
            log.info(
                f"Using best_ckpt_path recorded in W&B Summary (run={getattr(latest, 'id', None) or getattr(latest, 'name', '')}): {ckpt_path}"
            )
            return ckpt_path
    except Exception:
        pass

    log.warning("The latest W&B run has no usable ckpt (no Artifact and no summary.best_ckpt_path).")
    return None


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset."""
    # Allow omitting ckpt_path and auto-resolve from W&B by tags
    ckpt_path_cfg = cfg.get("ckpt_path") if hasattr(cfg, "get") else None
    if ckpt_path_cfg in (None, "", "???"):
        log.info("No ckpt_path provided. Trying to resolve the latest checkpoint from W&B by tags...")
        resolved = _resolve_ckpt_from_wandb(cfg)
        if not resolved:
            raise AssertionError("ckpt_path not provided, and failed to resolve a usable checkpoint from W&B.")
        # Write back for logging/debugging
        cfg.ckpt_path = resolved  # type: ignore[attr-defined]

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation."""
    # apply extra utilities
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()

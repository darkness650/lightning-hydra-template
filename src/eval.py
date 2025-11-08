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
from datetime import datetime

from pytorch_lightning.loggers import WandbLogger

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

def _resolve_ckpt_from_swanlab(cfg: DictConfig) -> Optional[str]:
    """Query SwanLab OpenApi for the latest finished experiment and return its best_ckpt_path.

    Steps:
    - Call OpenApi.list_experiments to get experiments
    - Filter by cfg.search_group (or cfg.group) against profile.config.group.value
    - Keep only FINISHED/SUCCEEDED/DONE
    - Sort by finishedAt desc and take the first
    - Return profile.config.best_ckpt_path.value
    If no valid result is found or API fails, return None (no local fallback here).
    """
    # Resolve query params
    project = _to_py(_get_cfg_value(cfg, ["logger", "swanlab", "project"])) or _to_py(
        _get_cfg_value(cfg, ["swanlab_project"])
    )
    search_group = _to_py(_get_cfg_value(cfg, ["search_group"])) or _to_py(
        _get_cfg_value(cfg, ["group"])
    )

    log.info(f"SwanLab query params: project={project}, group={search_group}")

    # Step 1: Remote API query (OpenApi.list_experiments)
    try:
        import swanlab  # type: ignore

        api = None
        if hasattr(swanlab, "OpenApi"):
            try:
                api = getattr(swanlab, "OpenApi")()
            except Exception as e:
                log.warning(f"Init OpenApi failed: {e}")
                api = None
        if api is not None and hasattr(api, "list_experiments"):
            # Prefer passing project when available
            try:
                resp = api.list_experiments(project=project)
            except Exception as e:
                log.warning(f"list_experiments(project=...) failed: {e}")
                try:
                    resp = api.list_experiments()
                except Exception as e2:
                    log.warning(f"list_experiments() failed: {e2}")
                    resp = None

            # Parse ApiResponse[List[Experiment]]
            data = None
            try:
                if hasattr(resp, "data"):
                    data = getattr(resp, "data")
                elif isinstance(resp, dict):
                    data = resp.get("data")
            except Exception as e:
                log.warning(f"Read resp.data failed: {e}")
                data = None

            exps = data if isinstance(data, list) else []
            if exps:
                # Unified field access for Pydantic model or dict
                def get_field(exp: Any, key: str) -> Any:
                    try:
                        if isinstance(exp, dict):
                            return exp.get(key)
                        return getattr(exp, key)
                    except Exception as e:
                        log.warning(f"Read field '{key}' failed: {e}")
                        return None

                def parse_time(v: Any) -> float:
                    if not v:
                        return 0.0
                    try:
                        s = str(v).replace("Z", "+00:00")
                        return datetime.fromisoformat(s).timestamp()
                    except Exception as e:
                        log.warning(f"Parse time failed (value={v}): {e}")
                        return 0.0

                def is_finished(exp: Any) -> bool:
                    try:
                        state = str(get_field(exp, "state") or "").upper()
                        return state in {"FINISHED", "SUCCEEDED", "DONE"}
                    except Exception as e:
                        log.warning(f"Check finished state failed: {e}")
                        return False

                def profile_of(exp: Any) -> Dict:
                    try:
                        prof = get_field(exp, "profile")
                        return prof if isinstance(prof, dict) else {}
                    except Exception as e:
                        log.warning(f"Get profile failed: {e}")
                        return {}

                def group_of(exp: Any) -> Optional[str]:
                    try:
                        prof = profile_of(exp)
                        return (
                            prof.get("config", {})
                                .get("group", {})
                                .get("value")
                        )
                    except Exception as e:
                        log.warning(f"Extract group failed: {e}")
                        return None

                def best_ckpt_of(exp: Any) -> Optional[str]:
                    try:
                        prof = profile_of(exp)
                        return (
                            prof.get("config", {})
                                .get("best_ckpt_path", {})
                                .get("value")
                        )
                    except Exception as e:
                        log.warning(f"Extract best_ckpt_path failed: {e}")
                        return None

                def finished_at(exp: Any) -> Any:
                    return get_field(exp, "finishedAt")

                def name_of(exp: Any) -> str:
                    v = get_field(exp, "name")
                    return str(v) if v is not None else ""

                # Filter
                filtered = []
                for e in exps:
                    try:
                        if not is_finished(e):
                            continue
                        g = group_of(e)
                        if search_group and str(g) != str(search_group):
                            continue
                        filtered.append(e)
                    except Exception as err:
                        log.warning(f"Filter experiment failed: {err}")
                        continue

                if filtered:
                    try:
                        filtered.sort(key=lambda e: parse_time(finished_at(e)), reverse=True)
                    except Exception as e:
                        log.warning(f"Sort experiments failed: {e}")
                    top = filtered[0]
                    ck = best_ckpt_of(top)
                    if isinstance(ck, str) and ck.endswith(".ckpt"):
                        log.info(
                            f"Resolved best_ckpt_path via SwanLab OpenApi (exp={name_of(top)}): {ck}"
                        )
                        return ck
                    else:
                        log.warning("Latest finished experiment has no best_ckpt_path.")
        # If api is None or has no method
        return None
    except Exception as e:
        log.warning(f"SwanLab OpenApi query failed: {e}")
        return None


def _resolve_ckpt_from_wandb(cfg: DictConfig) -> Optional[str]:
    """Filter W&B runs and return best_ckpt_path from the most recent run that has it.

    Rules:
    - Filter finished runs by tags (cfg.search_tags or cfg.tags).
    - Support custom user-defined `group` (not W&B native group):
      * Try server-side filter via config.group when provided.
      * Always apply local filter via run.summary['group'] or run.config['group'].
    - Sort by time (created_at/updated_at) desc and pick the first with best_ckpt_path
      from run.summary or run.config.
    - No artifact downloading here.
    """
    try:
        import wandb  # type: ignore
    except Exception:
        log.warning("W&B is not installed; skip resolving checkpoint from W&B.")
        return None

    # tags for filtering
    tags_val = (
        _get_cfg_value(cfg, ["search_tags"]) or _get_cfg_value(cfg, ["tags"]) or []
    )
    tags_py = _to_py(tags_val)
    if isinstance(tags_py, (tuple, set)):
        tags: List[str] = list(tags_py)
    elif isinstance(tags_py, list):
        tags = tags_py
    else:
        tags = [str(tags_py)] if tags_py else []

    # project/entity
    project = _to_py(
        _get_cfg_value(cfg, ["logger","wandb","project"]) or _get_cfg_value(cfg, ["wandb_project"]) or None
    )
    entity = _to_py(
        _get_cfg_value(cfg, ["logger","wandb","entity"]) or _get_cfg_value(cfg, ["wandb_entity"]) or None
    )

    search_group = _to_py(_get_cfg_value(cfg, ["search_group"]))

    if not project:
        log.warning("W&B project is missing (set project or wandb_project in cfg).")
        return None

    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    # base filters
    filters: Dict[str, Any] = {"state": {"$in": ["finished"]}}
    # if tags:
    #     filters["tags"] = {"$all": list(tags)}
    # Best effort server-side filter for custom group stored in config
    if search_group:
        filters["summary_metrics.group"] = str(search_group)

    log.info(
        f"Querying W&B runs: project={project_path}, tags={tags}, custom_group={search_group}"
    )
    try:
        runs = api.runs(project_path, filters=filters)
    except Exception as e:
        log.warning(f"Failed to fetch runs from W&B: {e}")
        return None

    if not runs:
        log.warning("No finished W&B runs matched the base filters.")
        return None

    def run_time(r):
        return getattr(r, "updated_at", None) or getattr(r, "created_at", None) or 0

    try:
        runs_sorted = sorted(runs, key=run_time, reverse=True)
    except Exception as e:
        log.warning(f"Sorting W&B runs failed: {e}")
        runs_sorted = list(runs)

    for r in runs_sorted:
        try:
            # local custom group filter
            if search_group is not None:
                grp = None
                if hasattr(r, "summary_metrics") and isinstance(r.summary_metrics, dict):
                    grp = r.summary_metrics.get("group")
                if grp is None and hasattr(r, "config") and isinstance(r.config, dict):
                    grp = r.config.get("group")
                if str(grp) != str(search_group):
                    continue

            # accept best_ckpt_path from summary first, fallback to config
            ckpt = None
            if hasattr(r, "summary_metrics") and isinstance(r.summary_metrics, dict):
                ckpt = r.summary_metrics.get("best_ckpt_path")
            if ckpt is None and hasattr(r, "config") and isinstance(r.config, dict):
                ckpt = r.config.get("best_ckpt_path")

            if isinstance(ckpt, str) and ckpt.strip():
                log.info(
                    f"Resolved best_ckpt_path from W&B run id={getattr(r,'id','')} name={getattr(r,'name','')}: {ckpt}"
                )
                return ckpt
        except Exception as e:
            log.warning(f"Inspect W&B run failed: {e}")
            continue

    log.warning("No W&B run with a usable best_ckpt_path was found after filtering.")
    return None


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset."""
    # Allow omitting ckpt_path and auto-resolve from SwanLab or W&B
    ckpt_path_cfg = cfg.get("ckpt_path") if hasattr(cfg, "get") else None
    if ckpt_path_cfg in (None, "", "???", "null"):
        source = str(cfg.get("ckpt_source", "auto") or "auto").lower()
        resolved: Optional[str] = None
        if source in ("swanlab", "auto"):
            log.info("Trying to resolve latest ckpt from SwanLab...")
            resolved = _resolve_ckpt_from_swanlab(cfg)
        if not resolved and source in ("wandb", "auto"):
            log.info("SwanLab did not resolve ckpt, trying W&B...")
            resolved = _resolve_ckpt_from_wandb(cfg)
        if not resolved:
            raise AssertionError("ckpt_path not provided and auto-resolve via SwanLab/W&B failed. Please specify ckpt_path on the command line.")
        cfg.ckpt_path = resolved  # type: ignore[attr-defined]

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    for lg in logger:
        if isinstance(lg, WandbLogger):
            try:
                if lg.experiment is not None:
                    lg.experiment.summary["group"] = cfg.get("group", "eval")
            except Exception as e:
                log.warning(f"Failed to log group to WandB summary: {e}")
        if hasattr(lg, "log_hyperparameters"):
            try:
                lg.log_hyperparameters({"group": cfg.get("group", "eval")})
            except Exception as e:
                log.warning(f"Failed to log group via log_hyperparameters: {e}")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
        "group": cfg.get("group", "train"),
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
from __future__ import annotations

from typing import Any


def init_wandb(
    project: str,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    mode: str | None = None,
    enabled: bool = True,
):
    if not enabled:
        return None
    import wandb

    return wandb.init(
        project=project,
        config=config or {},
        name=name,
        entity=entity,
        group=group,
        tags=tags,
        notes=notes,
        mode=mode,
    )


def log_metrics(metrics: dict[str, Any], step: int | None = None, enabled: bool = True) -> None:
    if not enabled:
        return
    import wandb

    wandb.log(metrics, step=step)


def log_images(
    key: str,
    images: list[Any],
    step: int | None = None,
    enabled: bool = True,
) -> None:
    if not enabled or not images:
        return
    import wandb

    wandb.log({key: [wandb.Image(img) for img in images]}, step=step)


def save_artifact(path: str, enabled: bool = True) -> None:
    if not enabled:
        return
    import wandb

    wandb.save(path)

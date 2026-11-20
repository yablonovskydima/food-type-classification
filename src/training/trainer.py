import json
import math
import platform
import random
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.constants import NUM_CLASSES
from src.data.dataloader import create_dataloaders
from src.models import build_model
from src.training.checkpoint import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    check_resume_compatible,
    load_training_state,
    save_best_model,
    save_training_state,
)
from src.training.config import TrainConfig
from src.training.losses import build_criterion
from src.training.optim import build_optimizer, build_scheduler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _limited(loader: DataLoader, limit_batches: int | None):
    for index, batch in enumerate(loader):
        if limit_batches is not None and index >= limit_batches:
            return
        yield batch


def _summarise(loss_sum: float, labels: list[np.ndarray], preds: list[np.ndarray]) -> dict:
    targets = np.concatenate(labels)
    predictions = np.concatenate(preds)
    return {
        "loss": loss_sum / len(targets),
        "accuracy": float((targets == predictions).mean()),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=list(range(NUM_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    amp_dtype: torch.dtype = torch.float16,
    grad_clip_norm: float | None = None,
    limit_batches: int | None = None,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
    epoch: int = 0,
) -> tuple[dict[str, float], int]:
    model.train()
    loss_sum, labels, preds = 0.0, [], []
    progress = tqdm(_limited(loader, limit_batches), desc=f"Epoch {epoch} train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Training loss became {loss.item()} at step {global_step}")
        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()

        global_step += 1
        loss_sum += loss.item() * len(targets)
        labels.append(targets.cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        progress.set_postfix(loss=f"{loss.item():.3f}")
        if writer is not None:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
    if not labels:
        raise ValueError("The training loader produced no batches")
    return _summarise(loss_sum, labels, preds), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, labels, preds = 0.0, [], []
    for images, targets in tqdm(_limited(loader, limit_batches), desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, targets).item()
        labels.append(targets.cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    if not labels:
        raise ValueError("The validation loader produced no batches")
    return _summarise(loss_sum, labels, preds)


def fit(
    config: TrainConfig,
    *,
    device: str | None = None,
    resume: str | None = None,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    run_dir = config.run_checkpoint_dir
    if resume is None and (run_dir / LAST_CHECKPOINT).exists():
        raise FileExistsError(
            f"{run_dir} already contains a run. Resume it with --resume "
            f"{run_dir / LAST_CHECKPOINT}, or change 'experiment' to start a new one."
        )

    seed_everything(config.seed)
    torch_device = resolve_device(device)
    amp = config.amp and torch_device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp and torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float16
    )

    loaders = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        pin_memory=torch_device.type == "cuda",
        use_imagenet_stats=config.use_imagenet_stats,
        randaugment=config.randaugment,
        random_erasing_p=config.random_erasing_p,
        imbalance=config.imbalance,
        seed=config.seed,
    )
    train_loader, val_loader = loaders["training"], loaders["validation"]
    steps_per_epoch = len(train_loader)
    if limit_batches is not None:
        steps_per_epoch = min(steps_per_epoch, limit_batches)
    if steps_per_epoch == 0:
        raise ValueError("Training split is smaller than one batch; reduce batch_size")

    model = build_model(pretrained=config.pretrained).to(torch_device)
    criterion = build_criterion(
        config.imbalance, train_loader.dataset.targets, config.label_smoothing
    ).to(torch_device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch)
    scaler = torch.amp.GradScaler(torch_device.type, enabled=amp and amp_dtype == torch.float16)

    start_epoch, best_score, epochs_without_improvement = 1, -math.inf, 0
    best_epoch, history = 0, []
    if resume is not None:
        state = load_training_state(resume, map_location=torch_device)
        check_resume_compatible(state["config"], config)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        start_epoch = state["epoch"] + 1
        best_score = state["best_score"]
        best_epoch = state["best_epoch"]
        epochs_without_improvement = state["epochs_without_improvement"]
        history = state["history"]
        print(f"Resumed from {resume} at epoch {start_epoch}")

    global_step = (start_epoch - 1) * steps_per_epoch
    stopped_early = False
    writer = SummaryWriter(log_dir=str(config.run_log_dir))
    writer.add_text("config", f"```\n{json.dumps(config.to_dict(), indent=2)}\n```", start_epoch)
    try:
        for epoch in range(start_epoch, config.epochs + 1):
            if config.should_stop_early(epochs_without_improvement):
                stopped_early = True
                break
            started = time.perf_counter()
            train_metrics, global_step = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                torch_device,
                amp=amp,
                amp_dtype=amp_dtype,
                grad_clip_norm=config.grad_clip_norm,
                limit_batches=limit_batches,
                writer=writer,
                global_step=global_step,
                epoch=epoch,
            )
            val_metrics = evaluate(model, val_loader, torch_device, limit_batches=limit_batches)
            for name, value in train_metrics.items():
                writer.add_scalar(f"{name}/train", value, epoch)
            for name, value in val_metrics.items():
                writer.add_scalar(f"{name}/val", value, epoch)
            writer.flush()

            improved = val_metrics["macro_f1"] > best_score
            if improved:
                best_score, best_epoch, epochs_without_improvement = (
                    val_metrics["macro_f1"],
                    epoch,
                    0,
                )
                save_best_model(run_dir / BEST_CHECKPOINT, model, config, epoch, val_metrics)
            else:
                epochs_without_improvement += 1

            history.append(
                {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds": round(time.perf_counter() - started, 1),
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )
            save_training_state(
                run_dir / LAST_CHECKPOINT,
                epoch=epoch,
                config=config.to_dict(),
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict(),
                torch_rng_state=torch.get_rng_state(),
                best_score=best_score,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
            )
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f} | "
                f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3f} "
                f"macro-F1 {val_metrics['macro_f1']:.3f}{' *' if improved else ''}"
            )
            if config.should_stop_early(epochs_without_improvement):
                print(
                    f"Early stopping: no validation macro-F1 improvement for "
                    f"{epochs_without_improvement} epochs (best epoch {best_epoch})"
                )
                stopped_early = True
                break
    finally:
        writer.close()

    best_row = next((row for row in history if row["epoch"] == best_epoch), {})
    summary = {
        "experiment": config.experiment,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "best_val_metrics": {
            key.removeprefix("val_"): value
            for key, value in best_row.items()
            if key.startswith("val_")
        },
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "training_seconds": round(sum(row["seconds"] for row in history), 1),
        "hardware": hardware_info(torch_device),
        "precision": str(amp_dtype).removeprefix("torch.") if amp else "float32",
        "best_checkpoint": str(run_dir / BEST_CHECKPOINT),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def hardware_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["gpu"] = properties.name
        info["gpu_memory_gb"] = round(properties.total_memory / 1024**3, 1)
        info["cuda"] = torch.version.cuda
    return info

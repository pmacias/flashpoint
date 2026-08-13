"""Small CNN + Grad-CAM on early-window raster crops (Phase B).

Torch-only module -- kept separate from evaluation.py so the tabular
pipeline (env `flashpoint`) never needs a torch import; this module is
meant to run in `flashpoint_mps`. See docs/cnn_plan.md (Q1) for the full
architecture rationale; summary of the two choices most likely to surprise
a reader:

- **No rotation/flip augmentation.** The aspect and wind-direction planes
  are sin/cos encodings of an absolute compass bearing -- north-facing and
  south-facing slopes are physically different in the northern hemisphere,
  and flipping the image without counter-rotating those two channel pairs
  would silently create physically inconsistent inputs (exactly the kind
  of subtle bug this project avoids elsewhere). Crops are instead stored
  at 144px (`STORE_SIZE`) with random +/-`jitter_px` crop-center jitter at
  train time -- a translation-only augmentation that needs no channel
  correction, center-cropped to `crop_size` at eval.
- **Per-fold, per-seed validation split.** Model selection (early stop,
  LR plateau) uses a random 15% of the *training years'* events only,
  stratified by year x label and re-drawn per seed -- never the held-out
  test year. `leave_one_year_out_raster` asserts this explicitly before
  computing normalization stats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from flashpoint.rasters import default_plane_names


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def select_channels(crops: np.ndarray, plane_names: list[str], channels: list[str]) -> np.ndarray:
    """Slice a (n, len(plane_names), S, S) crop store down to named planes."""
    idx = [plane_names.index(c) for c in channels]
    return crops[:, idx]


class EarlyCropDataset(Dataset):
    """RAM-resident dataset over an already channel-selected crop array.

    `crops` is (n, C, store_size, store_size); `mean`/`std` are (C,) stats
    that MUST be computed from train-fold data only (see
    `leave_one_year_out_raster`) -- this class does not compute or fit its
    own stats, so it can't accidentally leak test-year pixels into
    normalization. After normalizing, NaNs (edge padding + the dataset's
    ~1-1.5% missing weather pixels) become 0 -- exactly the train mean,
    per plan.
    """

    def __init__(
        self,
        crops: np.ndarray,
        labels: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        crop_size: int = 128,
        jitter_px: int = 8,
        train: bool = True,
        channel_dropout_p: float = 0.0,
        seed: int = 0,
    ):
        self.crops = crops
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.crop_size = crop_size
        self.jitter_px = jitter_px
        self.train = train
        self.channel_dropout_p = channel_dropout_p
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, i: int):
        crop = self.crops[i]  # (C, S, S)
        store_size = crop.shape[-1]
        center_off = (store_size - self.crop_size) // 2
        if self.train:
            jy = center_off + int(self.rng.integers(-self.jitter_px, self.jitter_px + 1))
            jx = center_off + int(self.rng.integers(-self.jitter_px, self.jitter_px + 1))
            jy = min(max(jy, 0), store_size - self.crop_size)
            jx = min(max(jx, 0), store_size - self.crop_size)
        else:
            jy = jx = center_off

        patch = crop[:, jy:jy + self.crop_size, jx:jx + self.crop_size]
        norm = (patch - self.mean[:, None, None]) / self.std[:, None, None]
        norm = np.nan_to_num(norm, nan=0.0)

        if self.train and self.channel_dropout_p > 0:
            drop = self.rng.random(norm.shape[0]) < self.channel_dropout_p
            norm[drop] = 0.0

        return torch.from_numpy(norm.copy()), torch.tensor(self.labels[i])

    @staticmethod
    def compute_stats(crops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nan-aware per-channel mean/std over an (n, C, S, S) array."""
        mean = np.nanmean(crops, axis=(0, 2, 3))
        std = np.nanstd(crops, axis=(0, 2, 3))
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.norm(self.conv(x))))


class SmallFireCNN(nn.Module):
    """4 conv blocks (16->32->64->128) -> GAP -> Dropout -> Linear(128, 1).

    ~100-110k params depending on n_planes (asserted < 500k by callers).
    GroupNorm since fold batches are small; global average pooling both
    keeps the param count small and gives Grad-CAM a clean 8x8 map off the
    last block (128px input -> four 2x pools -> 8x8).
    """

    def __init__(self, n_planes: int, dropout: float = 0.3):
        super().__init__()
        channels = [n_planes, 16, 32, 64, 128]
        self.blocks = nn.ModuleList(
            [_ConvBlock(channels[i], channels[i + 1]) for i in range(4)]
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


def _train_impl(
    model: SmallFireCNN,
    train_ds: EarlyCropDataset,
    val_ds: EarlyCropDataset,
    device: torch.device,
    *,
    max_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    pos_weight: float | None,
) -> tuple[SmallFireCNN, float]:
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    pw = torch.tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                total_loss += criterion(model(xb), yb).item() * len(yb)
        val_loss = total_loss / len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def train_one(
    model: SmallFireCNN,
    train_ds: EarlyCropDataset,
    val_ds: EarlyCropDataset,
    device: torch.device,
    *,
    max_epochs: int = 60,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    patience: int = 10,
    pos_weight: float | None = None,
) -> tuple[SmallFireCNN, float]:
    """AdamW + plateau LR + early stop (restore best val-loss weights).

    Falls back to CPU if MPS raises (e.g. an unsupported op) -- restarts
    training on CPU from the model's current weights rather than failing
    the whole fold.
    """
    try:
        return _train_impl(
            model, train_ds, val_ds, device,
            max_epochs=max_epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, pos_weight=pos_weight,
        )
    except RuntimeError:
        if device.type != "mps":
            raise
        return _train_impl(
            model, train_ds, val_ds, torch.device("cpu"),
            max_epochs=max_epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, patience=patience, pos_weight=pos_weight,
        )


def leave_one_year_out_raster(
    crops: np.ndarray,
    plane_names: list[str],
    event_ids: np.ndarray,
    years: np.ndarray,
    labels: np.ndarray,
    channels: list[str] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    crop_size: int = 128,
    jitter_px: int = 8,
    val_frac: float = 0.15,
    max_epochs: int = 60,
    batch_size: int = 32,
    patience: int = 10,
    channel_dropout_p: float = 0.0,
    return_predictions: bool = False,
    return_models: bool = False,
) -> pd.DataFrame | tuple:
    """Raster-CNN mirror of evaluation.leave_one_year_out.

    `crops` is the full (n, len(plane_names), store_size, store_size) crop
    store; `event_ids`/`years`/`labels` are aligned to its rows. Per fold,
    validation is a random `val_frac` of the *training years'* events,
    stratified by year x label and re-drawn per seed -- the test year is
    never touched by the split, the normalization stats, or training.

    Returns a DataFrame with the SAME columns as
    `evaluation.leave_one_year_out` (test_year, n_test, accuracy,
    macro_f1, positive_recall) plus `seed`, so results concatenate
    directly against the tabular table. With `return_predictions=True`,
    also returns a second DataFrame of per-event out-of-fold predictions.
    With `return_models=True`, also returns a dict `{(test_year, seed):
    {"model": SmallFireCNN (on CPU), "mean": ndarray, "std": ndarray}}` --
    the fold's normalization stats travel with its model since Grad-CAM
    (or any other reuse) needs the exact preprocessing that model was
    trained under.
    """
    channels = channels or default_plane_names(plane_names)
    years = np.asarray(years)
    labels = np.asarray(labels, dtype=np.float32)
    event_ids = np.asarray(event_ids)
    device = get_device()

    sub = select_channels(crops, plane_names, channels)  # (n, C, S, S), once

    rows = []
    pred_frames = []
    models: dict[tuple, dict] = {}
    for seed in seeds:
        for test_year in sorted(np.unique(years)):
            test_idx = np.nonzero(years == test_year)[0]
            trainval_idx = np.nonzero(years != test_year)[0]

            strat_key = [f"{y}_{int(l)}" for y, l in zip(years[trainval_idx], labels[trainval_idx])]
            train_idx, val_idx = train_test_split(
                trainval_idx, test_size=val_frac, stratify=strat_key, random_state=seed,
            )

            # Normalization stats: train-years only. Required verification
            # step -- assert no test-year event leaks into the stats.
            assert not set(event_ids[trainval_idx]) & set(event_ids[test_idx])
            mean, std = EarlyCropDataset.compute_stats(sub[trainval_idx])

            n_contained = int((labels[train_idx] == 0).sum())
            n_escalates = int((labels[train_idx] == 1).sum())
            pos_weight = n_contained / n_escalates if n_escalates > 0 else 1.0

            train_ds = EarlyCropDataset(
                sub[train_idx], labels[train_idx], mean, std,
                crop_size=crop_size, jitter_px=jitter_px, train=True,
                channel_dropout_p=channel_dropout_p, seed=seed,
            )
            val_ds = EarlyCropDataset(
                sub[val_idx], labels[val_idx], mean, std,
                crop_size=crop_size, jitter_px=jitter_px, train=False, seed=seed,
            )
            test_ds = EarlyCropDataset(
                sub[test_idx], labels[test_idx], mean, std,
                crop_size=crop_size, jitter_px=jitter_px, train=False, seed=seed,
            )

            model = SmallFireCNN(n_planes=len(channels))
            n_params = count_params(model)
            assert n_params < 500_000, f"param count {n_params} exceeds 500k budget"

            model, _ = train_one(
                model, train_ds, val_ds, device,
                max_epochs=max_epochs, batch_size=batch_size,
                patience=patience, pos_weight=pos_weight,
            )

            model.eval()
            probs = []
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(next(model.parameters()).device)
                    probs.append(torch.sigmoid(model(xb)).cpu().numpy())
            probs = np.concatenate(probs)
            preds = (probs >= 0.5).astype(int)
            y_true = labels[test_idx].astype(int)

            if return_models:
                models[(test_year, seed)] = {
                    "model": model.to("cpu"), "mean": mean, "std": std,
                }

            rows.append({
                "test_year": test_year,
                "n_test": len(test_idx),
                "seed": seed,
                "accuracy": accuracy_score(y_true, preds),
                "macro_f1": f1_score(y_true, preds, average="macro"),
                "positive_recall": recall_score(y_true, preds, labels=[1], average=None)[0],
            })
            if return_predictions:
                pred_frames.append(pd.DataFrame({
                    "test_year": test_year, "seed": seed,
                    "event_id": event_ids[test_idx],
                    "y_true": y_true, "y_pred": preds, "p_escalate": probs,
                }))

    metrics = pd.DataFrame(rows)
    outputs: tuple = (metrics,)
    if return_predictions:
        outputs += (pd.concat(pred_frames, ignore_index=True),)
    if return_models:
        outputs += (models,)
    return outputs if len(outputs) > 1 else outputs[0]


def grad_cam(model: SmallFireCNN, x: torch.Tensor) -> np.ndarray:
    """Gradient-weighted class activation map off the last conv block.

    Hooks `model.blocks[-1]`'s output (128, 8, 8) for a 128px crop),
    backprops the (summed) logit, channel-weights by the mean gradient,
    ReLUs, and upsamples to the input's spatial size. `x` is (B, C, H, W)
    on the model's device; returns a (B, H, W) numpy array normalized to
    [0, 1] per sample.
    """
    model.eval()
    activation = {}
    gradient = {}

    def _forward_hook(_module, _inp, out):
        activation["value"] = out
        out.register_hook(lambda grad: gradient.__setitem__("value", grad))

    handle = model.blocks[-1].register_forward_hook(_forward_hook)
    try:
        model.zero_grad()
        logits = model(x)
        logits.sum().backward()

        acts = activation["value"]      # (B, 128, 8, 8)
        grads = gradient["value"]       # (B, 128, 8, 8)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))  # (B, 8, 8)
        cam = F.interpolate(
            cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()

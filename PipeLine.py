#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PlantVillage T4x2 research training script
==========================================

Experiments:
  1) ConvNeXt-Tiny + color
  2) ConvNeXt-Tiny + segmented
  3) EfficientNet-B0 + color
  4) EfficientNet-B0 + segmented

Research protocol:
- Stratified 80/10/10 split per class, seed=42.
- Same sample index is used across color/segmented representations.
- Class-Balanced Focal Loss (effective-number class weights + focal modulation).
- Custom attention/pooling:
    ConvNeXt-Tiny -> ECA -> GeM -> classifier
    EfficientNet-B0 -> CBAM -> GeM -> classifier
- Progressive fine-tuning:
    warmup custom layers/head with frozen backbone,
    then full fine-tuning with lower LR for pretrained backbone.
- Checkpoint selected by validation Macro-F1, accuracy as tie-breaker.
- T4x2 launcher runs 2 experiments concurrently, then the other 2.
- Optional final selection:
    choose best representation per backbone,
    evaluate complementarity,
    validation-tune weighted probability ensemble,
    evaluate test exactly once.

Expected Kaggle input:
  /kaggle/input/.../SOME_ROOT/
    color/
      CLASS_A.npy
      CLASS_B.npy
      ...
    segmented/
      CLASS_A.npy
      CLASS_B.npy
      ...

Each .npy must have shape approximately:
  (N, 224, 224, 3) uint8
or another common image layout convertible to RGB.

Typical Kaggle usage:
  !python /kaggle/working/train_plantvillage_t4x2.py --mode all

If auto-detection fails:
  !DATA_ROOT=/kaggle/input/your-dataset/plantvillage_npy \
   python /kaggle/working/train_plantvillage_t4x2.py --mode all

Requirements:
  torch, torchvision, timm, numpy, pandas, Pillow, scikit-learn, matplotlib
"""

import os
import gc
import sys
import json
import math
import time
import random
import hashlib
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

import timm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

SEED = int(os.environ.get("SEED", "42"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))

FREEZE_EPOCHS = int(os.environ.get("FREEZE_EPOCHS", "3"))
FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", "15"))
PATIENCE = int(os.environ.get("PATIENCE", "5"))

HEAD_LR = float(os.environ.get("HEAD_LR", "3e-4"))
BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "3e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))

CB_BETA = float(os.environ.get("CB_BETA", "0.999"))
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
PRETRAINED = os.environ.get("PRETRAINED", "1") == "1"

PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", "/kaggle/working/plantvillage_t4x2")
)
SPLIT_DIR = PROJECT_DIR / "splits"
RUN_DIR = PROJECT_DIR / "runs"
ENSEMBLE_DIR = PROJECT_DIR / "ensemble"

for d in [PROJECT_DIR, SPLIT_DIR, RUN_DIR, ENSEMBLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPERIMENTS = [
    ("convnext", "color"),
    ("convnext", "segmented"),
    ("effnet", "color"),
    ("effnet", "segmented"),
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # benchmark=True is faster on fixed-size images.
    torch.backends.cudnn.benchmark = True


seed_everything()


# ============================================================
# DATA DISCOVERY + SPLIT
# ============================================================

def find_data_root():
    # Fixed Kaggle dataset path requested by the user.
    # If color/ and segmented/ are directly inside this folder, use it.
    # Otherwise search recursively inside this dataset root for the actual parent.
    fixed_root = Path("/kaggle/input/datasets/leminhhung0101/plantvillage-npy-dataset")

    if (fixed_root / "color").is_dir() and (fixed_root / "segmented").is_dir():
        return fixed_root

    candidates = []
    if fixed_root.exists():
        for cur, dirs, _files in os.walk(fixed_root):
            curp = Path(cur)
            if "color" in dirs and "segmented" in dirs:
                n_color = len(list((curp / "color").glob("*.npy")))
                n_seg = len(list((curp / "segmented").glob("*.npy")))
                if n_color > 0 and n_seg > 0:
                    candidates.append((-(n_color + n_seg), len(curp.parts), curp))

    if candidates:
        candidates.sort()
        return candidates[0][2]

    # Optional override remains available if the Kaggle dataset structure changes.
    manual = os.environ.get("DATA_ROOT", "").strip()
    if manual:
        root = Path(manual)
        if (root / "color").is_dir() and (root / "segmented").is_dir():
            return root
        raise FileNotFoundError(
            f"DATA_ROOT={manual} does not contain color/ and segmented/."
        )

    base = Path("/kaggle/input")
    candidates = []

    if not base.exists():
        raise FileNotFoundError("/kaggle/input does not exist.")

    for cur, dirs, _files in os.walk(base):
        curp = Path(cur)
        if "color" in dirs and "segmented" in dirs:
            n_color = len(list((curp / "color").glob("*.npy")))
            n_seg = len(list((curp / "segmented").glob("*.npy")))
            if n_color > 0 and n_seg > 0:
                candidates.append((-(n_color + n_seg), len(curp.parts), curp))

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect a folder under /kaggle/input containing "
            "both color/*.npy and segmented/*.npy. Set DATA_ROOT manually."
        )

    candidates.sort()
    return candidates[0][2]


def build_map(folder):
    return {p.stem: p for p in sorted(Path(folder).glob("*.npy"))}


def stable_class_seed(class_name):
    h = hashlib.md5(class_name.encode("utf-8")).hexdigest()
    return SEED + int(h[:8], 16) % 1_000_000


def prepare_splits():
    """
    Robust split preparation.

    - Never deletes files under /kaggle/input.
    - Uses only .npy class files that exist in BOTH color/ and segmented/.
    - Unmatched files are ignored and logged.
    - If paired arrays have different lengths, only min(N_color, N_segmented)
      samples are used; extra tail samples are ignored and logged.
    - Same sample index is kept for color and segmented.
    """
    root = find_data_root()
    color_map = build_map(root / "color")
    seg_map = build_map(root / "segmented")

    color_classes = set(color_map)
    seg_classes = set(seg_map)

    shared = sorted(color_classes & seg_classes)
    only_color = sorted(color_classes - seg_classes)
    only_segmented = sorted(seg_classes - color_classes)

    if len(shared) < 2:
        raise RuntimeError(
            "Fewer than 2 shared class files remain after matching.\n"
            f"Only color: {only_color[:20]}\n"
            f"Only segmented: {only_segmented[:20]}"
        )

    unmatched_report = {
        "data_root": str(root),
        "only_in_color": only_color,
        "only_in_segmented": only_segmented,
        "num_only_in_color": len(only_color),
        "num_only_in_segmented": len(only_segmented),
        "num_shared_classes": len(shared),
        "policy": "Ignore unmatched source files; do not delete /kaggle/input.",
    }
    (SPLIT_DIR / "unmatched_files.json").write_text(
        json.dumps(unmatched_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if only_color:
        print(f"[WARN] Ignoring {len(only_color)} files only in color/")
        for x in only_color[:10]:
            print("  color-only:", x)

    if only_segmented:
        print(f"[WARN] Ignoring {len(only_segmented)} files only in segmented/")
        for x in only_segmented[:10]:
            print("  segmented-only:", x)

    rows = []
    pairing_rows = []

    for class_name in shared:
        ca = np.load(color_map[class_name], mmap_mode="r")
        sa = np.load(seg_map[class_name], mmap_mode="r")

        n_color = int(len(ca))
        n_segmented = int(len(sa))
        n_usable = min(n_color, n_segmented)

        if n_usable < 10:
            pairing_rows.append({
                "class_name": class_name,
                "color_count": n_color,
                "segmented_count": n_segmented,
                "paired_count_used": 0,
                "dropped_color_tail": n_color,
                "dropped_segmented_tail": n_segmented,
                "status": "skipped_too_few_pairs",
            })
            print(
                f"[WARN] Skipping {class_name}: only {n_usable} paired samples."
            )
            continue

        pairing_rows.append({
            "class_name": class_name,
            "color_count": n_color,
            "segmented_count": n_segmented,
            "paired_count_used": n_usable,
            "dropped_color_tail": max(0, n_color - n_usable),
            "dropped_segmented_tail": max(0, n_segmented - n_usable),
            "status": "matched" if n_color == n_segmented else "trimmed_to_min_count",
        })

        if n_color != n_segmented:
            print(
                f"[WARN] {class_name}: color={n_color}, segmented={n_segmented}; "
                f"using first {n_usable} paired indices."
            )

        rng = np.random.default_rng(stable_class_seed(class_name))
        perm = rng.permutation(n_usable)

        n_train = int(np.floor(0.80 * n_usable))
        n_valid = int(np.floor(0.10 * n_usable))

        split_indices = {
            "train": perm[:n_train],
            "valid": perm[n_train:n_train+n_valid],
            "test": perm[n_train+n_valid:],
        }

        for split, indices in split_indices.items():
            for sample_idx in indices.tolist():
                rows.append({
                    "split": split,
                    "class_name": class_name,
                    "label": -1,
                    "sample_idx": int(sample_idx),
                    "color_path": str(color_map[class_name]),
                    "segmented_path": str(seg_map[class_name]),
                })

    pairing_df = pd.DataFrame(pairing_rows)
    pairing_df.to_csv(SPLIT_DIR / "pairing_report.csv", index=False)

    if not rows:
        raise RuntimeError("No usable paired samples remain.")

    df = pd.DataFrame(rows)

    surviving_classes = sorted(df["class_name"].unique().tolist())
    if len(surviving_classes) < 2:
        raise RuntimeError("Fewer than 2 usable classes remain.")

    class_to_idx = {name: i for i, name in enumerate(surviving_classes)}
    df["label"] = df["class_name"].map(class_to_idx).astype(int)

    # Leakage check.
    audit = (
        df.groupby(["class_name", "sample_idx"])["split"]
        .nunique()
        .reset_index(name="n_splits")
    )
    if (audit["n_splits"] != 1).any():
        raise RuntimeError("Split leakage detected.")

    for split in ["train", "valid", "test"]:
        out_dir = SPLIT_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        sdf = df[df["split"] == split].reset_index(drop=True)
        sdf.to_csv(out_dir / "manifest.csv", index=False)

    (SPLIT_DIR / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "data_root": str(root),
        "seed": SEED,
        "split_ratio": {"train": 0.8, "valid": 0.1, "test": 0.1},
        "num_classes": len(surviving_classes),
        "class_names": surviving_classes,
        "counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "ignored_color_only_files": len(only_color),
        "ignored_segmented_only_files": len(only_segmented),
        "pairing_policy": (
            "Match by .npy filename stem. If lengths differ, use min count. "
            "Do not modify source files."
        ),
        "important_assumption": (
            "For each class, color.npy[i] and segmented.npy[i] must refer to "
            "the same original image."
        ),
    }
    (SPLIT_DIR / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 90)
    print("DATA ROOT:", root)
    print("SURVIVING CLASSES:", len(surviving_classes))
    print("SPLIT COUNTS:")
    print(df["split"].value_counts())
    print("Reports:")
    print(" ", SPLIT_DIR / "unmatched_files.json")
    print(" ", SPLIT_DIR / "pairing_report.csv")
    print("=" * 90)

    return df


# ============================================================
# DATASET
# ============================================================

def ensure_hwc3(arr):
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)

    elif arr.ndim == 3:
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] != 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)

    else:
        raise ValueError(f"Unsupported sample shape: {arr.shape}")

    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] != 3:
        raise ValueError(f"Could not convert sample to HWC RGB: {arr.shape}")

    return np.ascontiguousarray(arr)


def make_transform(train=True, representation="color"):
    ops = []

    if train:
        ops.extend(
            [
                T.RandomResizedCrop(
                    IMG_SIZE,
                    scale=(0.82, 1.0),
                    ratio=(0.90, 1.10),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.20),
                T.RandomRotation(
                    degrees=18,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=(124, 116, 104),
                ),
            ]
        )

        # Color branch can tolerate slightly stronger photometric augmentation.
        if representation == "color":
            ops.append(
                T.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.12,
                    hue=0.03,
                )
            )
        else:
            ops.append(
                T.ColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                    saturation=0.05,
                    hue=0.01,
                )
            )

    else:
        ops.append(
            T.Resize(
                (IMG_SIZE, IMG_SIZE),
                interpolation=InterpolationMode.BILINEAR,
            )
        )

    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return T.Compose(ops)


class LeafNpyDataset(Dataset):
    def __init__(self, manifest_csv, representation, train=False):
        self.df = pd.read_csv(manifest_csv)
        self.representation = representation
        self.path_col = f"{representation}_path"
        self.transform = make_transform(
            train=train,
            representation=representation,
        )
        self._memmaps = {}

    def __len__(self):
        return len(self.df)

    def _get_memmap(self, path):
        if path not in self._memmaps:
            self._memmaps[path] = np.load(path, mmap_mode="r")
        return self._memmaps[path]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = self._get_memmap(row[self.path_col])[int(row["sample_idx"])]
        arr = ensure_hwc3(np.array(arr, copy=True))
        img = Image.fromarray(arr)

        x = self.transform(img)
        y = int(row["label"])

        return x, y, i


# ============================================================
# CUSTOM LAYERS
# ============================================================

class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA-Net)."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k,
            padding=(k - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()

        hidden = max(8, channels // reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False,
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)

        channel_gate = torch.sigmoid(
            self.mlp(avg) + self.mlp(mx)
        )
        x = x * channel_gate

        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial = x.amax(dim=1, keepdim=True)

        spatial_gate = torch.sigmoid(
            self.spatial(
                torch.cat([avg_spatial, max_spatial], dim=1)
            )
        )

        return x * spatial_gate


class GeM(nn.Module):
    """Learnable Generalized Mean Pooling."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0, max=8.0)

        x = F.relu(x)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / p)

        return x.flatten(1)


class PlantDiseaseModel(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True):
        super().__init__()

        if backbone_name == "convnext":
            timm_name = "convnext_tiny"
        elif backbone_name == "effnet":
            timm_name = "efficientnet_b0"
        else:
            raise ValueError(backbone_name)

        try:
            self.backbone = timm.create_model(
                timm_name,
                pretrained=pretrained,
                features_only=True,
            )
        except Exception as e:
            if pretrained:
                raise RuntimeError(
                    f"Could not load pretrained weights for {timm_name}. "
                    "Enable Kaggle Internet or attach pretrained weights.\n"
                    f"Original error: {e}"
                )
            self.backbone = timm.create_model(
                timm_name,
                pretrained=False,
                features_only=True,
            )

        channels = self.backbone.feature_info.channels()[-1]

        if backbone_name == "convnext":
            self.attention = ECALayer(channels)
        else:
            self.attention = CBAM(channels)

        self.pool = GeM(p=3.0)

        hidden = min(512, channels)

        self.head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(hidden, num_classes),
        )

        self.backbone_name = backbone_name

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = self.attention(feat)
        feat = self.pool(feat)
        logits = self.head(feat)
        return logits


# ============================================================
# CLASS-BALANCED FOCAL LOSS
# ============================================================

def compute_class_balanced_weights(
    train_manifest,
    num_classes,
    beta=CB_BETA,
):
    df = pd.read_csv(train_manifest)

    counts = np.bincount(
        df["label"].values,
        minlength=num_classes,
    ).astype(np.float64)

    effective_num = 1.0 - np.power(beta, counts)

    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)

    # Normalize mean class weight to 1.
    weights = weights / weights.mean()

    return counts, torch.tensor(
        weights,
        dtype=torch.float32,
    )


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_weights,
        gamma=FOCAL_GAMMA,
    ):
        super().__init__()
        self.register_buffer(
            "class_weights",
            class_weights,
        )
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
        )

        pt = torch.softmax(
            logits,
            dim=1,
        ).gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        pt = pt.clamp(1e-6, 1.0)

        focal = (1.0 - pt).pow(self.gamma)

        return (focal * ce).mean()


# ============================================================
# METRICS
# ============================================================

def metrics_from_logits(logits, y):
    pred = logits.argmax(1)

    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y,
            pred,
            average="weighted",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),
    }


def score_probs(y, probs):
    pred = probs.argmax(1)

    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y,
            pred,
            average="weighted",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# TRAINING
# ============================================================

def build_loaders(representation):
    train_ds = LeafNpyDataset(
        SPLIT_DIR / "train" / "manifest.csv",
        representation=representation,
        train=True,
    )

    valid_ds = LeafNpyDataset(
        SPLIT_DIR / "valid" / "manifest.csv",
        representation=representation,
        train=False,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, valid_loader


def set_backbone_trainable(model, trainable):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def make_optimizer(model, backbone_trainable):
    groups = []

    if backbone_trainable:
        groups.append(
            {
                "params": [
                    p
                    for p in model.backbone.parameters()
                    if p.requires_grad
                ],
                "lr": BACKBONE_LR,
            }
        )

    new_params = []

    for module in [
        model.attention,
        model.pool,
        model.head,
    ]:
        new_params.extend(
            [
                p
                for p in module.parameters()
                if p.requires_grad
            ]
        )

    groups.append(
        {
            "params": new_params,
            "lr": HEAD_LR,
        }
    )

    return torch.optim.AdamW(
        groups,
        weight_decay=WEIGHT_DECAY,
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):
    model.train()

    total_loss = 0.0
    logits_all = []
    y_all = []

    amp_enabled = device.type == "cuda"

    for x, y, _idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        y = y.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=amp_enabled
        ):
            logits = model(x)
            loss = criterion(
                logits,
                y,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(y)

        logits_all.append(
            logits.detach().float().cpu()
        )
        y_all.append(y.detach().cpu())

    logits_all = torch.cat(
        logits_all
    ).numpy()

    y_all = torch.cat(
        y_all
    ).numpy()

    m = metrics_from_logits(
        logits_all,
        y_all,
    )

    m["loss"] = total_loss / len(y_all)

    return m


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    logits_all = []
    y_all = []
    row_idx = []

    amp_enabled = device.type == "cuda"

    for x, y, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )
        y = y.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=amp_enabled
        ):
            logits = model(x)
            loss = criterion(
                logits,
                y,
            )

        total_loss += loss.item() * len(y)

        logits_all.append(
            logits.float().cpu()
        )
        y_all.append(y.cpu())
        row_idx.extend(idx.tolist())

    logits_all = torch.cat(
        logits_all
    ).numpy()

    y_all = torch.cat(
        y_all
    ).numpy()

    m = metrics_from_logits(
        logits_all,
        y_all,
    )

    m["loss"] = total_loss / len(y_all)

    return (
        m,
        logits_all,
        y_all,
        np.asarray(row_idx),
    )


def experiment_name(backbone, representation):
    return f"{backbone}_{representation}"


def experiment_dir(backbone, representation):
    d = RUN_DIR / experiment_name(
        backbone,
        representation,
    )
    d.mkdir(
        parents=True,
        exist_ok=True,
    )
    return d


def train_experiment(backbone, representation):
    if not (
        SPLIT_DIR / "train" / "manifest.csv"
    ).exists():
        raise FileNotFoundError(
            "No split manifest found. "
            "Run --mode prepare first."
        )

    meta = json.loads(
        (
            SPLIT_DIR / "split_meta.json"
        ).read_text()
    )

    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "This training script is intended for Kaggle GPU."
        )

    run_dir = experiment_dir(
        backbone,
        representation,
    )

    print("=" * 90)
    print(
        f"EXPERIMENT: {backbone} + {representation}"
    )
    print("DEVICE:", device)
    print(
        "VISIBLE CUDA DEVICES:",
        os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "ALL",
        ),
    )
    print("GPU COUNT SEEN BY PROCESS:", torch.cuda.device_count())
    print("=" * 90)

    model = PlantDiseaseModel(
        backbone,
        num_classes,
        pretrained=PRETRAINED,
    ).to(device)

    counts, class_weights = compute_class_balanced_weights(
        SPLIT_DIR / "train" / "manifest.csv",
        num_classes,
    )

    criterion = ClassBalancedFocalLoss(
        class_weights.to(device),
        gamma=FOCAL_GAMMA,
    )

    print("CLASS COUNTS:", counts.astype(int).tolist())
    print(
        "CLASS WEIGHTS:",
        np.round(
            class_weights.numpy(),
            4,
        ).tolist(),
    )

    train_loader, valid_loader = build_loaders(
        representation,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=True
    )

    history = []

    best_f1 = -1.0
    best_acc = -1.0
    bad_epochs = 0
    global_epoch = 0

    phases = [
        (
            "head_warmup",
            FREEZE_EPOCHS,
            False,
        ),
        (
            "full_finetune",
            FINETUNE_EPOCHS,
            True,
        ),
    ]

    for phase_name, n_epochs, train_backbone in phases:
        set_backbone_trainable(
            model,
            train_backbone,
        )

        optimizer = make_optimizer(
            model,
            train_backbone,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, n_epochs),
            eta_min=1e-6,
        )

        print(
            f"\nPHASE={phase_name} "
            f"train_backbone={train_backbone}"
        )

        for _ in range(n_epochs):
            global_epoch += 1

            train_m = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
            )

            valid_m, _, _, _ = evaluate(
                model,
                valid_loader,
                criterion,
                device,
            )

            scheduler.step()

            lr_list = [
                pg["lr"]
                for pg in optimizer.param_groups
            ]

            row = {
                "epoch": global_epoch,
                "phase": phase_name,
                **{
                    f"train_{k}": v
                    for k, v in train_m.items()
                },
                **{
                    f"val_{k}": v
                    for k, v in valid_m.items()
                },
                "lr_groups": "|".join(
                    f"{x:.8g}"
                    for x in lr_list
                ),
                "gem_p": float(
                    model.pool.p.detach().cpu().item()
                ),
            }

            history.append(row)

            pd.DataFrame(history).to_csv(
                run_dir / "history.csv",
                index=False,
            )

            print(
                f"[{experiment_name(backbone, representation)}] "
                f"epoch={global_epoch:02d} "
                f"train_loss={train_m['loss']:.4f} "
                f"val_loss={valid_m['loss']:.4f} "
                f"val_acc={valid_m['accuracy']:.4f} "
                f"val_macroF1={valid_m['macro_f1']:.4f} "
                f"GeM_p={row['gem_p']:.3f}"
            )

            state = {
                "experiment": experiment_name(
                    backbone,
                    representation,
                ),
                "backbone": backbone,
                "representation": representation,
                "epoch": global_epoch,
                "model": model.state_dict(),
                "val_metrics": valid_m,
                "num_classes": num_classes,
                "class_names": class_names,
                "config": {
                    "seed": SEED,
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "beta": CB_BETA,
                    "gamma": FOCAL_GAMMA,
                    "head_lr": HEAD_LR,
                    "backbone_lr": BACKBONE_LR,
                    "weight_decay": WEIGHT_DECAY,
                    "pretrained": PRETRAINED,
                },
            }

            torch.save(
                state,
                run_dir / "last.pt",
            )

            improved = (
                valid_m["macro_f1"] > best_f1 + 1e-6
                or (
                    abs(
                        valid_m["macro_f1"] - best_f1
                    ) <= 1e-6
                    and valid_m["accuracy"] > best_acc
                )
            )

            if improved:
                best_f1 = valid_m["macro_f1"]
                best_acc = valid_m["accuracy"]
                bad_epochs = 0

                torch.save(
                    state,
                    run_dir / "best.pt",
                )

                print(
                    f">>> NEW BEST "
                    f"macroF1={best_f1:.5f} "
                    f"acc={best_acc:.5f}"
                )

            elif train_backbone:
                bad_epochs += 1

            if (
                train_backbone
                and bad_epochs >= PATIENCE
            ):
                print(
                    f"Early stopping: "
                    f"{bad_epochs} non-improving epochs."
                )
                break

        if (
            train_backbone
            and bad_epochs >= PATIENCE
        ):
            break

    print(
        f"DONE {experiment_name(backbone, representation)} | "
        f"best_macroF1={best_f1:.5f} | "
        f"best_acc={best_acc:.5f}"
    )


# ============================================================
# INFERENCE
# ============================================================

def load_best_model(
    backbone,
    representation,
    device,
):
    ckpt_path = (
        experiment_dir(
            backbone,
            representation,
        )
        / "best.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}"
        )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    model = PlantDiseaseModel(
        backbone,
        ckpt["num_classes"],
        pretrained=False,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.to(device)
    model.eval()

    return model, ckpt


@torch.no_grad()
def predict_split(
    backbone,
    representation,
    split,
    device,
):
    ds = LeafNpyDataset(
        SPLIT_DIR / split / "manifest.csv",
        representation=representation,
        train=False,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    model, ckpt = load_best_model(
        backbone,
        representation,
        device,
    )

    probs_all = []
    y_all = []
    idx_all = []

    for x, y, idx in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=True
        ):
            logits = model(x)

        probs = torch.softmax(
            logits.float(),
            dim=1,
        )

        probs_all.append(
            probs.cpu().numpy()
        )
        y_all.append(y.numpy())
        idx_all.extend(idx.tolist())

    probs_all = np.concatenate(
        probs_all,
        axis=0,
    )
    y_all = np.concatenate(
        y_all,
        axis=0,
    )
    idx_all = np.asarray(idx_all)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return (
        probs_all,
        y_all,
        idx_all,
        ckpt,
    )


def align_by_row_index(*triplets):
    """
    triplet = (probs, y, idx)
    Returns aligned versions in common row-index order.
    """

    aligned = []

    ref_idx = None
    ref_y = None

    for probs, y, idx in triplets:
        order = np.argsort(idx)

        probs = probs[order]
        y = y[order]
        idx = idx[order]

        if ref_idx is None:
            ref_idx = idx
            ref_y = y
        else:
            if not np.array_equal(
                ref_idx,
                idx,
            ):
                raise RuntimeError(
                    "Prediction row indices do not align."
                )

            if not np.array_equal(
                ref_y,
                y,
            ):
                raise RuntimeError(
                    "Ground-truth labels do not align."
                )

        aligned.append(
            (
                probs,
                y,
                idx,
            )
        )

    return aligned


# ============================================================
# MODEL SELECTION + DIVERSITY + ENSEMBLE
# ============================================================

def choose_best_representation_per_backbone():
    """
    Chooses the best VALID Macro-F1 representation for each backbone.
    """

    rows = []

    for backbone, representation in EXPERIMENTS:
        ckpt_path = (
            experiment_dir(
                backbone,
                representation,
            )
            / "best.pt"
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing {ckpt_path}. "
                "Train all 4 experiments first."
            )

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        row = {
            "backbone": backbone,
            "representation": representation,
            **{
                f"val_{k}": v
                for k, v in ckpt["val_metrics"].items()
            },
            "checkpoint": str(ckpt_path),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        ENSEMBLE_DIR / "all_experiments_valid.csv",
        index=False,
    )

    selected = {}

    for backbone in ["convnext", "effnet"]:
        sdf = df[
            df["backbone"] == backbone
        ].copy()

        sdf = sdf.sort_values(
            ["val_macro_f1", "val_accuracy"],
            ascending=False,
        )

        top = sdf.iloc[0]

        selected[backbone] = {
            "backbone": backbone,
            "representation": top["representation"],
            "val_macro_f1": float(
                top["val_macro_f1"]
            ),
            "val_accuracy": float(
                top["val_accuracy"]
            ),
        }

    (
        ENSEMBLE_DIR
        / "selected_models.json"
    ).write_text(
        json.dumps(
            selected,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("SELECTED MODELS:")
    print(
        json.dumps(
            selected,
            indent=2,
        )
    )

    return selected


def complementarity_stats(
    y,
    pred_a,
    pred_b,
):
    a_ok = pred_a == y
    b_ok = pred_b == y

    return {
        "both_correct": int(
            np.sum(a_ok & b_ok)
        ),
        "a_only_correct": int(
            np.sum(a_ok & ~b_ok)
        ),
        "b_only_correct": int(
            np.sum(~a_ok & b_ok)
        ),
        "both_wrong": int(
            np.sum(~a_ok & ~b_ok)
        ),
        "disagreement_rate": float(
            np.mean(pred_a != pred_b)
        ),
    }


def bootstrap_ci(
    y,
    pred,
    metric,
    n_boot=500,
    seed=SEED,
):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []

    for _ in range(n_boot):
        ix = rng.integers(
            0,
            n,
            n,
        )

        if metric == "accuracy":
            vals.append(
                accuracy_score(
                    y[ix],
                    pred[ix],
                )
            )

        elif metric == "macro_f1":
            vals.append(
                f1_score(
                    y[ix],
                    pred[ix],
                    average="macro",
                    zero_division=0,
                )
            )

        else:
            raise ValueError(metric)

    return [
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    ]


def build_final_ensemble():
    """
    Selection rule:
    1) Best representation per backbone from validation Macro-F1.
    2) Validation-tune alpha in:
         p_final = alpha * p_convnext + (1-alpha) * p_effnet
    3) Lock alpha.
    4) Evaluate test once.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "Final inference expects GPU."
        )

    selected = (
        ENSEMBLE_DIR
        / "selected_models.json"
    )

    if selected.exists():
        selected = json.loads(
            selected.read_text()
        )
    else:
        selected = choose_best_representation_per_backbone()

    conv_rep = selected["convnext"]["representation"]
    eff_rep = selected["effnet"]["representation"]

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    vc, vy1, vi1, cck = predict_split(
        "convnext",
        conv_rep,
        "valid",
        device,
    )

    ve, vy2, vi2, eck = predict_split(
        "effnet",
        eff_rep,
        "valid",
        device,
    )

    aligned = align_by_row_index(
        (vc, vy1, vi1),
        (ve, vy2, vi2),
    )

    vc, vy, vi = aligned[0]
    ve, _vy2, _vi2 = aligned[1]

    valid_div = complementarity_stats(
        vy,
        vc.argmax(1),
        ve.argmax(1),
    )

    # --------------------------------------------------------
    # VALIDATION GRID SEARCH FOR ALPHA
    # --------------------------------------------------------

    alpha_rows = []
    best = None

    for alpha in np.linspace(
        0.0,
        1.0,
        51,
    ):
        p = (
            alpha * vc
            + (1.0 - alpha) * ve
        )

        m = score_probs(
            vy,
            p,
        )

        row = {
            "alpha_convnext": float(alpha),
            "alpha_effnet": float(
                1.0 - alpha
            ),
            **m,
        }

        alpha_rows.append(row)

        key = (
            m["macro_f1"],
            m["accuracy"],
        )

        if (
            best is None
            or key > best[0]
        ):
            best = (
                key,
                float(alpha),
                m,
            )

    pd.DataFrame(
        alpha_rows
    ).to_csv(
        ENSEMBLE_DIR
        / "alpha_search_valid.csv",
        index=False,
    )

    alpha = best[1]

    # --------------------------------------------------------
    # TEST PREDICTIONS -- AFTER EVERYTHING IS LOCKED
    # --------------------------------------------------------

    tc, ty1, ti1, _ = predict_split(
        "convnext",
        conv_rep,
        "test",
        device,
    )

    te, ty2, ti2, _ = predict_split(
        "effnet",
        eff_rep,
        "test",
        device,
    )

    aligned_test = align_by_row_index(
        (tc, ty1, ti1),
        (te, ty2, ti2),
    )

    tc, ty, ti = aligned_test[0]
    te, _ty2, _ti2 = aligned_test[1]

    p_ens = (
        alpha * tc
        + (1.0 - alpha) * te
    )

    pred_c = tc.argmax(1)
    pred_e = te.argmax(1)
    pred_ens = p_ens.argmax(1)

    conv_test = score_probs(
        ty,
        tc,
    )
    eff_test = score_probs(
        ty,
        te,
    )
    ens_test = score_probs(
        ty,
        p_ens,
    )

    test_div = complementarity_stats(
        ty,
        pred_c,
        pred_e,
    )

    result = {
        "selected_convnext_representation": conv_rep,
        "selected_effnet_representation": eff_rep,
        "validation_diversity": valid_div,
        "alpha_convnext": alpha,
        "alpha_effnet": 1.0 - alpha,
        "validation_ensemble": best[2],
        "convnext_test": conv_test,
        "effnet_test": eff_test,
        "ensemble_test": ens_test,
        "test_diversity": test_div,
        "ensemble_accuracy_95ci": bootstrap_ci(
            ty,
            pred_ens,
            "accuracy",
        ),
        "ensemble_macro_f1_95ci": bootstrap_ci(
            ty,
            pred_ens,
            "macro_f1",
        ),
        "selection_rule": (
            "Best representation per backbone and ensemble alpha selected "
            "using validation only. Test evaluated once after locking."
        ),
    }

    (
        ENSEMBLE_DIR
        / "final_results.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    class_names = cck["class_names"]

    report = classification_report(
        ty,
        pred_ens,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report
    ).T.to_csv(
        ENSEMBLE_DIR
        / "classification_report.csv"
    )

    test_manifest = pd.read_csv(
        SPLIT_DIR
        / "test"
        / "manifest.csv"
    )

    order = np.argsort(ti)

    test_manifest = (
        test_manifest
        .iloc[order]
        .reset_index(drop=True)
    )

    test_manifest["pred_convnext"] = pred_c
    test_manifest["pred_effnet"] = pred_e
    test_manifest["pred_ensemble"] = pred_ens
    test_manifest["ensemble_confidence"] = p_ens.max(1)

    test_manifest.to_csv(
        ENSEMBLE_DIR
        / "test_predictions.csv",
        index=False,
    )

    # Confusion matrix figure
    import matplotlib.pyplot as plt

    cm = confusion_matrix(
        ty,
        pred_ens,
    )

    fig = plt.figure(
        figsize=(14, 12)
    )
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)

    ax.set_title(
        "Final Weighted Ensemble Confusion Matrix"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(
        range(len(class_names))
    )
    ax.set_yticks(
        range(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=6,
    )
    ax.set_yticklabels(
        class_names,
        fontsize=6,
    )

    fig.colorbar(
        im,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        ENSEMBLE_DIR
        / "confusion_matrix.png",
        dpi=180,
    )

    plt.close(fig)

    (
        ENSEMBLE_DIR
        / "ensemble_config.json"
    ).write_text(
        json.dumps(
            {
                "convnext": {
                    "representation": conv_rep,
                    "checkpoint": str(
                        experiment_dir(
                            "convnext",
                            conv_rep,
                        )
                        / "best.pt"
                    ),
                },
                "effnet": {
                    "representation": eff_rep,
                    "checkpoint": str(
                        experiment_dir(
                            "effnet",
                            eff_rep,
                        )
                        / "best.pt"
                    ),
                },
                "alpha_convnext": alpha,
                "alpha_effnet": 1.0 - alpha,
                "class_names": class_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 90)
    print("FINAL ENSEMBLE RESULT")
    print("=" * 90)
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


# ============================================================
# T4x2 LAUNCHER
# ============================================================

def current_script_path():
    return str(
        Path(__file__).resolve()
    )


def launch_one(
    gpu_id,
    backbone,
    representation,
):
    run_dir = experiment_dir(
        backbone,
        representation,
    )

    log_path = (
        run_dir
        / "console.log"
    )

    env = os.environ.copy()

    # Each child process sees exactly one T4 as cuda:0.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        current_script_path(),
        "--mode",
        "train-one",
        "--backbone",
        backbone,
        "--representation",
        representation,
    ]

    log_f = open(
        log_path,
        "w",
        buffering=1,
    )

    print(
        f"Launching GPU {gpu_id}: "
        f"{backbone} + {representation}"
    )

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return proc, log_f, log_path


def wait_group(group):
    failures = []

    for proc, log_f, log_path, label in group:
        rc = proc.wait()
        log_f.close()

        if rc != 0:
            failures.append(
                (
                    label,
                    rc,
                    str(log_path),
                )
            )

    if failures:
        msg = "\n".join(
            [
                f"{label}: exit={rc}, log={log_path}"
                for label, rc, log_path in failures
            ]
        )
        raise RuntimeError(
            "One or more training jobs failed:\n"
            + msg
        )


def train_all_t4x2():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected."
        )

    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Need T4x2 / 2 visible GPUs. "
            f"Detected {torch.cuda.device_count()}."
        )

    # Group 1:
    # GPU0 -> ConvNeXt + color
    # GPU1 -> ConvNeXt + segmented
    group1_specs = [
        (
            0,
            "convnext",
            "color",
        ),
        (
            1,
            "convnext",
            "segmented",
        ),
    ]

    # Group 2:
    # GPU0 -> EfficientNet + color
    # GPU1 -> EfficientNet + segmented
    group2_specs = [
        (
            0,
            "effnet",
            "color",
        ),
        (
            1,
            "effnet",
            "segmented",
        ),
    ]

    for group_idx, specs in enumerate(
        [group1_specs, group2_specs],
        start=1,
    ):
        print("=" * 90)
        print(
            f"STARTING TRAINING GROUP {group_idx}/2"
        )
        print("=" * 90)

        group = []

        for (
            gpu_id,
            backbone,
            representation,
        ) in specs:
            proc, log_f, log_path = launch_one(
                gpu_id,
                backbone,
                representation,
            )

            group.append(
                (
                    proc,
                    log_f,
                    log_path,
                    experiment_name(
                        backbone,
                        representation,
                    ),
                )
            )

        wait_group(group)

        print(
            f"GROUP {group_idx}/2 COMPLETED."
        )

    print(
        "All 4 experiments completed."
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_runs():
    rows = []

    for backbone, representation in EXPERIMENTS:
        ckpt_path = (
            experiment_dir(
                backbone,
                representation,
            )
            / "best.pt"
        )

        if not ckpt_path.exists():
            continue

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

        rows.append(
            {
                "backbone": backbone,
                "representation": representation,
                **{
                    f"val_{k}": v
                    for k, v in ckpt["val_metrics"].items()
                },
                "checkpoint": str(ckpt_path),
            }
        )

    if not rows:
        print(
            "No completed experiment checkpoints found."
        )
        return

    df = pd.DataFrame(rows)

    df = df.sort_values(
        [
            "val_macro_f1",
            "val_accuracy",
        ],
        ascending=False,
    )

    print(df.to_string(index=False))

    df.to_csv(
        PROJECT_DIR
        / "experiment_summary.csv",
        index=False,
    )


# ============================================================
# SAFE CLEANUP
# ============================================================

def cleanup_working_artifacts():
    """
    Delete only generated outputs under /kaggle/working.
    Never delete anything under /kaggle/input.
    """
    import shutil

    targets = [
        SPLIT_DIR,
        RUN_DIR,
        ENSEMBLE_DIR,
        PROJECT_DIR / "experiment_summary.csv",
    ]

    print("Cleaning generated artifacts only...")

    for target in targets:
        target = Path(target)
        resolved = str(target.resolve())

        if resolved.startswith("/kaggle/input"):
            raise RuntimeError(
                f"Safety stop: refusing to delete input path: {target}"
            )

        if target.is_dir():
            print("  removing dir:", target)
            shutil.rmtree(target)
        elif target.is_file():
            print("  removing file:", target)
            target.unlink()

    for d in [SPLIT_DIR, RUN_DIR, ENSEMBLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("Cleanup completed. Kaggle input dataset was not modified.")


# ============================================================
# MAIN
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        required=True,
        choices=[
            "prepare",
            "train-one",
            "train-all",
            "select",
            "ensemble",
            "summary",
            "cleanup",
            "all",
        ],
    )

    ap.add_argument(
        "--backbone",
        choices=[
            "convnext",
            "effnet",
        ],
    )

    ap.add_argument(
        "--representation",
        choices=[
            "color",
            "segmented",
        ],
    )

    return ap.parse_args()


def main():
    args = parse_args()

    if args.mode == "prepare":
        prepare_splits()

    elif args.mode == "train-one":
        if (
            args.backbone is None
            or args.representation is None
        ):
            raise ValueError(
                "--backbone and --representation "
                "are required for --mode train-one."
            )

        train_experiment(
            args.backbone,
            args.representation,
        )

    elif args.mode == "train-all":
        train_all_t4x2()

    elif args.mode == "select":
        choose_best_representation_per_backbone()

    elif args.mode == "ensemble":
        build_final_ensemble()

    elif args.mode == "summary":
        summarize_runs()

    elif args.mode == "cleanup":
        cleanup_working_artifacts()

    elif args.mode == "all":
        print("STEP 1/4: PREPARE SPLIT")
        prepare_splits()

        print("\nSTEP 2/4: TRAIN 4 EXPERIMENTS ON T4x2")
        train_all_t4x2()

        print("\nSTEP 3/4: SELECT BEST REPRESENTATION PER BACKBONE")
        choose_best_representation_per_backbone()

        print("\nSTEP 4/4: BUILD VALIDATION-WEIGHTED ENSEMBLE + TEST")
        build_final_ensemble()

        print("\nDONE.")
        print(
            f"Artifacts saved under: {PROJECT_DIR}"
        )


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import math, os, random, time, glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchxrayvision as xrv
import torchvision.models as tvm

from bpd_torch.models.model import BPDModel
from bpd_torch.data.dataset_fold import BPDDataset_5fold, BPDDatasetRGB_5fold
from bpd_torch.data.augmentor import get_augmentations, cut_mixup_collate as mixup_collate


from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryConfusionMatrix
)


@dataclass
class Experiment:
    name: str
    backbone: str
    freezing: bool
    mixup: bool
    epochs: int = 30
    runs: int = 6
    unfreeze_layer4: float = 0.10
    unfreeze_layer3: float = 0.30
    unfreeze_layer2: float = 0.60
    init_mixup_prob: float = 0.5
    mixup_alpha: float = 1
    probe_epochs: int = 0
    probe_lr: float = 1e-3
    probe_mixup: bool = True


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_folds(pos_ids, neg_ids, k=5, seed=0):
    rng = np.random.RandomState(seed)
    pos_ids = rng.permutation(pos_ids)
    neg_ids = rng.permutation(neg_ids)
    pos_chunks = np.array_split(pos_ids, k)
    neg_ptr = 0
    for chunk in pos_chunks:
        n = len(chunk)
        neg_chunk = neg_ids[neg_ptr:neg_ptr + n]
        neg_ptr += n
        yield np.concatenate([chunk, neg_chunk])


def run_linear_probe(model, dl_train, dl_test, *,
                     epochs: int, lr: float, device: str) -> None:
    """Train *only* the classifier head for `epochs` epochs."""
    opt = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    crit, scaler = nn.BCEWithLogitsLoss(), GradScaler()

    for ep in range(epochs):
        model.train();
        correct = total = 0
        for batch in dl_train:
            if isinstance(batch[1], (list, tuple)):
                x, (ya, yb, lam) = batch
                x, ya, yb = x.to(device), ya.to(device), yb.to(device)
            else:
                x, ya = batch
                x, ya = x.to(device), ya.to(device)
                yb = None
                lam = 1.0

            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(x)
                loss = crit(out, ya.unsqueeze(1)) if yb is None else \
                    lam * crit(out, ya.unsqueeze(1)) + (1 - lam) * crit(out, yb.unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(opt);
            scaler.update()

        model.eval();
        hits = n = 0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)
                hits += ((model(x) > 0).cpu().numpy().flatten() == y.cpu().numpy()).sum()
                n += len(y)
        print(f"  [LP] epoch {ep + 1}/{epochs}  test acc {hits / n:.3f}")


def run_one_fold(exp: Experiment, run_idx: int, fold_idx: int,
                 master_df: pd.DataFrame, train_patient_ids: np.ndarray,
                 test_patient_ids: np.ndarray) -> dict:
    set_seed(1000 + run_idx + fold_idx)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    probe_done = False

    # 1) backbone ----------------------------------------------------------
    if exp.backbone == "xrv":
        base = xrv.models.ResNet(weights="resnet50-res512-all")
    elif exp.backbone == "torchvision":
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError("backbone must be 'xrv' or 'torchvision'")

    # 2) model & optimiser -------------------------------------------------
    model = BPDModel(base).to(device)
    model.compile(freeze_backbone=exp.freezing, lr_backbone=2e-4, lr_head=1e-3)

    optimizer = model.get_optimizer()
    if exp.freezing:
        assert len(optimizer.param_groups) == 3, \
            "compile() must create separate groups for layer3, layer4 and head."
        optimizer.add_param_group({
            "params": model.inner_model.layer2.parameters(),
            "lr": 1e-4,
            "weight_decay": 1e-4,
        })

    # 3) data --------------------------------------------------------------
    p_pos = master_df[master_df.patient_id.isin(test_patient_ids) & (master_df.label == 1)].patient_id.nunique()
    p_neg = master_df[master_df.patient_id.isin(test_patient_ids) & (master_df.label == 0)].patient_id.nunique()

    if exp.backbone == "torchvision":
        ds_train = BPDDatasetRGB_5fold(master_df, train_patient_ids, stage="train",
                                       augmentor=get_augmentations(model.get_resolution()))
        ds_test = BPDDatasetRGB_5fold(master_df, test_patient_ids, stage="test")
    else:
        ds_train = BPDDataset_5fold(master_df, train_patient_ids, stage="train",
                                    augmentor=get_augmentations(model.get_resolution()))
        ds_test = BPDDataset_5fold(master_df, test_patient_ids, stage="test")

    print(
        f"  [Fold {fold_idx + 1}] Train: {len(ds_train)} images, Test: {len(ds_test)} images | Test patients: {p_pos} pos / {p_neg} neg")

    y = ds_train.data.label.to_numpy()
    class_w = 1.0 / np.bincount(y)
    sample_w = [class_w[i] for i in y]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    collate = None
    if exp.mixup:
        collate = lambda b: mixup_collate(b, alpha=exp.mixup_alpha, prob=exp.init_mixup_prob)

    dl_train = DataLoader(ds_train, batch_size=8, sampler=sampler, num_workers=12, pin_memory=True, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # 4) scheduler (One‑Cycle, discriminative LR) -------------------------
    max_lr = [1e-3, 2.5e-4, 5e-4, 1e-4]
    if not exp.freezing:
        max_lr = max_lr[:2]

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=exp.epochs,
                                              steps_per_epoch=len(dl_train), pct_start=0.3, anneal_strategy="cos")
    crit = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    uf4, uf3, uf2 = math.floor(exp.unfreeze_layer4 * exp.epochs), math.floor(
        exp.unfreeze_layer3 * exp.epochs), math.floor(exp.unfreeze_layer2 * exp.epochs)
    best_acc = 0.0

    if exp.probe_epochs > 0:
        print(f"[run {run_idx}, fold {fold_idx + 1}] starting {exp.probe_epochs}-epoch linear probe")
        if not exp.probe_mixup:
            # we do mixup only after linear probing:
            dl_train_probe = DataLoader(ds_train, batch_size=8, sampler=sampler, num_workers=12, pin_memory=True,
                                  collate_fn=None)
            run_linear_probe(model, dl_train_probe, dl_test, epochs=exp.probe_epochs, lr=exp.probe_lr, device=device)
        else:
            run_linear_probe(model, dl_train, dl_test, epochs=exp.probe_epochs, lr=exp.probe_lr, device=device)


    # 5) training loop -----------------------------------------------------
    for epoch in range(exp.epochs):
        if exp.freezing:
            if epoch == uf4: print(f"[epoch {epoch}] unfreezing layer4"); model.unfreeze_layers(
                model.inner_model.layer4)
            if epoch == uf3: print(f"[epoch {epoch}] unfreezing layer3"); model.unfreeze_layers(
                model.inner_model.layer3)
            if epoch == uf2: print(f"[epoch {epoch}] unfreezing layer2"); model.unfreeze_layers(
                model.inner_model.layer2)

        model.train()
        for batch in dl_train:
            if isinstance(batch[1], (list, tuple)):
                x, (ya, yb, lam) = batch
                x, ya, yb = x.to(device), ya.to(device), yb.to(device)
            else:
                x, ya = batch;
                x, ya = x.to(device), ya.to(device);
                yb, lam = None, 1.0

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(x)
                loss = crit(out, ya.unsqueeze(1)) if yb is None else lam * crit(out, ya.unsqueeze(1)) + (
                        1 - lam) * crit(out, yb.unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer);
            scaler.update();
            scheduler.step()

        model.eval();
        correct_te = total_te = 0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)
                preds = (model(x) > 0).cpu().numpy().flatten()
                correct_te += (preds == y.cpu().numpy()).sum();
                total_te += len(y)
        test_acc = correct_te / total_te
        best_acc = max(best_acc, test_acc)
        print(
            f"{exp.name} | run {run_idx} | fold {fold_idx + 1} | epoch {epoch + 1}/{exp.epochs} | test acc {test_acc:.3f}")

    # train_eval_loader = DataLoader(ds_train, batch_size=16, shuffle=False, num_workers=4)
    # correct = total = 0
    # with torch.no_grad():
    #     for x, y in train_eval_loader:
    #         x, y = x.to(device), y.to(device)
    #         preds = (model(x) > 0).cpu().numpy().flatten()
    #         correct += (preds == y.cpu().numpy()).sum();
    #         total += len(y)
    # train_acc = correct / total

    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        for x, y in dl_test:
            x, y = x.to(device), y.to(device)
            # Store raw model outputs (logits) for AUROC
            # and true labels
            preds = model(x).squeeze()  # The model returns logits
            all_preds.append(preds.cpu())
            all_y.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_y = torch.cat(all_y).int()  # Ensure labels are integers

    # Initialize metric calculators
    auroc = BinaryAUROC()
    f1 = BinaryF1Score()
    precision = BinaryPrecision()
    recall = BinaryRecall()  # Sensitivity
    specificity = BinarySpecificity()
    conf_matrix = BinaryConfusionMatrix()

    # Calculate all metrics
    final_auroc = auroc(all_preds, all_y).item()
    final_f1 = f1(all_preds, all_y).item()
    final_precision = precision(all_preds, all_y).item()
    final_recall = recall(all_preds, all_y).item()
    final_specificity = specificity(all_preds, all_y).item()

    # For accuracy, we can use the binarized predictions
    binary_preds = (all_preds > 0).int()
    test_acc = (binary_preds == all_y).float().mean().item()

    # The confusion matrix gives you TN, FP, FN, TP
    cm = conf_matrix(all_preds, all_y).numpy()
    tn, fp, fn, tp = cm.ravel()

    return {
        "experiment": exp.name, "run": run_idx, "fold": fold_idx + 1,
        "test_accuracy": test_acc,
        "auroc": final_auroc,
        "f1_score": final_f1,
        "sensitivity": final_recall, # Recall is Sensitivity
        "specificity": final_specificity,
        "precision": final_precision,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


EXPERIMENTS: List[Experiment] = [
    # ── Chest-X-ray-pre-trained backbone (1-channel, XRV weights) ──
    Experiment(
        name="lpft_progressive_xrv_mixup_lpnomix",
        backbone="xrv",
        freezing=True,
        probe_epochs=10,
        mixup=True,
        epochs=20,
        # unfreeze_layer2=0.80,
        probe_mixup=False,
    ),
    Experiment(
        name="lpft_progressive_xrv_mixup_lpnomix_40epoch",
        backbone="xrv",
        freezing=True,
        probe_epochs=10,
        mixup=True,
        epochs=30,
        # unfreeze_layer2=0.80,
        probe_mixup=False,
    ),
    Experiment(name="lpft_progressive_xrv",
               backbone="xrv", freezing=True, mixup=False,
               probe_epochs=10, epochs=20),
    Experiment(name="lp_full_finetune_xrv",
               backbone="xrv", freezing=False, mixup=False,
               probe_epochs=10, epochs=20),
    Experiment(name="lp_full_finetune_xrv_mixup",
               backbone="xrv", freezing=False, mixup=True,
               probe_epochs=10, epochs=20),
    Experiment(name="lp_full_finetune_xrv_mixup_lpnomix",
               backbone="xrv", freezing=False, mixup=True,
               probe_epochs=10, epochs=20, probe_mixup=False),
    Experiment(name="lp_full_finetune_xrv_mixup_lpnomix_40epoch",
               backbone="xrv", freezing=False, mixup=True,
               probe_epochs=10, epochs=30, probe_mixup=False),
    Experiment(
        name="lpft_progressive_xrv_mixup",
        backbone="xrv",
        freezing=True,
        probe_epochs=10,
        mixup=True,
        epochs=20,
    ),
    Experiment(
        name="progressive_freezing_xrv",
        backbone="xrv",
        freezing=True,
        mixup=False,
    ),
    Experiment(
        name="progressive_freezing_xrv_mixup",
        backbone="xrv",
        freezing=True,
        mixup=True,
    ),
    Experiment(name="full_finetune_xrv",
               backbone="xrv", freezing=False, mixup=False),

    Experiment(
        name="full_finetune_xrv_mixup",
        backbone="xrv",
        freezing=False,
        mixup=True,
        epochs=30,
    ),

    # ── ImageNet-RGB backbone (3-channel, TorchVision weights) ──
    Experiment(name="lpft_progressive_rgb",
               backbone="torchvision", freezing=True, mixup=False,
               probe_epochs=10, epochs=20),
    Experiment(name="progressive_freezing_rgb",
               backbone="torchvision", freezing=True, mixup=False),
    Experiment(name="full_finetune_rgb",
               backbone="torchvision", freezing=False, mixup=False),

    Experiment(
        name="full_finetune_rgb_mixup",
        backbone="torchvision",
        freezing=False,
        mixup=True,
        epochs=30,  # More epochs for mixup
    ),
    Experiment(
        name="progressive_freezing_rgb_mixup",
        backbone="torchvision",
        freezing=True,
        mixup=True,
        epochs=30,  # More epochs for mixup
    ),
]


def get_parser():
    """Create argument parser for ablation study."""
    parser = argparse.ArgumentParser(
        description="BPDneo ablation study - evaluate different model configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing images and labels"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="img",
        help="Subdirectory within data-dir containing preprocessed images"
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="labels.xlsx",
        help="Excel file with patient labels (within data-dir)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: data-dir/output/TIMESTAMP)"
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[exp.name for exp in EXPERIMENTS],
        default=None,
        help="Specific experiments to run (default: run all)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=6,
        help="Number of random runs per experiment"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Base random seed for reproducibility"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)"
    )
    
    return parser


def load_data(args):
    """Load and prepare the dataset."""
    # Load labels
    labels_path = args.data_dir / args.labels_file
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels_df = pd.read_excel(labels_path, sheet_name="Tabelle1", usecols=["patient_id", "bpd"])
    labels_df["label"] = labels_df["bpd"].map({
        "no BPD": 0, "mild": 0, "moderate": 1, "severe": 1, "keine Angabe": -1
    })
    labels_df = labels_df[labels_df['label'] != -1].dropna(subset=['label'])
    labels_df['label'] = labels_df['label'].astype(int)
    
    # Load images
    image_dir = args.data_dir / args.image_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    all_images = glob.glob(str(image_dir / "*.png"))
    if not all_images:
        raise ValueError(f"No PNG images found in {image_dir}")
    
    image_df = pd.DataFrame({'image_path': all_images})
    image_df['patient_id'] = image_df['image_path'].apply(lambda x: os.path.basename(x).split('_')[0])
    
    # Merge data
    master_df = image_df.merge(labels_df, on="patient_id", how="inner")
    
    print(f"Loaded {len(master_df)} images from {len(master_df.patient_id.nunique())} patients")
    print(f"Label distribution: {master_df.label.value_counts().to_dict()}")
    
    return master_df


def run_ablation_study(args, experiments: List[Experiment], master_df: pd.DataFrame, output_dir: Path):
    """Run the complete ablation study."""
    pos_ids = master_df.loc[master_df.label == 1, "patient_id"].unique()
    neg_ids = master_df.loc[master_df.label == 0, "patient_id"].unique()
    all_patient_ids = np.concatenate([pos_ids, neg_ids])
    
    print(f"Dataset: {len(pos_ids)} positive, {len(neg_ids)} negative patients")
    
    results: List[dict] = []
    for exp in experiments:
        # Override runs if specified
        exp.runs = args.runs
        
        for run in range(1, exp.runs + 1):
            print(f"\n=== {exp.name} — run {run}/{exp.runs} ===")
            for fold_idx, test_patient_ids in enumerate(
                    balanced_folds(pos_ids, neg_ids, k=args.folds, seed=args.seed + run)):
                train_patient_ids = np.setdiff1d(all_patient_ids, test_patient_ids)
                
                # Pass additional args to run_one_fold
                res = run_one_fold_with_args(
                    exp, run, fold_idx, master_df, 
                    train_patient_ids, test_patient_ids,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device
                )
                results.append(res)
                
                # Save intermediate results
                pd.DataFrame(results).to_csv(output_dir / "ablation_results.csv", index=False)
    
    print(f"\nFinal results saved to {output_dir / 'ablation_results.csv'}")
    return results


def run_one_fold_with_args(exp: Experiment, run_idx: int, fold_idx: int,
                           master_df: pd.DataFrame, train_patient_ids: np.ndarray,
                           test_patient_ids: np.ndarray, batch_size: int = 8,
                           num_workers: int = 12, device: Optional[str] = None) -> dict:
    """Wrapper for run_one_fold with additional arguments."""
    set_seed(1000 + run_idx + fold_idx)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Rest of the function remains the same as run_one_fold but with parameterized batch_size and num_workers
    probe_done = False

    # 1) backbone ----------------------------------------------------------
    if exp.backbone == "xrv":
        base = xrv.models.ResNet(weights="resnet50-res512-all")
    elif exp.backbone == "torchvision":
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError("backbone must be 'xrv' or 'torchvision'")

    # 2) model & optimiser -------------------------------------------------
    model = BPDModel(base).to(device)
    model.compile(freeze_backbone=exp.freezing, lr_backbone=2e-4, lr_head=1e-3)

    optimizer = model.get_optimizer()
    if exp.freezing:
        assert len(optimizer.param_groups) == 3, \
            "compile() must create separate groups for layer3, layer4 and head."
        optimizer.add_param_group({
            "params": model.inner_model.layer2.parameters(),
            "lr": 1e-4,
            "weight_decay": 1e-4,
        })

    # 3) data --------------------------------------------------------------
    p_pos = master_df[master_df.patient_id.isin(test_patient_ids) & (master_df.label == 1)].patient_id.nunique()
    p_neg = master_df[master_df.patient_id.isin(test_patient_ids) & (master_df.label == 0)].patient_id.nunique()

    if exp.backbone == "torchvision":
        ds_train = BPDDatasetRGB_5fold(master_df, train_patient_ids, stage="train",
                                       augmentor=get_augmentations(model.get_resolution()))
        ds_test = BPDDatasetRGB_5fold(master_df, test_patient_ids, stage="test")
    else:
        ds_train = BPDDataset_5fold(master_df, train_patient_ids, stage="train",
                                    augmentor=get_augmentations(model.get_resolution()))
        ds_test = BPDDataset_5fold(master_df, test_patient_ids, stage="test")

    print(
        f"  [Fold {fold_idx + 1}] Train: {len(ds_train)} images, Test: {len(ds_test)} images | Test patients: {p_pos} pos / {p_neg} neg")

    y = ds_train.data.label.to_numpy()
    class_w = 1.0 / np.bincount(y)
    sample_w = [class_w[i] for i in y]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    collate = None
    if exp.mixup:
        collate = lambda b: mixup_collate(b, alpha=exp.mixup_alpha, prob=exp.init_mixup_prob)

    dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, 
                         num_workers=num_workers, pin_memory=True, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, 
                        num_workers=min(4, num_workers), pin_memory=True)

    # Continue with rest of run_one_fold logic...
    # 4) scheduler (One‑Cycle, discriminative LR) -------------------------
    max_lr = [1e-3, 2.5e-4, 5e-4, 1e-4]
    if not exp.freezing:
        max_lr = max_lr[:2]

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=exp.epochs,
                                              steps_per_epoch=len(dl_train), pct_start=0.3, anneal_strategy="cos")
    crit = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    uf4, uf3, uf2 = math.floor(exp.unfreeze_layer4 * exp.epochs), math.floor(
        exp.unfreeze_layer3 * exp.epochs), math.floor(exp.unfreeze_layer2 * exp.epochs)
    best_acc = 0.0

    if exp.probe_epochs > 0:
        print(f"[run {run_idx}, fold {fold_idx + 1}] starting {exp.probe_epochs}-epoch linear probe")
        if not exp.probe_mixup:
            dl_train_probe = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, 
                                      num_workers=num_workers, pin_memory=True, collate_fn=None)
            run_linear_probe(model, dl_train_probe, dl_test, epochs=exp.probe_epochs, lr=exp.probe_lr, device=device)
        else:
            run_linear_probe(model, dl_train, dl_test, epochs=exp.probe_epochs, lr=exp.probe_lr, device=device)

    # 5) training loop -----------------------------------------------------
    for epoch in range(exp.epochs):
        if exp.freezing:
            if epoch == uf4: print(f"[epoch {epoch}] unfreezing layer4"); model.unfreeze_layers(
                model.inner_model.layer4)
            if epoch == uf3: print(f"[epoch {epoch}] unfreezing layer3"); model.unfreeze_layers(
                model.inner_model.layer3)
            if epoch == uf2: print(f"[epoch {epoch}] unfreezing layer2"); model.unfreeze_layers(
                model.inner_model.layer2)

        model.train()
        for batch in dl_train:
            if isinstance(batch[1], (list, tuple)):
                x, (ya, yb, lam) = batch
                x, ya, yb = x.to(device), ya.to(device), yb.to(device)
            else:
                x, ya = batch;
                x, ya = x.to(device), ya.to(device);
                yb, lam = None, 1.0

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(x)
                loss = crit(out, ya.unsqueeze(1)) if yb is None else lam * crit(out, ya.unsqueeze(1)) + (
                        1 - lam) * crit(out, yb.unsqueeze(1))
            scaler.scale(loss).backward()
            scaler.step(optimizer);
            scaler.update();
            scheduler.step()

        model.eval();
        correct_te = total_te = 0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(device), y.to(device)
                preds = (model(x) > 0).cpu().numpy().flatten()
                correct_te += (preds == y.cpu().numpy()).sum();
                total_te += len(y)
        test_acc = correct_te / total_te
        best_acc = max(best_acc, test_acc)
        print(
            f"{exp.name} | run {run_idx} | fold {fold_idx + 1} | epoch {epoch + 1}/{exp.epochs} | test acc {test_acc:.3f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        for x, y in dl_test:
            x, y = x.to(device), y.to(device)
            preds = model(x).squeeze()
            all_preds.append(preds.cpu())
            all_y.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_y = torch.cat(all_y).int()

    # Initialize metric calculators
    auroc = BinaryAUROC()
    f1 = BinaryF1Score()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    specificity = BinarySpecificity()
    conf_matrix = BinaryConfusionMatrix()

    # Calculate all metrics
    final_auroc = auroc(all_preds, all_y).item()
    final_f1 = f1(all_preds, all_y).item()
    final_precision = precision(all_preds, all_y).item()
    final_recall = recall(all_preds, all_y).item()
    final_specificity = specificity(all_preds, all_y).item()

    binary_preds = (all_preds > 0).int()
    test_acc = (binary_preds == all_y).float().mean().item()

    cm = conf_matrix(all_preds, all_y).numpy()
    tn, fp, fn, tp = cm.ravel()

    return {
        "experiment": exp.name, "run": run_idx, "fold": fold_idx + 1,
        "test_accuracy": test_acc,
        "auroc": final_auroc,
        "f1_score": final_f1,
        "sensitivity": final_recall,
        "specificity": final_specificity,
        "precision": final_precision,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    # Load data
    master_df = load_data(args)
    
    # Select experiments
    if args.experiments:
        experiments = [exp for exp in EXPERIMENTS if exp.name in args.experiments]
        if not experiments:
            raise ValueError(f"No valid experiments selected from: {args.experiments}")
    else:
        experiments = EXPERIMENTS
    
    print(f"Running {len(experiments)} experiment(s): {[exp.name for exp in experiments]}")
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = args.data_dir / "output" / time.strftime("%Y%m%d-%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Run study
    run_ablation_study(args, experiments, master_df, args.output_dir)


if __name__ == "__main__":
    main()

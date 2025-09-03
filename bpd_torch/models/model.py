# model.py  – revised
# unified model wrapper with configurable compile()

from __future__ import annotations

import torch
from torch import nn, optim
from torch.nn import init
from typing import List


class BPDModel(nn.Module):
    """Wrap either a TorchXRayVision ``ResNet`` or a torchvision ``ResNet``.

    * Detects backend at runtime and exposes the raw torchvision model at
      ``self.inner_model`` so later code can address layers uniformly.
    * ``compile()`` builds an optimiser and optionally freezes the backbone.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

        # ── pick the true torchvision model regardless of wrapper ─────────
        if hasattr(self.base_model, "model"):
            self.inner_model = self.base_model.model  # XRV wrapper
            self.is_xrv = True
        else:
            self.inner_model = self.base_model        # torchvision model
            self.is_xrv = False

        # ── replace FC with identity – we'll add our own head ─────────────
        in_features = self.inner_model.fc.in_features
        self.inner_model.fc = nn.Identity()
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(in_features, 1)
        self.fc.apply(init_weights)

        self._optimizer: optim.Optimizer | None = None

    # ------------------------------------------------------------------
    # forward / utils
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.is_xrv:
            feats = self.base_model.features(x)
        else:
            feats = self.inner_model(x)
        feats = self.dropout(feats)
        return self.fc(feats)

    def get_resolution(self) -> int:
        """Return default training resolution (used by augmentor)."""
        return 512

    # ------------------------------------------------------------------
    # Optimiser / freezing logic – CONFIGURABLE
    # ------------------------------------------------------------------
    def compile(
        self,
        *,
        freeze_backbone: bool = True,
        lr_backbone: float = 1e-4,
        lr_head: float = 5e-4,
        wd_backbone: float = 1e-4,
        wd_head: float = 1e-5,
    ) -> None:
        """Build an AdamW optimiser and (optionally) freeze the backbone.

        *When ``freeze_backbone`` is ``True`` the backbone starts frozen –
        only the head learns.  Layers **remain present** in the optimiser so
        that they can be unfrozen later without touching the optimiser / LR
        scheduler.*
        """
        # ── freeze / unfreeze tensors ------------------------------------
        for p in self.inner_model.parameters():
            p.requires_grad = not freeze_backbone

        # ── build parameter groups ---------------------------------------
        param_groups: List[dict] = [
            {"params": self.fc.parameters(), "lr": lr_head, "weight_decay": wd_head},
        ]

        if freeze_backbone:
            # keep layer3/4 frozen for now but add them so scheduler knows them
            for blk in (self.inner_model.layer3, self.inner_model.layer4):
                for p in blk.parameters():
                    p.requires_grad = False
            param_groups += [
                {"params": self.inner_model.layer3.parameters(), "lr": lr_backbone, "weight_decay": wd_backbone},
                {"params": self.inner_model.layer4.parameters(), "lr": lr_backbone, "weight_decay": wd_backbone},
            ]
        else:
            # full‑finetune: one group for the entire backbone
            param_groups.append({
                "params": self.inner_model.parameters(),
                "lr": lr_backbone,
                "weight_decay": wd_backbone,
            })

        self._optimizer = optim.AdamW(param_groups)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def load_model(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=torch.device("cpu"))
        self.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and self._optimizer is not None:
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def get_optimizer(self) -> optim.Optimizer:
        if self._optimizer is None:
            raise RuntimeError("compile() must be called before get_optimizer()")
        return self._optimizer

    # Optionally expose an easy unfreeze utility -------------------------
    def unfreeze_layers(self, *layers: nn.Sequential) -> None:
        """Set ``requires_grad = True`` for all params in *layers*."""
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = True


# ----------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

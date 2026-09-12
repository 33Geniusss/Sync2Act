from __future__ import annotations

from torch import nn

from .act_lite import ACTLite
from .bc_mlp import BCMLP
from .quality_act import QualityAwareACT


def build_policy(config: dict) -> nn.Module:
    name = config.get("name", "bc_mlp")
    common = {"state_dim": config.get("state_dim", 6), "action_dim": config.get("action_dim", 3)}
    if name == "bc_mlp":
        return BCMLP(
            **common,
            input_mode=config.get("input_mode", "image_state"),
            hidden_dim=config.get("hidden_dim", 128),
        )
    temporal = {
        **common,
        "horizon": config.get("horizon", 16),
        "hidden_dim": config.get("hidden_dim", 96),
        "num_layers": config.get("num_layers", 2),
        "num_heads": config.get("num_heads", 4),
        "dropout": config.get("dropout", 0.0),
    }
    if name in {"act_lite", "quality_act_weighted", "quality_weighted_loss"}:
        return ACTLite(**temporal)
    if name in {
        "quality_act",
        "quality_act_input",
        "quality_act_full",
        "quality_act_shuffled",
        "quality_act_constant",
        "quality_input",
        "quality_full",
        "quality_shuffled",
        "quality_constant",
    }:
        return QualityAwareACT(
            **temporal, use_quality_features=config.get("use_quality_features", True)
        )
    raise ValueError(f"Unknown policy: {name}")

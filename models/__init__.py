"""Model components for Logical-JEPA.

Every module here is implemented from scratch in PyTorch and initialised
randomly: no ImageNet backbone, no pretrained I-JEPA / DINO / CLIP weights.
"""

from .context_encoder import ContextEncoder, build_context_encoder
from .logical_jepa import LogicalJEPA, build_model
from .patch_embed import (
    PatchEmbed,
    build_2d_sincos_pos_embed,
    gather_pos_embed,
    gather_tokens,
)
from .predictor import JEPAPredictor, build_predictor
from .target_encoder import TargetEncoder, normalize_targets
from .transformer import (
    DropPath,
    Mlp,
    MultiHeadSelfAttention,
    TransformerBlock,
    init_vit_weights,
    trunc_normal_,
)

__all__ = [
    "PatchEmbed",
    "build_2d_sincos_pos_embed",
    "gather_tokens",
    "gather_pos_embed",
    "MultiHeadSelfAttention",
    "Mlp",
    "TransformerBlock",
    "DropPath",
    "init_vit_weights",
    "trunc_normal_",
    "ContextEncoder",
    "build_context_encoder",
    "TargetEncoder",
    "normalize_targets",
    "JEPAPredictor",
    "build_predictor",
    "LogicalJEPA",
    "build_model",
]

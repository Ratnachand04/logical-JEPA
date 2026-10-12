"""Tests for pretrained encoder initialisation.

The guarantees that matter, and why each is tested:

* **The transfer is exact.** :func:`test_matches_an_independent_dino_forward`
  compares our encoder, loaded with DINO weights, against a ViT forward
  written directly from the raw checkpoint tensors. If the CLS handling, the
  position table or the input re-normalisation were subtly wrong, features
  would still *look* plausible and every downstream number would quietly
  degrade. It needs the downloaded checkpoint and skips otherwise.
* **No partial loads.** A "pretrained" model with a few random layers left
  inside would raise no error at all; the loader refuses instead.
* **Rebuilding a checkpoint never downloads.** Evaluation and the demo must
  work offline from the saved state dict alone.
* **The from-scratch default is untouched**, so every earlier result still
  reproduces from its old checkpoints.

Everything except the parity test runs on synthetic state dicts, no network.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from masking import build_mask_generator  # noqa: E402
from models import build_model  # noqa: E402
from models import pretrained as P  # noqa: E402
from models.context_encoder import build_context_encoder  # noqa: E402
from utils.config import Config  # noqa: E402

# A tiny stand-in spec, so mapping logic is tested in milliseconds.
TINY = P.PretrainedSpec(
    name="tiny_test", url="http://invalid.local/tiny.pth",
    embed_dim=32, depth=2, num_heads=4, patch_size=16, source_grid=14,
)


@pytest.fixture(autouse=True)
def _register_tiny(monkeypatch):
    monkeypatch.setitem(P.PRETRAINED, "tiny_test", TINY)


def _fake_state(spec=TINY, seed=0):
    """A state dict with DINO's exact key layout and random values."""
    g = torch.Generator().manual_seed(seed)
    d, p = spec.embed_dim, spec.patch_size
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    sd = {
        "cls_token": r(1, 1, d),
        "pos_embed": r(1, spec.source_grid**2 + 1, d),
        "patch_embed.proj.weight": r(d, 3, p, p),
        "patch_embed.proj.bias": r(d),
        "norm.weight": r(d), "norm.bias": r(d),
    }
    for i in range(spec.depth):
        b = f"blocks.{i}."
        sd.update({
            b + "norm1.weight": r(d), b + "norm1.bias": r(d),
            b + "attn.qkv.weight": r(3 * d, d), b + "attn.qkv.bias": r(3 * d),
            b + "attn.proj.weight": r(d, d), b + "attn.proj.bias": r(d),
            b + "norm2.weight": r(d), b + "norm2.bias": r(d),
            b + "mlp.fc1.weight": r(4 * d, d), b + "mlp.fc1.bias": r(4 * d),
            b + "mlp.fc2.weight": r(d, 4 * d), b + "mlp.fc2.bias": r(d),
        })
    return sd


def _tiny_encoder(img_size=256):
    return build_context_encoder({"pretrained": "tiny_test", "img_size": img_size})


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_unknown_pretrained_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown pretrained encoder"):
        P.get_spec("resnet_from_nowhere")


@pytest.mark.parametrize("name", ["dino_vits16", "dino_vitb16"])
def test_registry_specs_are_self_consistent(name):
    spec = P.get_spec(name)
    assert spec.embed_dim % spec.num_heads == 0
    assert spec.url.startswith("https://")
    assert spec.size_bytes and spec.size_bytes > 1_000_000
    assert len(spec.mean) == len(spec.std) == 3


def test_the_default_backbone_has_a_pinned_checksum():
    """A truncated or substituted download must be caught, not loaded."""
    assert P.get_spec("dino_vits16").sha256 is not None


# --------------------------------------------------------------------------- #
# Position interpolation
# --------------------------------------------------------------------------- #
def test_interpolation_is_identity_at_the_source_grid():
    pos = torch.randn(1, 14 * 14, 8)
    assert torch.equal(P.interpolate_pos_embed(pos, 14, 14), pos)


def test_interpolation_resamples_to_the_target_grid():
    pos = torch.randn(1, 14 * 14, 8)
    assert P.interpolate_pos_embed(pos, 14, 16).shape == (1, 256, 8)


def test_interpolation_preserves_a_smooth_field():
    """A linear ramp across the grid must stay a monotone ramp after resampling."""
    ramp = torch.linspace(0, 1, 14).repeat(14, 1).reshape(1, 196, 1)
    out = P.interpolate_pos_embed(ramp, 14, 16).reshape(16, 16)
    assert (out[:, 1:] >= out[:, :-1] - 1e-4).all()
    assert out[0, 0] < 0.05 and out[0, -1] > 0.95


# --------------------------------------------------------------------------- #
# Weight transfer
# --------------------------------------------------------------------------- #
def test_every_encoder_weight_is_copied_exactly():
    enc, sd = _tiny_encoder(), _fake_state()
    report = P.load_pretrained_into(enc, "tiny_test", sd)

    own = dict(enc.named_parameters())
    for key, value in sd.items():
        if key in ("cls_token", "pos_embed"):
            continue
        assert torch.equal(own[key], value), f"{key} not copied"
    assert report["tensors_loaded"] == len(own)


def test_cls_token_carries_its_own_position():
    enc, sd = _tiny_encoder(), _fake_state()
    P.load_pretrained_into(enc, "tiny_test", sd)
    assert torch.allclose(enc.cls_embed, sd["cls_token"] + sd["pos_embed"][:, :1])


def test_patch_positions_are_resampled_onto_our_grid():
    enc, sd = _tiny_encoder(img_size=256), _fake_state()
    P.load_pretrained_into(enc, "tiny_test", sd)
    expected = P.interpolate_pos_embed(sd["pos_embed"][:, 1:], 14, 16)
    assert torch.allclose(enc.pos_embed, expected)


def test_a_partial_checkpoint_is_refused():
    """Random layers left inside a 'pretrained' model would raise nothing."""
    sd = _fake_state()
    del sd["blocks.1.mlp.fc2.weight"]
    with pytest.raises(RuntimeError, match="uninitialised"):
        P.load_pretrained_into(_tiny_encoder(), "tiny_test", sd)


def test_a_shape_mismatch_is_refused():
    sd = _fake_state()
    sd["blocks.0.attn.proj.weight"] = torch.randn(8, 8)
    with pytest.raises(ValueError, match="shape mismatch"):
        P.load_pretrained_into(_tiny_encoder(), "tiny_test", sd)


def test_the_wrong_architecture_is_refused():
    wrong = build_context_encoder({"embed_dim": 32, "depth": 3, "num_heads": 4})
    with pytest.raises(ValueError, match="depth"):
        P.load_pretrained_into(wrong, "tiny_test", _fake_state())


def test_input_renormalisation_maps_our_pipeline_onto_imagenet_statistics():
    """Data is normalised to 0.5/0.5; the encoder must see ImageNet stats."""
    enc = _tiny_encoder()
    pixels = torch.rand(1, 3, 32, 32)
    ours = (pixels - 0.5) / 0.5
    mean = torch.tensor(TINY.mean).view(1, 3, 1, 1)
    std = torch.tensor(TINY.std).view(1, 3, 1, 1)
    assert torch.allclose(ours * enc.in_scale + enc.in_shift, (pixels - mean) / std, atol=1e-6)


# --------------------------------------------------------------------------- #
# Finetune modes
# --------------------------------------------------------------------------- #
def test_frozen_mode_trains_nothing_in_the_encoder():
    assert P.apply_finetune_mode(_tiny_encoder(), "frozen") == 0


def test_last_k_trains_only_the_last_blocks_and_final_norm():
    enc = _tiny_encoder()
    P.apply_finetune_mode(enc, "last_k", last_k=1)
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith(("blocks.1.", "norm.")) for n in trainable)


def test_full_mode_trains_everything():
    enc = _tiny_encoder()
    assert P.apply_finetune_mode(enc, "full") == sum(p.numel() for p in enc.parameters())


def test_invalid_finetune_settings_are_rejected():
    with pytest.raises(ValueError, match="Unknown finetune mode"):
        P.apply_finetune_mode(_tiny_encoder(), "half")
    with pytest.raises(ValueError, match="finetune_last_k"):
        P.apply_finetune_mode(_tiny_encoder(), "last_k", last_k=99)


# --------------------------------------------------------------------------- #
# The full model
# --------------------------------------------------------------------------- #
def _cfg(**encoder):
    return Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"pretrained": "tiny_test", **encoder},
        "predictor": {"predictor_dim": 32, "depth": 1, "num_heads": 4},
        "loss": {"kind": "cosine"},
    })


def _model_with_fake_weights(monkeypatch, **encoder):
    monkeypatch.setattr(P, "fetch_state_dict", lambda name, progress=True: _fake_state())
    import models.logical_jepa as lj
    monkeypatch.setattr(lj, "load_pretrained_into",
                        lambda enc, name: P.load_pretrained_into(enc, name, _fake_state()))
    return build_model(_cfg(**encoder))


def test_the_teacher_starts_from_the_pretrained_weights(monkeypatch):
    model = _model_with_fake_weights(monkeypatch)
    for s, t in zip(model.context_encoder.parameters(),
                    model.target_encoder.encoder.parameters()):
        assert torch.equal(s, t)
    assert torch.equal(model.target_encoder.encoder.pos_embed, model.context_encoder.pos_embed)


def test_frozen_model_trains_only_the_predictor(monkeypatch):
    model = _model_with_fake_weights(monkeypatch, finetune="frozen")
    trainable = {id(p) for p in model.trainable_parameters()}
    assert trainable == {id(p) for p in model.predictor.parameters()}


def test_a_training_step_leaves_a_frozen_encoder_unchanged(monkeypatch):
    model = _model_with_fake_weights(monkeypatch, finetune="frozen")
    before = [p.clone() for p in model.context_encoder.parameters()]
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)

    gen = build_mask_generator({"strategy": "multiscale"}, model.grid_size, seed=0)
    loss, _ = model(torch.randn(2, 3, 256, 256), gen())
    loss.backward()
    opt.step()

    for old, new in zip(before, model.context_encoder.parameters()):
        assert torch.equal(old, new)
    assert any(p.grad is not None for p in model.predictor.parameters())


def test_rebuilding_from_a_checkpoint_never_downloads(monkeypatch):
    """Evaluation and the demo must work offline from the state dict alone."""
    model = _model_with_fake_weights(monkeypatch)
    state = model.state_dict()

    def _no_network(*args, **kwargs):
        raise AssertionError("a checkpoint rebuild tried to fetch pretrained weights")

    monkeypatch.setattr(P, "fetch_state_dict", _no_network)
    import models.logical_jepa as lj
    monkeypatch.setattr(lj, "load_pretrained_into", _no_network)

    rebuilt = build_model(_cfg(), load_pretrained=False)
    rebuilt.load_state_dict(state)

    x = torch.randn(1, 3, 256, 256)
    model.eval(), rebuilt.eval()
    with torch.no_grad():
        assert torch.allclose(model.encode_targets(x), rebuilt.encode_targets(x), atol=1e-6)


def test_pretrained_positions_and_cls_are_saved_in_checkpoints(monkeypatch):
    keys = _model_with_fake_weights(monkeypatch).state_dict().keys()
    assert "context_encoder.pos_embed" in keys
    assert "context_encoder.cls_embed" in keys


def test_from_scratch_checkpoints_keep_their_old_layout():
    """Old checkpoints load into new code: no new keys in the default model."""
    model = build_model(Config({
        "data": {"img_size": 256, "patch_size": 16},
        "encoder": {"embed_dim": 64, "depth": 2, "num_heads": 4},
        "predictor": {"predictor_dim": 64, "depth": 1, "num_heads": 4},
    }))
    keys = model.state_dict().keys()
    assert "context_encoder.pos_embed" not in keys
    assert not any("cls_embed" in k for k in keys)
    assert model.context_encoder.cls_embed is None
    assert model.pretrained is None and model.finetune == "full"


def test_the_demo_reports_what_was_actually_trained(monkeypatch):
    """A frozen encoder must not be advertised as trained, nor as from scratch."""
    import importlib.util

    from anomaly.scoring import build_scorer
    from evaluate import build_sweep_bank

    spec = importlib.util.spec_from_file_location(
        "demo_server", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "app", "server.py"))
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    model = _model_with_fake_weights(monkeypatch, finetune="frozen").eval()
    cfg = _cfg(finetune="frozen")
    cfg.set_path("data.num_train_images", 16)
    cfg.set_path("anomaly.sweep_windows", [4])
    bank = build_sweep_bank(cfg, model, torch.device("cpu"))
    monkeypatch.setitem(server.STATE, "model", model)
    monkeypatch.setitem(server.STATE, "cfg", cfg)
    monkeypatch.setitem(server.STATE, "scorer", build_scorer(cfg, model, bank))
    monkeypatch.setitem(server.STATE, "device", torch.device("cpu"))

    info = server.app.test_client().get("/api/status").get_json()["model"]
    assert info["pretrained"] == "tiny_test"
    assert info["finetune"] == "frozen"
    assert info["train_images"] == 16
    assert info["trainable_params"] == sum(p.numel() for p in model.predictor.parameters())


# --------------------------------------------------------------------------- #
# Parity with real DINO (needs the downloaded checkpoint)
# --------------------------------------------------------------------------- #
def _reference_vit(x, sd, heads):
    """Independent ViT forward from raw tensors -- shares no project code."""
    x = F.conv2d(x, sd["patch_embed.proj.weight"], sd["patch_embed.proj.bias"], stride=16)
    x = x.flatten(2).transpose(1, 2)
    x = torch.cat([sd["cls_token"].expand(x.size(0), -1, -1), x], 1) + sd["pos_embed"]
    B, N, D = x.shape
    hd = D // heads
    i = 0
    while f"blocks.{i}.norm1.weight" in sd:
        p = f"blocks.{i}."
        h = F.layer_norm(x, (D,), sd[p + "norm1.weight"], sd[p + "norm1.bias"], 1e-6)
        qkv = F.linear(h, sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"])
        qkv = qkv.reshape(B, N, 3, heads, hd).permute(2, 0, 3, 1, 4)
        a = (qkv[0] @ qkv[1].transpose(-2, -1) * hd**-0.5).softmax(-1) @ qkv[2]
        x = x + F.linear(a.transpose(1, 2).reshape(B, N, D),
                         sd[p + "attn.proj.weight"], sd[p + "attn.proj.bias"])
        h = F.layer_norm(x, (D,), sd[p + "norm2.weight"], sd[p + "norm2.bias"], 1e-6)
        h = F.gelu(F.linear(h, sd[p + "mlp.fc1.weight"], sd[p + "mlp.fc1.bias"]))
        x = x + F.linear(h, sd[p + "mlp.fc2.weight"], sd[p + "mlp.fc2.bias"])
        i += 1
    return F.layer_norm(x, (D,), sd["norm.weight"], sd["norm.bias"], 1e-6)[:, 1:]


@pytest.mark.skipif(not os.path.isfile(P.checkpoint_path("dino_vits16")),
                    reason="DINO ViT-S/16 checkpoint not downloaded")
def test_matches_an_independent_dino_forward():
    """Our encoder with DINO weights must compute exactly what DINO computes.

    Run at 224 px (DINO's own 14x14 grid) so no interpolation is involved, and
    fed through our 0.5/0.5 pipeline while the reference gets ImageNet
    statistics -- so the input conversion is checked too.
    """
    sd = P.fetch_state_dict("dino_vits16", progress=False)
    enc = build_context_encoder({"pretrained": "dino_vits16", "img_size": 224}).eval()
    P.load_pretrained_into(enc, "dino_vits16", sd)

    torch.manual_seed(0)
    pixels = torch.rand(2, 3, 224, 224)
    mean = torch.tensor(P.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(P.IMAGENET_STD).view(1, 3, 1, 1)

    with torch.no_grad():
        ours = enc((pixels - 0.5) / 0.5)
        ref = _reference_vit((pixels - mean) / std, sd, heads=6)

    assert ours.shape == ref.shape == (2, 196, 384)
    assert (ours - ref).abs().max() < 1e-3
    assert F.cosine_similarity(ours, ref, dim=-1).min() > 0.99999

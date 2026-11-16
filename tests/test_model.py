"""Checks that double as teaching points:  python -m pytest tests -q"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidetalk.model import GatedCrossAttention, SlideTalker  # noqa: E402


def test_shapes_follow_the_text_not_the_slide():
    xa = GatedCrossAttention(768, 12)
    x, mem = torch.randn(2, 7, 768), torch.randn(2, 300, 768)
    out = xa(x, mem, torch.ones(2, 300, dtype=torch.bool))
    assert out.shape == x.shape                       # one output row per word, however many tiles
    assert xa.last_attn.shape == (2, 12, 7, 300)      # every word, every head, a weight for every tile
    assert torch.allclose(xa.last_attn.sum(-1), torch.ones(2, 12, 7), atol=1e-5)


def test_closed_gate_changes_nothing():
    xa = GatedCrossAttention(768, 12)
    x = torch.randn(1, 5, 768)
    assert torch.equal(xa(x, torch.randn(1, 50, 768), torch.ones(1, 50, dtype=torch.bool)), x)


def test_padding_tiles_get_no_attention():
    xa = GatedCrossAttention(768, 12)
    mask = torch.ones(1, 40, dtype=torch.bool)
    mask[0, 30:] = False
    xa(torch.randn(1, 4, 768), torch.randn(1, 40, 768), mask)
    assert xa.last_attn[..., 30:].abs().max() == 0


@pytest.fixture(scope="module")
def model():
    return SlideTalker().eval()


def test_starts_as_plain_gpt2(model):
    ids = model.tok("Question: What is this?\nAnswer:", return_tensors="pt").input_ids
    model.set_slide(None)
    plain = model.lm(input_ids=ids).logits
    model.set_slide(torch.randn(1, 64, 768))
    with_slide = model.lm(input_ids=ids).logits
    assert torch.allclose(plain, with_slide, atol=1e-5)  # gates are shut at initialisation


def test_open_gate_makes_answers_depend_on_slide(model):
    ids = model.tok("Question: What is this?\nAnswer:", return_tensors="pt").input_ids
    with torch.no_grad():
        for b in model.xattn:
            b.gate.fill_(1.0)
        model.set_slide(torch.randn(1, 64, 768))
        a = model.lm(input_ids=ids).logits
        model.set_slide(torch.randn(1, 64, 768))
        b = model.lm(input_ids=ids).logits
        for blk in model.xattn:
            blk.gate.zero_()
    assert not torch.allclose(a, b)

"""SlideTalk model: a frozen GPT-2 that can look at a slide through gated cross-attention.

    tile features (N, 768) ──► projection ──► memory (N, 768) ─────────────┐ keys, values
                                                                            ▼
    "Question: …\\nAnswer:" ──► GPT-2 block ×3 ──► gated cross-attention ──► GPT-2 block ×3 ──► … ──► answer
                                                    queries come from the text

Trained : projection (0.6M) + 4 gated cross-attention blocks (9.4M)
Frozen  : GPT-2 (124M) and the Phikon tile encoder (already applied, features are on disk)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

XATTN_AFTER = (2, 5, 8, 11)  # GPT-2 block indices (0-based): cross-attention after blocks 3, 6, 9, 12


class GatedCrossAttention(nn.Module):
    """Text tokens ask, tile tokens answer.

    x    (B, T, d)  text hidden states, one row per word piece         -> queries
    mem  (B, N, d)  slide memory, one row per tile                      -> keys and values
    mask (B, N)     True for real tiles, False for padding
    returns (B, T, d): x plus a gated, weighted average of tile values
    """

    def __init__(self, d: int = 768, heads: int = 12):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.ln = nn.LayerNorm(d)
        self.W_q = nn.Linear(d, d)  # from text:  what am I looking for?
        self.W_k = nn.Linear(d, d)  # from tiles: what do I contain?
        self.W_v = nn.Linear(d, d)  # from tiles: what do I hand over if chosen?
        self.W_o = nn.Linear(d, d)  # merge the heads
        # tanh(0) = 0: at the start the block adds nothing, so GPT-2 behaves exactly as released.
        # Training opens the gate once looking at the slide starts to lower the loss (Flamingo, 2022).
        self.gate = nn.Parameter(torch.zeros(1))
        self.last_attn: torch.Tensor | None = None  # (B, heads, T, N), kept for attention maps

    def forward(self, x, mem, mask):
        B, T, _ = x.shape
        N = mem.shape[1]
        q = self.W_q(self.ln(x)).view(B, T, self.h, self.dh).transpose(1, 2)  # (B, h, T, dh)
        k = self.W_k(mem).view(B, N, self.h, self.dh).transpose(1, 2)        # (B, h, N, dh)
        v = self.W_v(mem).view(B, N, self.h, self.dh).transpose(1, 2)        # (B, h, N, dh)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                # (B, h, T, N): word vs tile
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)                                         # each word's weights sum to 1
        self.last_attn = attn.detach()
        read = (attn @ v).transpose(1, 2).reshape(B, T, -1)                   # (B, T, d): what each word read
        return x + torch.tanh(self.gate) * self.W_o(read)


class SlideTalker(nn.Module):
    def __init__(self, lm_name: str = "gpt2", tile_dim: int = 768, mem_dropout: float = 0.0):
        super().__init__()
        self.tok = GPT2TokenizerFast.from_pretrained(lm_name)
        self.tok.pad_token = self.tok.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
        self.lm.requires_grad_(False)
        d, heads = self.lm.config.n_embd, self.lm.config.n_head
        self.proj = nn.Sequential(nn.LayerNorm(tile_dim), nn.Linear(tile_dim, d))  # tile space -> word space
        # Training only: randomly zero some numbers of the slide memory so no single feature is relied on.
        self.mem_drop = nn.Dropout(mem_dropout)
        self.xattn = nn.ModuleList([GatedCrossAttention(d, heads) for _ in XATTN_AFTER])
        self._memory = None  # (memory, mask) of the slide currently being discussed
        for j, i in enumerate(XATTN_AFTER):
            self.lm.transformer.h[i].register_forward_hook(self._hook(j))
        self.train(False)

    # GPT-2's own blocks run unchanged; a forward hook passes each chosen block's output through ours.
    def _hook(self, j):
        def run(_block, _inputs, out):
            if self._memory is None:  # no slide set: plain GPT-2
                return None
            h = out[0] if isinstance(out, tuple) else out
            h = self.xattn[j](h, *self._memory)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return run

    def train(self, mode: bool = True):
        super().train(mode)
        self.lm.eval()  # GPT-2 is frozen: keep its dropout off
        return self

    @property
    def device(self):
        return self.proj[1].weight.device

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("lm.")}

    def load_trainable(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("lm.") for k in missing), (missing, unexpected)

    # ------------------------------------------------------------------ slide memory
    def set_slide(self, feats: torch.Tensor | None, mask: torch.Tensor | None = None):
        """feats (B, N, 768) tile features. None means 'no slide' (plain GPT-2)."""
        if feats is None:
            self._memory = None
            return
        if mask is None:
            mask = torch.ones(feats.shape[:2], dtype=torch.bool, device=feats.device)
        self._memory = (self.mem_drop(self.proj(feats)), mask)

    @staticmethod
    def prompt(question: str) -> str:
        return f"Question: {question}\nAnswer:"

    # ------------------------------------------------------------------ training
    def loss(self, feats, mask, questions, answers):
        """Next-word loss on the answer words only (the question words are not graded)."""
        self.set_slide(feats, mask)
        ids, labels = [], []
        for q, a in zip(questions, answers):
            p = self.tok(self.prompt(q)).input_ids
            t = self.tok(" " + a).input_ids + [self.tok.eos_token_id]
            ids.append(p + t)
            labels.append([-100] * len(p) + t)
        L = max(map(len, ids))
        pad = self.tok.eos_token_id
        input_ids = torch.tensor([s + [pad] * (L - len(s)) for s in ids], device=self.device)
        target = torch.tensor([s + [-100] * (L - len(s)) for s in labels], device=self.device)
        attention = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in ids], device=self.device)
        return self.lm(input_ids=input_ids, attention_mask=attention, labels=target).loss

    # ------------------------------------------------------------------ talking
    @torch.no_grad()
    def answer(self, feats: torch.Tensor | None, question: str, max_new: int = 24) -> str:
        """Greedy decoding about one slide. feats: (N, 768) tile features, or None for no slide."""
        self.set_slide(None if feats is None else feats[None])
        ids = self.tok(self.prompt(question), return_tensors="pt").input_ids.to(self.device)
        start, past = ids.shape[1], None
        step = ids
        for _ in range(max_new):
            out = self.lm(input_ids=step, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().view(1, 1)
            if nxt.item() == self.tok.eos_token_id:
                break
            ids, step = torch.cat([ids, nxt], 1), nxt
        return self.tok.decode(ids[0, start:]).strip()

    @torch.no_grad()
    def attention_map(self, feats: torch.Tensor, question: str) -> np.ndarray:
        """How much the first answer word attended to each tile: (N,), averaged over heads and blocks."""
        self.set_slide(feats[None])
        ids = self.tok(self.prompt(question), return_tensors="pt").input_ids.to(self.device)
        self.lm(input_ids=ids)
        att = torch.stack([b.last_attn[0, :, -1] for b in self.xattn])  # (blocks, heads, N)
        return att.mean((0, 1)).float().cpu().numpy()

    def gates(self) -> list[float]:
        return [torch.tanh(b.gate).item() for b in self.xattn]

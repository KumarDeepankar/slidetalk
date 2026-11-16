"""Scoring a SlideTalker by asking it questions and checking the generated answer word for word."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch

from .data import ASK_ORGAN, ASK_SUBTYPE, SUBTYPE_NAMES, SUBTYPES, fixed_sample, read_slide

EVAL_TILES = 512  # same number of tiles the model sees in training

# question kind -> (question for a row, correct answer for a row)
PROBES = {
    "subtype": (lambda r: ASK_SUBTYPE[0], lambda r: r.subtype),
    "organ": (lambda r: ASK_ORGAN[0], lambda r: r.organ),
    "yes/no (true)": (lambda r: f"Is this a {r.subtype}?", lambda r: "yes"),
    "yes/no (false)": (lambda r: f"Is this a {SUBTYPE_NAMES[(SUBTYPE_NAMES.index(r.subtype) + 2) % 5]}?",
                       lambda r: "no"),
    "choice within organ": (
        lambda r: f"This is a {r.organ} tumor. Is it "
                  + " or ".join(f"a {s}" for o, s in SUBTYPES.values() if o == r.organ) + "?",
        lambda r: r.subtype),
}


def eval_tiles(file_id: str, device, n: int = EVAL_TILES, draw: int = 0) -> torch.Tensor:
    feats, _ = read_slide(file_id)
    return torch.from_numpy(feats[fixed_sample(file_id, n, draw)].astype(np.float32)).to(device)


def ask_all(model, df: pd.DataFrame, kinds=tuple(PROBES), tiles=None, votes: int = 1,
            n_tiles: int = EVAL_TILES) -> pd.DataFrame:
    """One row per (slide, question kind): question, truth, answer, correct.

    tiles: optional function row_index -> tile tensor, to show the model a different slide (controls).
    votes: ask with this many different tile samples of the slide and keep the most common answer.
    """
    out = []
    for i, r in df.iterrows():
        views = ([tiles(i)] if tiles is not None else
                 [eval_tiles(r.file_id, model.device, n_tiles, d) for d in range(votes)])
        for k in kinds:
            q, truth = PROBES[k][0](r), PROBES[k][1](r)
            a = Counter(model.answer(x, q) for x in views).most_common(1)[0][0]
            out.append(dict(file_id=r.file_id, subtype=r.subtype, kind=k, question=q, truth=truth, answer=a,
                            correct=a.strip().lower() == truth))
    return pd.DataFrame(out)


@torch.no_grad()
def val_loss(model, df: pd.DataFrame) -> float:
    """Average next-word loss on every probe question of every slide: a smoother signal than accuracy
    on a few dozen slides, so it is what we use to pick the best checkpoint."""
    losses = []
    for _, r in df.iterrows():
        x = eval_tiles(r.file_id, model.device)
        qs = [PROBES[k][0](r) for k in PROBES]
        ans = [PROBES[k][1](r) for k in PROBES]
        feats = x[None].expand(len(qs), -1, -1)
        mask = torch.ones(feats.shape[:2], dtype=torch.bool, device=model.device)
        losses.append(model.loss(feats, mask, qs, ans).item())
    return float(np.mean(losses))

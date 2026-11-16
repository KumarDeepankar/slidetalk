"""Data for SlideTalk.

Inputs  : one .h5 file per slide in data/features/phikon/ (made by scripts/extract_features.py)
            features (N, 768)  Phikon embedding of every tissue tile
            grid     (N, 2)    tile column/row on the slide (used only to draw attention maps)
Labels  : TCGA project -> organ and cancer subtype (data/manifest.csv)
Split   : patient-level train / val / test (data/splits.csv), so no patient is in two splits.

Training text is generated from the labels: several phrasings of a few question types.
"""
from __future__ import annotations

import random
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "data/features/phikon"

SUBTYPES = {  # TCGA project -> (organ, subtype)
    "TCGA-LUAD": ("lung", "lung adenocarcinoma"),
    "TCGA-LUSC": ("lung", "lung squamous cell carcinoma"),
    "TCGA-KIRC": ("kidney", "clear cell renal cell carcinoma"),
    "TCGA-KIRP": ("kidney", "papillary renal cell carcinoma"),
    "TCGA-KICH": ("kidney", "chromophobe renal cell carcinoma"),
}
SUBTYPE_NAMES = [s for _, s in SUBTYPES.values()]

ASK_SUBTYPE = ["What type of cancer is shown in this slide?", "What is the diagnosis?",
               "Which tumor subtype is this?", "What cancer do you see in this whole slide image?"]
ASK_ORGAN = ["Which organ is this tissue from?", "What is the tissue of origin?",
             "Which organ does this slide come from?"]


def questions_for(row, rng: random.Random) -> list[tuple[str, str, str]]:
    """(kind, question, answer) for one slide. Every answer needs the slide to be right."""
    qa = [("subtype", q, row.subtype) for q in ASK_SUBTYPE]
    qa += [("organ", q, row.organ) for q in ASK_ORGAN]
    other = rng.choice([s for s in SUBTYPE_NAMES if s != row.subtype])
    qa += [("yes/no", f"Is this a {row.subtype}?", "yes"), ("yes/no", f"Is this a {other}?", "no")]
    same_organ = [s for o, s in SUBTYPES.values() if o == row.organ]
    qa.append(("choice", f"This is a {row.organ} tumor. Is it " + " or ".join(f"a {s}" for s in same_organ) + "?",
               row.subtype))
    return qa


def slides(split: str | None = None) -> pd.DataFrame:
    """Slides whose features exist on disk, with organ / subtype / split columns."""
    m = pd.read_csv(ROOT / "data/manifest.csv")
    m = m.merge(pd.read_csv(ROOT / "data/splits.csv")[["file_id", "split"]], on="file_id")
    m["organ"] = m.project.map(lambda p: SUBTYPES[p][0])
    m["subtype"] = m.project.map(lambda p: SUBTYPES[p][1])
    m = m[m.file_id.map(lambda f: (FEAT_DIR / f"{f}.h5").exists())]
    if split:
        m = m[m.split == split]
    return m.reset_index(drop=True)


def read_slide(file_id: str):
    with h5py.File(FEAT_DIR / f"{file_id}.h5", "r") as f:
        return f["features"][:], f["grid"][:]


def fixed_sample(file_id: str, n: int, draw: int = 0) -> np.ndarray:
    """Same n tile indices every time for a given slide (for evaluation and chat).

    draw picks a different, still repeatable, sample of the same slide (used for voting at test time).
    """
    with h5py.File(FEAT_DIR / f"{file_id}.h5", "r") as f:
        total = f["features"].shape[0]
    rng = np.random.default_rng(zlib.crc32(file_id.encode()) + draw)
    return np.sort(rng.permutation(total)[:n])


class TilePool:
    """Keeps up to `pool` random tiles per slide in RAM (fp16), and samples `n` of them per step.

    A slide has ~4,000 tiles (up to 20,000). Holding 2,048 per slide keeps ~300 slides under 1 GB,
    and drawing a fresh 512 each time is a cheap augmentation: the model never sees the same view twice.
    """

    def __init__(self, pool: int = 2048):
        self.pool, self.cache = pool, {}

    def get(self, file_id: str) -> np.ndarray:
        if file_id not in self.cache:
            feats, _ = read_slide(file_id)
            keep = np.random.default_rng(0).permutation(len(feats))[:self.pool]
            self.cache[file_id] = feats[keep].astype(np.float16)
        return self.cache[file_id]

    def sample(self, file_id: str, n: int, rng: np.random.Generator) -> np.ndarray:
        x = self.get(file_id)
        return x[rng.permutation(len(x))[:n]]


def pad_tiles(tile_sets: list[np.ndarray], device) -> tuple[torch.Tensor, torch.Tensor]:
    """List of (n_i, 768) arrays -> feats (B, N, 768) and mask (B, N), True where a real tile is."""
    N = max(len(t) for t in tile_sets)
    feats = torch.zeros(len(tile_sets), N, tile_sets[0].shape[1])
    mask = torch.zeros(len(tile_sets), N, dtype=torch.bool)
    for i, t in enumerate(tile_sets):
        feats[i, :len(t)] = torch.from_numpy(t.astype(np.float32))
        mask[i, :len(t)] = True
    return feats.to(device), mask.to(device)


def make_batch(rows: pd.DataFrame, pool: TilePool, n_tiles, n_questions: int, rng: random.Random,
               np_rng: np.random.Generator, device):
    """Each slide contributes `n_questions` (question, answer) pairs, all looking at the same tiles.

    n_tiles: an int, or (low, high) to draw a random tile count per slide. Seeing 128 tiles one step and
    512 the next stops the model leaning on any particular tile (a cheap regulariser).
    """
    tiles, questions, answers = [], [], []
    for _, row in rows.iterrows():
        n = n_tiles if isinstance(n_tiles, int) else int(np_rng.integers(n_tiles[0], n_tiles[1] + 1))
        x = pool.sample(row.file_id, n, np_rng)
        for _, q, a in rng.sample(questions_for(row, rng), n_questions):
            tiles.append(x)
            questions.append(q)
            answers.append(a)
    feats, mask = pad_tiles(tiles, device)
    return feats, mask, questions, answers

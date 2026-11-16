"""Build a slide manifest from the NIH Genomic Data Commons (open-access TCGA H&E diagnostic slides).

Each row = one diagnostic (FFPE, "DX") H&E slide with its project, organ and subtype labels.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from ..data import SUBTYPES

GDC = "https://api.gdc.cancer.gov"

FIELDS = [
    "file_id", "file_name", "file_size", "md5sum",
    "cases.submitter_id", "cases.project.project_id",
    "cases.diagnoses.primary_diagnosis", "cases.diagnoses.ajcc_pathologic_stage",
    "cases.diagnoses.ajcc_pathologic_t", "cases.diagnoses.ajcc_pathologic_n",
    "cases.diagnoses.tissue_or_organ_of_origin", "cases.demographic.gender",
]


def _first(lst, key, default=None):
    for item in lst or []:
        if item.get(key):
            return item[key]
    return default


def query_slides(projects=tuple(SUBTYPES)) -> pd.DataFrame:
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": list(projects)}},
        {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
        {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]}
    rows, start = [], 0
    while True:
        r = requests.post(f"{GDC}/files", json={"filters": filt, "fields": ",".join(FIELDS),
                                               "size": 1000, "from": start}, timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        for h in data["hits"]:
            case = h["cases"][0]
            proj = case["project"]["project_id"]
            diag = case.get("diagnoses", [])
            rows.append({
                "file_id": h["file_id"], "file_name": h["file_name"], "file_size": h["file_size"],
                "md5": h["md5sum"], "patient": case["submitter_id"], "project": proj,
                "organ": SUBTYPES[proj][0], "subtype": SUBTYPES[proj][1],
                "primary_diagnosis": _first(diag, "primary_diagnosis"),
                "stage": _first(diag, "ajcc_pathologic_stage"),
                "pT": _first(diag, "ajcc_pathologic_t"), "pN": _first(diag, "ajcc_pathologic_n"),
                "site": _first(diag, "tissue_or_organ_of_origin"),
                "gender": (case.get("demographic") or {}).get("gender"),
            })
        start += len(data["hits"])
        if start >= data["pagination"]["total"] or not data["hits"]:
            break
    return pd.DataFrame(rows)


def select_slides(df: pd.DataFrame, per_class: int, max_mb: float, seed: int = 0) -> pd.DataFrame:
    """One slide per patient, size-capped for laptop download budgets, balanced per subtype."""
    df = df.sample(frac=1.0, random_state=seed).drop_duplicates("patient")
    parts = []
    for _, g in df.groupby("project"):
        small = g[g.file_size / 1e6 <= max_mb].head(per_class)
        # rare classes (e.g. KICH) have few small slides: top up with the smallest remaining ones
        rest = g.drop(small.index).sort_values("file_size").head(per_class - len(small))
        parts += [small, rest]
    return pd.concat(parts).reset_index(drop=True)


def download(file_id: str, out: Path, retries: int = 3) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(f"{GDC}/data/{file_id}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
            tmp.rename(out)
            return out
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return out


def make_splits(manifest: pd.DataFrame, seed: int = 0, frac=(0.7, 0.1, 0.2)) -> pd.DataFrame:
    """Patient-level train / val / test split, balanced per subtype (one slide per patient here)."""
    parts = []
    for _, g in manifest.groupby("project"):
        g = g.sample(frac=1.0, random_state=seed)
        n_tr, n_va = round(len(g) * frac[0]), round(len(g) * frac[1])
        parts.append(g.assign(split=["train"] * n_tr + ["val"] * n_va + ["test"] * (len(g) - n_tr - n_va)))
    return pd.concat(parts)[["file_id", "patient", "project", "split"]]

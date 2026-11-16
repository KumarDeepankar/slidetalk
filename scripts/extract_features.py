"""Stage 0+1: build manifest, then stream slides: download -> tile -> embed -> save features -> delete slide.

Resumable: slides with an existing feature file are skipped. Disk use stays at ~1-2 slides at a time.

  python scripts/extract_features.py              # the 300 slides in data/manifest.csv (60 per subtype)
  python scripts/extract_features.py --limit 25    # a quick start: 25 slides, balanced over subtypes

Delete data/manifest.csv to pick a fresh set from the GDC (--per-class, --max-mb).
"""
import argparse
import sys
import threading
import queue
import time
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidetalk.prep import gdc  # noqa: E402
from slidetalk.prep.tile_encoder import build_tile_encoder  # noqa: E402
from slidetalk.prep.tiling import TileDataset, tile_coords  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--max-mb", type=float, default=400)
    ap.add_argument("--tile-encoder", default="phikon")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = keep every tissue tile (paper setting)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N slides this run (debug)")
    args = ap.parse_args()

    root = Path(args.root)
    feat_dir = root / "features" / args.tile_encoder
    feat_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.csv"

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        allslides = gdc.query_slides()
        allslides.to_csv(root / "gdc_all_slides.csv", index=False)
        manifest = gdc.select_slides(allslides, args.per_class, args.max_mb)
        manifest.to_csv(manifest_path, index=False)
    if not (root / "splits.csv").exists():
        gdc.make_splits(manifest).to_csv(root / "splits.csv", index=False)
    print(manifest.groupby("project").size().to_string())

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    encode, transform, dim = build_tile_encoder(args.tile_encoder, device)

    # round-robin over subtypes so a partially finished run is still class-balanced
    manifest = manifest.assign(_k=manifest.groupby("project").cumcount()).sort_values(["_k", "project"])
    todo = [r for r in manifest.itertuples() if not (feat_dir / f"{r.file_id}.h5").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} slides to process")

    # Prefetch the next download while the current slide is being embedded.
    dl_q: "queue.Queue" = queue.Queue(maxsize=1)

    def downloader():
        for r in todo:
            p = root / "slides" / r.file_name
            try:
                if not p.exists():
                    gdc.download(r.file_id, p)
                dl_q.put((r, p))
            except Exception as e:  # noqa: BLE001
                print(f"download failed {r.file_id}: {e}", flush=True)
        dl_q.put(None)


    threading.Thread(target=downloader, daemon=True).start()

    while (item := dl_q.get()) is not None:
        r, path = item
        t0 = time.time()
        try:
            slide = openslide.OpenSlide(str(path))
            coords, tile0, grid = tile_coords(slide)
            slide.close()
            if args.max_tiles and len(coords) > args.max_tiles:
                keep = np.sort(np.random.default_rng(0).choice(len(coords), args.max_tiles, replace=False))
                coords, grid = coords[keep], grid[keep]
            dl = DataLoader(TileDataset(path, coords, tile0, transform), batch_size=args.batch,
                            num_workers=args.workers, persistent_workers=False, prefetch_factor=4)
            feats = torch.cat([encode(x).cpu() for x in dl]).numpy().astype(np.float16)
            with h5py.File(feat_dir / f"{r.file_id}.h5.tmp", "w") as f:
                f["features"] = feats
                f["coords"] = coords
                f["grid"] = grid
                f.attrs["tile0"] = tile0
            (feat_dir / f"{r.file_id}.h5.tmp").rename(feat_dir / f"{r.file_id}.h5")
            dt = time.time() - t0
            print(f"[{r.project}] {r.file_id[:8]} tiles={len(coords):6d} {dt:6.1f}s ({len(coords)/dt:.0f} tiles/s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"failed {r.file_id}: {e}", flush=True)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

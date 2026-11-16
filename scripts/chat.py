"""Chat with a slide.

    python scripts/chat.py                                   # first test slide, interactive
    python scripts/chat.py --slide <file_id> -q "Which organ is this tissue from?"
    python scripts/chat.py --list                            # show test slides you can pick

Every answer saves an attention map (the previous one is deleted): one square per tile, brighter = the first answer
word looked at that tile more. Grey squares are tiles the model was not shown.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidetalk.data import fixed_sample, read_slide, slides  # noqa: E402
from slidetalk.evaluate import EVAL_TILES  # noqa: E402
from slidetalk.model import SlideTalker  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "runs/slidetalk"


def save_map(grid_all, idx, attn, path, cell=8):
    g = grid_all - grid_all.min(0)
    H, W = (g[:, 1].max() + 1) * cell, (g[:, 0].max() + 1) * cell
    img = np.full((H, W, 3), 255, np.uint8)
    for cx, cy in g:  # every tissue tile, light grey
        img[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell] = 225
    a = attn / attn.max()
    colors = cv2.applyColorMap((a * 255).astype(np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0]
    for (cx, cy), c in zip(g[idx], colors):  # tiles the model saw, coloured by attention
        img[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell] = c
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "best.pt"))
    ap.add_argument("--slide", help="file_id (default: first test slide)")
    ap.add_argument("-q", "--question", action="append")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tiles", type=int, default=EVAL_TILES)
    args = ap.parse_args()

    test = slides("test")
    if args.list:
        print(test[["file_id", "subtype"]].to_string())
        return
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SlideTalker().to(device)
    model.load_trainable(torch.load(args.ckpt, map_location=device)["state"])
    model.eval()

    fid = args.slide or test.file_id[0]
    truth = slides().set_index("file_id").subtype.get(fid, "unknown")
    feats, grid = read_slide(fid)
    idx = fixed_sample(fid, args.tiles)
    x = torch.from_numpy(feats[idx].astype(np.float32)).to(device)
    print(f"slide {fid}\n{len(feats)} tissue tiles, showing the model {len(idx)}   (truth: {truth})")

    last_map = []  # only the latest attention map is kept on disk

    def ask(q, n):
        print(f"\nQ: {q}\nA: {model.answer(x, q)}")
        for old in last_map:
            old.unlink(missing_ok=True)
        last_map.clear()
        path = OUT / f"attention_{fid[:8]}_{n}.png"
        save_map(grid, idx, model.attention_map(x, q), path)
        last_map.append(path)
        print(f"   attention map -> {path}")

    if args.question:
        for n, q in enumerate(args.question):
            ask(q, n)
        return
    print("\nAsk about the slide (empty line quits). Try: What type of cancer is shown in this slide? / "
          "Which organ is this tissue from? / Is this a lung adenocarcinoma?")
    n = 0
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        ask(q, n)
        n += 1


if __name__ == "__main__":
    main()

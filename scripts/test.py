"""Test SlideTalk on held-out patients, plus two controls that show cross-attention is doing the work.

    python scripts/test.py                                        # newest model
    python scripts/test.py --ckpt runs/slidetalk/best.pt --votes 5   # any model, 5 tile samples per slide

1. Accuracy per question type on the test split.
2. Control A, no slide: the same questions with cross-attention switched off (plain GPT-2).
3. Control B, swapped slide: every question is asked while showing a different patient's slide.
   If answers follow the slide that is shown, the information flows through cross-attention.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidetalk.data import ASK_SUBTYPE, SUBTYPE_NAMES, slides  # noqa: E402
from slidetalk.evaluate import EVAL_TILES, ask_all, eval_tiles  # noqa: E402
from slidetalk.model import SlideTalker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/slidetalk/best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--votes", type=int, default=1, help="tile samples per slide; majority answer wins")
    ap.add_argument("--tiles", type=int, default=EVAL_TILES)
    args = ap.parse_args()
    OUT = Path(args.ckpt).resolve().parent
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SlideTalker().to(device)
    model.load_trainable(torch.load(args.ckpt, map_location=device)["state"])
    model.eval()
    df = slides(args.split)
    print(f"{len(df)} {args.split} slides   gates {['%+.2f' % g for g in model.gates()]}\n")

    res = ask_all(model, df, votes=args.votes, n_tiles=args.tiles)
    print(f"{args.votes} tile sample(s) of {args.tiles} tiles per slide")
    acc = res.groupby("kind", sort=False).correct.mean()
    print("1. accuracy by question type")
    for k, v in acc.items():
        print(f"   {k:22s} {v:.2f}")

    sub = res[res.kind == "subtype"]
    print("\n   subtype confusion (rows = truth, columns = answer)")
    short = {s: s.split()[0][:5] for s in SUBTYPE_NAMES}
    cm = pd.crosstab(sub.truth.map(short), sub.answer.map(lambda a: short.get(a, "other")))
    print("   " + cm.to_string().replace("\n", "\n   "))

    q = ASK_SUBTYPE[0]
    none_answers = pd.Series([model.answer(None, q)] * 1)
    print(f"\n2. control A, no slide: '{q}' -> '{none_answers[0]}' (same answer for every patient)")

    perm = np.roll(np.arange(len(df)), len(df) // 2)
    swapped = ask_all(model, df, kinds=("subtype",), tiles=lambda i: eval_tiles(df.file_id[perm[i]], device))
    shown = df.subtype.values[perm]
    follows_shown = float((swapped.answer.str.lower().values == shown).mean())
    follows_asked = float(swapped.correct.mean())
    print("\n3. control B, swapped slide (each question shown another patient's slide)")
    print(f"   answer matches the slide shown        {follows_shown:.2f}")
    print(f"   answer matches the original patient   {follows_asked:.2f}")
    print("\n   chance for 5 subtypes: 0.20")

    out = dict(split=args.split, n_slides=len(df), votes=args.votes, tiles=args.tiles, accuracy=acc.to_dict(), no_slide_answer=none_answers[0],
               swapped_follows_shown=follows_shown, swapped_follows_original=follows_asked, gates=model.gates())
    tag = f"{args.split}_{Path(args.ckpt).stem}_v{args.votes}_t{args.tiles}"
    (OUT / f"{tag}_results.json").write_text(json.dumps(out, indent=1))
    res.to_csv(OUT / f"{tag}_answers.csv", index=False)
    print(f"\nsaved {OUT / f'{tag}_results.json'} and every answer in {tag}_answers.csv")


if __name__ == "__main__":
    main()

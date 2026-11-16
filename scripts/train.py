"""Train SlideTalk: only the projection and the 4 gated cross-attention blocks learn.

    python scripts/train.py                 # 25 epochs, about 8 min on an M3 Pro
    python scripts/train.py --epochs 3      # quick check that everything runs
    python scripts/train.py --mem-dropout 0 --min-tiles 512 --weight-decay 0.01 --out runs/x   # first version

Regularisers (all on by default): dropout on the slide memory, a random tile count per slide (128-512),
and weight decay. The best checkpoint is picked by validation loss, not accuracy on a few dozen slides.

Each epoch prints the loss, validation accuracy and the four gate values tanh(gate).
Watch the gates: they start at 0 (GPT-2 ignores the slide) and open as the model learns to use it.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slidetalk.data import TilePool, make_batch, slides  # noqa: E402
from slidetalk.evaluate import ask_all, val_loss  # noqa: E402
from slidetalk.model import SlideTalker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--slides-per-step", type=int, default=8)
    ap.add_argument("--questions-per-slide", type=int, default=4)
    ap.add_argument("--tiles", type=int, default=512, help="most tiles sampled per slide per step")
    ap.add_argument("--min-tiles", type=int, default=128, help="fewest tiles per slide per step (= --tiles: fixed)")
    ap.add_argument("--mem-dropout", type=float, default=0.1, help="dropout on the slide memory")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--out", default="runs/slidetalk")
    ap.add_argument("--log-every", type=int, default=5, help="print the running loss every N steps")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gate-lr", type=float, default=1e-2, help="gates are single numbers: let them move faster")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT = ROOT / args.out

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    rng, np_rng = random.Random(args.seed), np.random.default_rng(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_df, val_df = slides("train"), slides("val")
    print(f"slides with features: {len(train_df)} train, {len(val_df)} val   device: {device}")

    model = SlideTalker(mem_dropout=args.mem_dropout).to(device)
    gates = [b.gate for b in model.xattn]
    others = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".gate")]
    n_train = sum(p.numel() for p in others) + len(gates)
    print(f"trainable: {n_train / 1e6:.1f}M parameters   frozen GPT-2: "
          f"{sum(p.numel() for p in model.lm.parameters()) / 1e6:.0f}M")
    opt = torch.optim.AdamW([{"params": others, "lr": args.lr, "weight_decay": args.weight_decay},
                             {"params": gates, "lr": args.gate_lr, "weight_decay": 0.0}])
    steps_per_epoch = math.ceil(len(train_df) / args.slides_per_step)
    total, warmup = args.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    pool = TilePool()
    OUT.mkdir(parents=True, exist_ok=True)
    log, best = [], float("inf")
    n_tiles = args.tiles if args.min_tiles >= args.tiles else (args.min_tiles, args.tiles)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        order = train_df.sample(frac=1, random_state=epoch)
        for s in range(0, len(order), args.slides_per_step):
            batch = make_batch(order.iloc[s:s + args.slides_per_step], pool, n_tiles,
                               args.questions_per_slide, rng, np_rng, device)
            loss = model.loss(*batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(others, 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step = len(losses)
            if args.log_every and step % args.log_every == 0:
                print(f"  epoch {epoch:2d}  step {step:3d}/{steps_per_epoch}  loss {np.mean(losses[-args.log_every:]):.3f}"
                      f"  lr {sched.get_last_lr()[0]:.1e}", flush=True)
        model.eval()
        res = ask_all(model, val_df, kinds=("subtype", "organ"))
        acc = res.groupby("kind").correct.mean()
        row = dict(epoch=epoch, loss=float(np.mean(losses)), val_loss=val_loss(model, val_df),
                   val_subtype=float(acc["subtype"]),
                   val_organ=float(acc["organ"]), gates=model.gates(), seconds=round(time.time() - t0))
        log.append(row)
        print(f"epoch {epoch:2d}  loss {row['loss']:.3f}  val loss {row['val_loss']:.3f}  val subtype {row['val_subtype']:.2f}  "
              f"organ {row['val_organ']:.2f}  gates [{' '.join(f'{g:+.2f}' for g in row['gates'])}]  "
              f"{row['seconds']}s", flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save({"state": model.trainable_state(), "epoch": epoch, "args": vars(args)}, OUT / "best.pt")
        (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
    torch.save({"state": model.trainable_state(), "epoch": args.epochs, "args": vars(args)}, OUT / "last.pt")
    best_ep = min(log, key=lambda r: r["val_loss"])["epoch"]
    print(f"best val loss {best:.3f} at epoch {best_ep}; checkpoints in {OUT}")


if __name__ == "__main__":
    main()

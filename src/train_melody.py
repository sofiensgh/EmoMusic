"""Train the mood-conditioned LSTM on data/melodies.pt (built by build_music_dataset).

    python -m src.train_melody --epochs 30
"""
import argparse
import math
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import MELODY_CKPT, MELODY_DATA, MODELS_DIR, OUTPUT_DIR, SEQ_LEN
from .melody_lstm import MelodyLSTM


class Windows(Dataset):
    """Cuts every sequence into overlapping windows of SEQ_LEN+1 tokens."""

    def __init__(self, seqs, moods, stride=4):
        self.items = [(s, m, i) for s, m in zip(seqs, moods)
                      for i in range(0, len(s) - SEQ_LEN, stride)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        s, m, i = self.items[k]
        w = s[i:i + SEQ_LEN + 1]
        return w[:-1], w[1:], m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(MELODY_DATA))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--stride", type=int, default=4, help="step between training windows (bigger = fewer near-duplicates)")
    p.add_argument("--patience", type=int, default=0, help="stop after N epochs without val improvement (0 = off)")
    p.add_argument("--out", default=str(MELODY_CKPT))
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(a.data)
    pieces = data.get("pieces", list(range(len(data["sequences"]))))
    ids = sorted(set(pieces))
    random.Random(0).shuffle(ids)
    val_ids = set(ids[:max(1, len(ids) // 10)])  # 10% of pieces (all their transpositions) for validation
    train, val = [], []
    for s, m, pid in zip(data["sequences"], data["moods"], pieces):
        (val if pid in val_ids else train).append((s, m))
    train_ds, val_ds = Windows(*zip(*train), stride=a.stride), Windows(*zip(*val), stride=SEQ_LEN)
    print(f"device {device} | train windows {len(train_ds)} | val windows {len(val_ds)}")
    train_dl = DataLoader(train_ds, a.batch, shuffle=True, drop_last=len(train_ds) > a.batch)
    val_dl = DataLoader(val_ds, a.batch)

    model = MelodyLSTM(hidden=a.hidden, dropout=a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, y, m in val_dl:
                x, y, m = x.to(device), y.to(device), m.to(device)
                out, _ = model(x, m)
                tot += loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1)).item() * len(x)
                n += len(x)
        return tot / max(n, 1)

    hist_t, hist_v, best, since_best = [], [], float("inf"), 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, m in train_dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            out, _ = model(x, m)
            loss = loss_fn(out.reshape(-1, out.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoids exploding gradients in RNNs
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        sched.step()
        tl, vl = tot / n, evaluate()
        hist_t.append(tl); hist_v.append(vl)
        flag = ""
        if vl < best:
            best, since_best = vl, 0
            torch.save({"hparams": model.hparams, "state_dict": model.state_dict(), "val_loss": vl}, a.out)
            flag = "  * saved"
        else:
            since_best += 1
        print(f"ep {ep:3d}  loss {tl:.3f}/{vl:.3f}  val perplexity {math.exp(vl):6.1f}  {time.time() - t0:.0f}s{flag}",
              flush=True)
        if a.patience and since_best >= a.patience:
            print(f"early stop: no improvement for {a.patience} epochs")
            break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(hist_t, label="train"); plt.plot(hist_v, label="val")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.title("Melody LSTM"); plt.legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "melody_training_curves.png", dpi=120)
    print(f"best val loss {best:.3f} -> {a.out}")


if __name__ == "__main__":
    main()

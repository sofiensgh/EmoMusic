"""Train the facial-emotion classifier.

    python -m src.train_emotion --data data/fer2013.csv --epochs 40
    python -m src.train_emotion --data data/fer2013 --arch resnet18
    python -m src.train_emotion --smoke-test          # random data, checks the pipeline
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import EMOTION_CKPT, MODELS_DIR, MOODS, OUTPUT_DIR
from .emotion_cnn import build_model
from .emotion_dataset import FaceDataset, load_fer


def run_epoch(model, loader, loss_fn, device, opt=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    preds_all, y_all = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            pred = out.argmax(1)
            loss_sum += loss.item() * len(y)
            correct += (pred == y).sum().item()
            total += len(y)
            preds_all.append(pred.cpu())
            y_all.append(y.cpu())
    return loss_sum / total, correct / total, torch.cat(preds_all), torch.cat(y_all)


def plot_results(history, cm, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ["train_loss", "val_loss"]:
        ax[0].plot(history[k], label=k)
    for k in ["train_acc", "val_acc"]:
        ax[1].plot(history[k], label=k)
    ax[0].set_title("Loss"); ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(out_dir / "emotion_training_curves.png", dpi=120)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(MOODS)), MOODS); ax.set_yticks(range(len(MOODS)), MOODS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (val)")
    for i in range(len(MOODS)):
        for j in range(len(MOODS)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(out_dir / "emotion_confusion_matrix.png", dpi=120)
    plt.close("all")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/fer2013")  # folder with train/ and test/ (or a fer2013.csv)
    p.add_argument("--arch", default="cnn", choices=["cnn", "resnet18"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if a.smoke_test:
        rng = np.random.default_rng(0)
        tr = (rng.integers(0, 255, (256, 48, 48), dtype=np.uint8), rng.integers(0, 4, 256))
        va = (rng.integers(0, 255, (64, 48, 48), dtype=np.uint8), rng.integers(0, 4, 64))
        a.epochs, a.arch = 1, "cnn"
    else:
        tr, va = load_fer(a.data)
    print(f"train {len(tr[1])}  val {len(va[1])}  class counts {np.bincount(tr[1], minlength=4)}")

    train_dl = DataLoader(FaceDataset(*tr, augment=True), a.batch, shuffle=True, num_workers=a.workers)
    val_dl = DataLoader(FaceDataset(*va), a.batch, num_workers=a.workers)

    model = build_model(a.arch, pretrained=not a.smoke_test).to(device)
    # class weights fight the imbalance (FER has many more "happy" than "angry")
    counts = np.bincount(tr[1], minlength=len(MOODS)).clip(min=1)
    weights = torch.tensor(counts.sum() / (len(MOODS) * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl))

    class StepOpt:  # step the scheduler after every optimizer step
        def zero_grad(self): opt.zero_grad()
        def step(self): opt.step(); sched.step()

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    best = -1.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "smoke_emotion.pth" if a.smoke_test else EMOTION_CKPT
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tl, ta, _, _ = run_epoch(model, train_dl, loss_fn, device, StepOpt())
        vl, vacc, vp, vy = run_epoch(model, val_dl, loss_fn, device)
        for k, v in zip(history, [tl, vl, ta, vacc]):
            history[k].append(v)
        flag = ""
        if vacc > best:
            best = vacc
            torch.save({"arch": a.arch, "state_dict": model.state_dict(), "val_acc": vacc, "moods": MOODS}, out_path)
            best_preds = (vp, vy)
            flag = "  * saved"
        print(f"ep {ep:3d}  loss {tl:.3f}/{vl:.3f}  acc {ta:.3f}/{vacc:.3f}  {time.time() - t0:.0f}s{flag}")

    vp, vy = best_preds
    cm = np.zeros((len(MOODS), len(MOODS)), dtype=int)
    for t, pr in zip(vy.tolist(), vp.tolist()):
        cm[t, pr] += 1
    plot_results(history, cm, OUTPUT_DIR)
    print(f"best val acc {best:.3f} -> {out_path}")
    print(f"plots -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

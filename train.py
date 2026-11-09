"""Train the classifier head: synthetic drawings -> frozen EfficientNet-B0 features -> linear head.

    uv run python train.py [--per-class 60] [--epochs 60]

Writes models/classifier.pt (head weights + classes + metrics) and models/classifier_metrics.json, which also
scores the heuristic classifier on the same held-out images so the two can be compared honestly.
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from blueprint import CLASSES
from blueprint.classifier import HeuristicClassifier
from blueprint.samples import generate_dataset

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=Path, default=HERE / "models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.per_class, args.epochs, args.out, args.seed)


def train(per_class: int, epochs: int, out: Path, seed: int = 0) -> dict:
    import torch
    from torchvision import models, transforms

    torch.manual_seed(seed)
    random.seed(seed)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        items = generate_dataset(Path(tmp), per_class, seed)
        random.shuffle(items)
        print(f"generated {len(items)} synthetic drawings in {time.perf_counter() - t0:.1f}s")
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        backbone.classifier = torch.nn.Identity()
        backbone.eval()
        tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])  # fmt: skip
        feats, labels, heur = [], [], []
        heuristic = HeuristicClassifier()
        with torch.no_grad():
            for i in range(0, len(items), 32):
                batch = items[i : i + 32]
                imgs = [cv2.imread(str(p)) for p, _ in batch]
                x = torch.stack([tf(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs])
                feats.append(backbone(x))
                labels += [CLASSES.index(c) for _, c in batch]
                heur += [CLASSES.index(heuristic.predict(im)["class"]) for im in imgs]
    x = torch.cat(feats)
    y = torch.tensor(labels)
    heur_t = torch.tensor(heur)
    n_test = max(8, len(items) // 5)
    x_tr, y_tr, x_te, y_te, h_te = x[n_test:], y[n_test:], x[:n_test], y[:n_test], heur_t[:n_test]
    print(f"features {tuple(x.shape)} in {time.perf_counter() - t0:.1f}s; train {len(y_tr)}, test {len(y_te)}")

    head = torch.nn.Linear(x.shape[1], len(CLASSES))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(perm), 32):
            idx = perm[i : i + 32]
            loss = torch.nn.functional.cross_entropy(head(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(x_te).argmax(1)
        train_acc = float((head(x_tr).argmax(1) == y_tr).float().mean())
    acc = float((pred == y_te).float().mean())
    heur_acc = float((h_te == y_te).float().mean())
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(y_te.tolist(), pred.tolist(), strict=True):
        conf[t, p] += 1
    per_class_acc = {c: round(float(conf[i, i] / max(conf[i].sum(), 1)), 3) for i, c in enumerate(CLASSES)}
    metrics = {
        "method": "efficientnet_b0 frozen + linear head",
        "classes": CLASSES,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "per_class_images": per_class,
        "epochs": epochs,
        "accuracy": round(acc, 4),
        "train_accuracy": round(train_acc, 4),
        "per_class_accuracy": per_class_acc,
        "confusion": conf.tolist(),
        "heuristic_accuracy_same_split": round(heur_acc, 4),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "classes": CLASSES, "metrics": metrics}, out / "classifier.pt")
    (out / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"EfficientNet head accuracy {acc:.1%} (train {train_acc:.1%}); heuristic {heur_acc:.1%}; "
          f"per class {per_class_acc}; {metrics['train_seconds']}s")  # fmt: skip
    print(f"saved {out / 'classifier.pt'}")
    return metrics


if __name__ == "__main__":
    main()

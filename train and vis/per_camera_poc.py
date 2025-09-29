import argparse
import json
import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Paths ---
ARTIFACTS = Path("./artifacts")
DATA_DIR = ARTIFACTS / "synthetic_data"
MODEL_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"

for d in [ARTIFACTS, DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128


# --- Dataset Generation ---
def generate_synthetic_dataset(n_samples: int):
    manifest_rows = []
    for i in range(int(n_samples)):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (30, 30, 30))
        seg = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
        depth = Image.new("F", (IMG_SIZE, IMG_SIZE), 1.0)

        draw = ImageDraw.Draw(img)
        draw_seg = ImageDraw.Draw(seg)
        draw_depth = ImageDraw.Draw(depth)

        w = random.randint(12, 40)
        h = random.randint(12, 40)
        x = random.randint(0, IMG_SIZE - w - 1)
        y = random.randint(0, IMG_SIZE - h - 1)
        color = (
            int(random.uniform(60, 255)),
            int(random.uniform(60, 255)),
            int(random.uniform(60, 255)),
        )
        draw.rectangle([x, y, x + w, y + h], fill=color)
        draw_seg.rectangle([x, y, x + w, y + h], fill=1)

        dval = float(random.uniform(0.05, 0.9))
        for yy in range(y, y + h + 1):
            draw_depth.line([(x, yy), (x + w, yy)], fill=dval)

        xc = (x + w / 2) / IMG_SIZE
        yc = (y + h / 2) / IMG_SIZE
        w_norm = w / IMG_SIZE
        h_norm = h / IMG_SIZE

        img_path = DATA_DIR / f"img_{i:04d}.png"
        seg_path = DATA_DIR / f"seg_{i:04d}.png"
        depth_path = DATA_DIR / f"depth_{i:04d}.npy"
        img.save(img_path)
        seg.save(seg_path)
        np.save(depth_path, np.array(depth, dtype=np.float32))

        manifest_rows.append(
            {
                "image": str(img_path),
                "seg": str(seg_path),
                "depth": str(depth_path),
                "class": 1,
                "xc": float(xc),
                "yc": float(yc),
                "w": float(w_norm),
                "h": float(h_norm),
            }
        )

    manifest_csv = ARTIFACTS / "dataset_manifest.csv"
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "seg", "depth", "class", "xc", "yc", "w", "h"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_csv


# --- Dataset Class ---
class PerCameraDataset(Dataset):
    def __init__(self, manifest_csv, transform=None, max_samples=None):
        rows = []
        with open(manifest_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        self.rows = rows
        self.transform = transform or self._default_transform

    def _default_transform(self, img: Image.Image):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=2)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).float()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[int(idx)]
        img = Image.open(r["image"]).convert("RGB")
        seg = Image.open(r["seg"]).convert("L")
        depth = np.load(r["depth"])

        img_t = self.transform(img)
        seg_t = torch.from_numpy(np.array(seg, dtype=np.uint8, copy=True)).long()
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).float()

        ann_class = torch.tensor(int(r["class"]), dtype=torch.long)
        bbox = torch.tensor(
            [float(r["xc"]), float(r["yc"]), float(r["w"]), float(r["h"])],
            dtype=torch.float,
        )

        return {"image": img_t, "seg": seg_t, "depth": depth_t,
                "ann": {"class": ann_class, "bbox": bbox}}


# --- Model ---
class TinyPerCameraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.seg = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 2, 1),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.det = nn.Sequential(
            nn.Flatten(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 5)
        )

    def forward(self, x):
        f = self.conv(x)
        return {
            "seg": self.seg(f),
            "depth": self.depth(f),
            "det": self.det(self.pool(f)),
        }


# --- Collate ---
def collate_fn(batch):
    imgs = torch.stack([b["image"] for b in batch])
    segs = torch.stack([b["seg"] for b in batch])
    depths = torch.stack([b["depth"] for b in batch])
    classes = torch.stack([b["ann"]["class"] for b in batch])
    bboxes = torch.stack([b["ann"]["bbox"] for b in batch])
    return {"image": imgs, "seg": segs, "depth": depths,
            "ann": {"class": classes, "bbox": bboxes}}


# --- Train ---
def train(manifest_csv, epochs=1, batch_size=8, max_samples=128):
    ds = PerCameraDataset(manifest_csv, max_samples=max_samples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPerCameraNet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()
    det_cls_loss_fn = nn.BCEWithLogitsLoss()
    det_bbox_loss_fn = nn.L1Loss()

    stats_loss = []
    model.train()
    for _ in range(int(epochs)):
        for batch in loader:
            imgs, seg_t, depth_t = batch["image"].to(device), batch["seg"].to(device), batch["depth"].to(device)
            cls_t = batch["ann"]["class"].to(device).float().unsqueeze(1)
            bbox_t = batch["ann"]["bbox"].to(device)

            opt.zero_grad()
            out = model(imgs)

            seg_logits = nn.functional.interpolate(out["seg"], size=seg_t.shape[-2:], mode="bilinear")
            depth_pred = nn.functional.interpolate(out["depth"], size=depth_t.shape[-2:], mode="bilinear")

            seg_loss = seg_loss_fn(seg_logits, seg_t)
            depth_loss = depth_loss_fn(depth_pred, depth_t)
            det_cls_loss = det_cls_loss_fn(out["det"][:, 0:1], cls_t)
            det_bbox_loss = det_bbox_loss_fn(out["det"][:, 1:5], bbox_t)

            loss = seg_loss + depth_loss + det_cls_loss + 2.0 * det_bbox_loss
            loss.backward()
            opt.step()
            stats_loss.append(float(loss.item()))

    ckpt_p = MODEL_DIR / "poc_checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict()}, str(ckpt_p))

    # Save final results
    avg_loss = sum(stats_loss) / len(stats_loss)
    results = {"ckpt": str(ckpt_p), "avg_loss": avg_loss}
    with open(RESULTS_DIR / "poc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=128)
    args = parser.parse_args()

    manifest = generate_synthetic_dataset(args.samples)
    res = train(manifest, epochs=args.epochs, batch_size=args.batch, max_samples=args.max_samples)
    print(json.dumps(res, indent=2))

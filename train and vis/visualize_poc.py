import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from per_camera_poc import TinyPerCameraNet, PerCameraDataset, ARTIFACTS

# --- Paths ---
MODEL_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"
DATA_DIR = ARTIFACTS / "synthetic_data"

CHECKPOINT_PATH = MODEL_DIR / "poc_checkpoint.pt"

# --- Load checkpoint ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyPerCameraNet().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- Load dataset (few examples for visualization) ---
manifest_csv = ARTIFACTS / "dataset_manifest.csv"
dataset = PerCameraDataset(manifest_csv, max_samples=5)  # just visualize 5 samples

# --- Visualization ---
for i in range(len(dataset)):
    sample = dataset[i]
    img = sample["image"].unsqueeze(0).to(device)
    gt_seg = sample["seg"].numpy()
    gt_depth = sample["depth"].squeeze(0).numpy()

    with torch.no_grad():
        out = model(img)

        # Resize seg to match ground truth
        seg_logits = F.interpolate(
            out["seg"], size=gt_seg.shape, mode="bilinear", align_corners=False
        )
        pred_seg = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

        # Resize depth to match
        pred_depth = F.interpolate(
            out["depth"], size=gt_depth.shape, mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0).cpu().numpy()

    # Debug info: unique values
    print(f"Sample {i}")
    print(" GT Seg unique:", np.unique(gt_seg))
    print(" Pred Seg unique:", np.unique(pred_seg))

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(sample["image"].permute(1, 2, 0).numpy())
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_seg, cmap="gray")
    axes[1].set_title("Ground Truth Segmentation")
    axes[1].axis("off")

    axes[2].imshow(pred_seg, cmap="gray")
    axes[2].set_title("Predicted Segmentation")
    axes[2].axis("off")

    axes[3].imshow(pred_depth, cmap="viridis")
    axes[3].set_title("Predicted Depth")
    axes[3].axis("off")
    
    save_path = ARTIFACTS/RESULTS_DIR/ f"sample_{i:03d}.png"
    
    plt.savefig(save_path, dpi=150)
    plt.tight_layout()
    plt.show()

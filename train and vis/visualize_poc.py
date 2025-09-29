import os
import json
import csv
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from per_camera_poc import TinyPerCameraNet, PerCameraDataset, ARTIFACTS  # reuse your training code

CHECKPOINT_PATH = ARTIFACTS / "models/poc_checkpoint.pt"
RESULTS_PATH = ARTIFACTS / "results/poc_results.json"
PREDICTIONS_DIR = ARTIFACTS / "predictions"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = TinyPerCameraNet()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def visualize_results():
    # --- pick a sample image from synthetic dataset ---
    manifest_csv = ARTIFACTS / "dataset_manifest.csv"
    ds = PerCameraDataset(manifest_csv, max_samples=1)
    sample = ds[0]
    input_img = Image.fromarray((np.transpose(sample["image"].numpy(), (1, 2, 0)) * 255).astype(np.uint8))
    gt_mask = Image.fromarray(sample["seg"].numpy().astype(np.uint8) * 255)

    # --- model prediction ---
    model = load_checkpoint()
    with torch.no_grad():
        inp_tensor = sample["image"].unsqueeze(0)
        out = model(inp_tensor)
        pred_mask = out["seg"].argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
        pred_mask_img = Image.fromarray(pred_mask * 255)

    # --- visualize ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(input_img); axs[0].set_title("Input Image"); axs[0].axis("off")
    axs[1].imshow(gt_mask, cmap="gray"); axs[1].set_title("Ground Truth"); axs[1].axis("off")
    axs[2].imshow(pred_mask_img, cmap="viridis"); axs[2].set_title("Predicted"); axs[2].axis("off")
    plt.tight_layout(); plt.show()

    # save predicted mask
    pred_mask_img.save(PREDICTIONS_DIR / "predicted_mask.png")
    print(f"Saved predicted mask → {PREDICTIONS_DIR}/predicted_mask.png")

if __name__ == "__main__":
    visualize_results()

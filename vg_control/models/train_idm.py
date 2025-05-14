from __future__ import annotations
import ast
import torch
import tqdm
import datetime
import pathlib
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from data_utils.constants            import ROOT_DIR, ACTION_DIM
from vg_control.data.idm.idm_dataset import InverseDynamicsDataset
from vg_control.models.resnet        import RGB_ResNetInverseDynamics

BATCH_SIZE    = 6
FRAMES_BEFORE = 2
FRAMES_AFTER  = 2
VAL_SPLIT     = 0.25
NUM_EPOCHS    = 10
LR            = 1e-3
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

buttons_loss = nn.BCEWithLogitsLoss()
mouse_loss   = nn.MSELoss()
MOUSE_LAMBDA = 1.0

# build dataloaders -----------------------------------------------------------

def build_loaders(datalist_csv: str | Path):
    full = InverseDynamicsDataset(datalist_csv, root=ROOT_DIR)
    val_len = int(len(full) * VAL_SPLIT)
    train_ds, val_ds = random_split(full, [len(full) - val_len, val_len])
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw))

# pick context frames ---------------------------------------------------------
def split_frames(prev, nxt): return prev[:, -1], nxt[:, 0]

# training / validation step --------------------------------------------------

def step(model, batch, optim=None):
    train = optim is not None
    model.train() if train else model.eval()
    if train: optim.zero_grad()

    frames = batch['obs'].to(DEVICE).float()    # (B,T,3,H,W)
    target = batch['action'].to(DEVICE).float() # (B, 8)
    
    preds = model(frames.contiguous())
    loss = (
        buttons_loss(preds[:, :6], target[:, :6])
        + MOUSE_LAMBDA * mouse_loss(preds[:, 6:], target[:, 6:]))

    if train:
        loss.backward()
        optim.step()

    return loss.item()

# main loop -------------------------------------------------------------------

def train_idm(datalist_csv: str | Path):
    train_ld, val_ld = build_loaders(datalist_csv)
    clip_size = train_ld.dataset.dataset.meta['clip_size']
    model = RGB_ResNetInverseDynamics(
                num_frames=clip_size,
                resnet_type="resnet18",
                pretrained=True,
                dropout_p=0.2,
                output_dim=ACTION_DIM).to(DEVICE)

    optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_ld)*NUM_EPOCHS)

    for ep in range(NUM_EPOCHS):
        print(f"\nEpoch {ep+1}/{NUM_EPOCHS}")
        tl = 0.
        for batch in tqdm.tqdm(train_ld, desc="train"):
            l = step(model, batch, optim); tl += l; sched.step()
        print(f"train loss {tl/len(train_ld):.3f}")

        vl = 0.
        with torch.no_grad():
            for batch in tqdm.tqdm(val_ld, desc="val"):
                l = step(model, batch); vl += l
        print(f"val   loss {vl/len(val_ld):.3f}")

        ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save({
            "epoch": ep+1,
            "val_loss": vl/len(val_ld),
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
        }, ckpt_dir / f"idm_{ts}_e{ep+1}.pth")
        print("checkpoint saved")


if __name__ == "__main__":
    ROOT = Path(r"datalist.csv")
    train_idm(ROOT)

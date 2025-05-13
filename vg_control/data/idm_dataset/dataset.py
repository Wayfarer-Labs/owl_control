import json, random
import torch
import numpy as np
import pandas as pd
from decord import VideoReader, cpu

class InverseDynamicsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 index_file="data_dump/idm_index.parquet",
                 frames_before=2,
                 frames_after=2,
                 transform=None,
                 device="cpu"):
        self.meta = pd.read_parquet(index_file)
        self.frames_before = frames_before
        self.frames_after  = frames_after
        self.transform = transform      # torchvision transforms, albumentations...
        self.device = device

        # one VideoReader per file to avoid re-opening in __getitem__
        self._vreaders = {}

        # simple action-space-to-vector: you’ll likely replace this
        self._kb_onehot = {k:i for i,k in enumerate(range(256))}  # 0-255 keycodes
        self._btn_onehot = {k:i+256 for i,k in enumerate(range(8))}

    def _get_reader(self, path):
        if path not in self._vreaders:
            self._vreaders[path] = VideoReader(path, ctx=cpu(0))
        return self._vreaders[path]

    def _encode_action(self, event_type, event_args):
        if event_type == "KEYBOARD":
            keycode, key_down = event_args
            onehot = torch.zeros(264)
            onehot[self._kb_onehot[keycode]] = 1.
            onehot[-8] = float(key_down)      # append down/up flag
            return onehot
        elif event_type == "MOUSE_BUTTON":
            btn, state = event_args
            onehot = torch.zeros(264)
            onehot[self._btn_onehot[btn]] = 1.
            onehot[-7] = float(state)
            return onehot
        else:
            return torch.zeros(264)  # fallback

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        vr = self._get_reader(row.video)

        fr_idxs = list(range(row.frame_prev,
                             row.frame_prev + self.frames_before)) \
                + list(range(row.frame_next - self.frames_after + 1,
                             row.frame_next + 1))

        frames = vr.get_batch(fr_idxs).asnumpy()  # (T, H, W, 3) uint8
        frames = torch.from_numpy(frames).permute(0,3,1,2).float() / 255.

        if self.transform is not None:
            frames = self.transform(frames)

        prev = frames[:self.frames_before]
        nxt  = frames[self.frames_before:]

        action = self._encode_action(row.event_type, row.event_args)

        sample = dict(prev=prev, action=action, next=nxt,
                      meta=dict(**row))

        return sample

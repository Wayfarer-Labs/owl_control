from __future__ import annotations

import os
import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from toolz import pipe

from data_utils import utils
from data_utils.utils import code_from_ascii, seek_video_dirs
from data_utils.constants import FPS, ROOT_DIR, KEYBINDS


def to_button_data_tensor(
    action_data: pd.DataFrame,
    *,
    fps: int = FPS,
    filter_out_keys: tuple[str, ...] = ("LMB", "RMB"),
) -> torch.Tensor:
    """
    Turns button data into a tensor. Intended to be used for a whole video.

    Args:
        button_data: pd.DataFrame containing button input data
        fps: frames per second
        filter_out_keys: keys to filter out
    """
    valid_codes: list[int] = [code_from_ascii(k) for k in KEYBINDS if k not in filter_out_keys]
    return pipe(
        action_data,
        utils._normalize_timestamps,
        utils._filter_event_types,
        utils._filter_keys(valid_codes),
        utils._convert_events,
        utils._add_frame_column(fps),
        utils._simplify_event_types,
        utils._collapse_by_frame,
        utils._events_to_tensor(KEYBINDS),
    )

def mouse_data_to_tensor(
    action_data: pd.DataFrame,
    *,
    fps: int = FPS,
) -> torch.Tensor:
    return pipe(
        action_data,
        utils._normalize_timestamps,
        utils._filter_mouse_moves,
        utils._add_frame_column(fps),
        utils._parse_mouse_args,
        utils._aggregate_mouse_by_frame,
        utils._mouse_to_tensor,
    )


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=ROOT_DIR)
    p.add_argument("--fps", type=int, default=FPS)
    # TODO Maybe add split size here optionally and we can chunk shit in preprocess_videos
    return p.parse_args()


def preprocess_videos(root: str | os.PathLike):
    for vid_dir in tqdm(seek_video_dirs(root), desc="Processing videos"):
        csv_path     = Path(vid_dir) / "inputs.csv"
        mouse_path   = Path(vid_dir) / "mouse_full.pt"
        buttons_path = Path(vid_dir) / "buttons_full.pt"

        if buttons_path.exists() ^ mouse_path.exists(): print(f'Warning: {vid_dir} missing 1/2 files: \
                                                              {buttons_path.exists()=} , {mouse_path.exists()=}')
        if buttons_path.exists() and mouse_path.exists(): continue  # skip if already processed

        data = pd.read_csv(csv_path)
        button_data, mouse_data = to_button_data_tensor(data, fps=FPS), mouse_data_to_tensor(data, fps=FPS)
        
        torch.save(button_data, buttons_path)
        torch.save(mouse_data, mouse_path)


def main():
    args = _parse_args()
    global FPS, ROOT_DIR
    FPS, ROOT_DIR = args.fps, args.root
    preprocess_videos(ROOT_DIR)


if __name__ == "__main__":
    main()

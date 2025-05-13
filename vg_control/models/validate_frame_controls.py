"""
Validates controls and frames match up properly
"""

from game_gen_v2.common.nn.vae import VAE

import torch
import torch.nn.functional as F
import os
from pathlib import Path
from tinygrad.helpers import Timing
import numpy as np

from game_gen_v2.data.pipelines.data_config import FPS_OUT, KEYBINDS, OUT_DIR

FPS = FPS_OUT

FPS = 15
DATA_DIR = "E:/datasets/train_data_raw"

import cv2

def write_np_array_to_video(frames, output_path, fps=FPS, controls = None):
    """
    Write a numpy array of frames to a video file.

    :param frames: numpy array of shape [n_frames, height, width, channels]
    :param output_path: string, path to save the output video
    :param fps: int, frames per second for the output video
    """
    if not isinstance(frames, np.ndarray):
        raise ValueError("Input frames must be a numpy array")
    
    if len(frames.shape) != 4:
        raise ValueError("Input frames must have shape [n_frames, height, width, channels]")
    
    n_frames, height, width, channels = frames.shape


    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i, frame in enumerate(frames):
        if channels == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        frame = draw_controls(frame, controls[i])
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

def draw_controls(frame, control_vector):
    key_labels = KEYBINDS + ["LMB", "RMB"]
    key_pressed = [bool(value) for value in control_vector[:-2]]
    mouse_x_axis = float(control_vector[-2])/255
    mouse_y_axis = float(control_vector[-1])/255

    # Draw mouse arrow
    circle_center = (20, 20)
    cv2.circle(frame, circle_center, 15, (255, 255, 255), 1)
    arrow_end = (
        int(circle_center[0] + mouse_x_axis * 10),
        int(circle_center[1] + mouse_y_axis * 10)
    )
    cv2.arrowedLine(frame, circle_center, arrow_end, (0, 255, 0), 2)

    # Draw key boxes
    box_size = 12
    box_gap = 5
    start_x = 10
    start_y = 226  # 256 - 30 (bottom margin)

    for i, (label, pressed) in enumerate(zip(key_labels, key_pressed)):
        box_color = (0, 255, 0) if pressed else (0, 0, 255)
        box_x = start_x + i * (box_size + box_gap)
        cv2.rectangle(frame, (box_x, start_y), (box_x + box_size, start_y + box_size), box_color, -1)
        
        # Add label above the box
        label_y = start_y - 5
        cv2.putText(frame, label, (box_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return frame



def torch_to_np(x):
    # x is [b,c,h,w]
    if x.dtype == torch.uint8:
        return x.permute(0, 2, 3, 1).cpu().numpy()

    x = torch.clamp(x, -1, 1)
    x = (x + 1) * 127.5
    x = x.to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()

if __name__ == "__main__":
    # Get the first pair of vt and it files
    data_path = Path(DATA_DIR)
    vt_files = sorted(data_path.glob("*_video.pt"), key=lambda x: int(x.stem.split('_')[0]))
    it_files = sorted(data_path.glob("*_controls.pt"), key=lambda x: int(x.stem.split('_')[0]))

    first_vt_file = vt_files[1]
    first_it_file = it_files[1]

    with Timing("Time to load: "):
        video_tensor = torch.load(first_vt_file)
        input_tensor = torch.load(first_it_file)

    if video_tensor.shape[-3] > 3:
        vae = VAE()
        original_frames = vae.decode(video_tensor)
    else:
        original_frames = video_tensor
        original_frames = F.interpolate(original_frames.float(), (256, 256)).byte()
    original_frames = torch_to_np(original_frames)
    

    write_np_array_to_video(original_frames, "sanity/test.mp4", controls=input_tensor)


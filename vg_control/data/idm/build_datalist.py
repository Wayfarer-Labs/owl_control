from __future__ import annotations
from pathlib import Path
import decord, pandas as pd, json
from data_utils.constants import ROOT_DIR
from data_utils.utils import seek_video_dirs

# TODO Get decord working on GPU

def _num_frames(video_path: Path) -> int:
    vr = decord.VideoReader(str(video_path), num_threads=0)
    return len(vr)

def build_datalist(
        root: Path | str,
        out_csv: str = "datalist.csv",
        *,
        clip_size: int  = 6,         # number of frames to use per fwd step of IDM
        stride: int     = 1,         # frames skipped between each consecutive clip
        frame_skip: int = 1,         # frames skipped between each frame in clip, akin to 'dilation'
        allow_overlap: bool = False, # if False, ensures clips do not overlap by adjusting stride
        return_df: bool = False,
    ) -> pd.DataFrame | None:
    assert (frame_skip * clip_size * stride) > 0
    root = Path(root)
    clip_size *= frame_skip
    
    # clips not overlapping is equivalent to stride >= clip_size.
    # if the stride is larger than the clip size, then clips wont overlap by default.
    if not allow_overlap: 
        if stride >= clip_size: print(f"{allow_overlap=}, but stride is larger than (clip size x frame_skip) = {clip_size},\
                                       so clips won't overlap anyways.")
        stride += (clip_size if stride < clip_size else 0)

    rows: list[tuple[str, str, str]] = []

    for video_dir in seek_video_dirs(root):
        if not (videos := list(video_dir.glob("*.mp4"))) or len(videos) != 1:
            print(f"Found {len(videos)} videos in {video_dir} but expected 1") ; continue

        video_path: Path = videos[0].absolute()
        n_frames = _num_frames(video_path)

        rows.extend((str(video_path), start, clip_size)
                    for start in range(0,(n_frames- clip_size)+1,stride))

    df = pd.DataFrame(rows, columns=["video", "start", "clip_size"])
    print(f'Created {len(df)} rows in dataframe.')

    meta = dict(clip_size=clip_size, frame_skip=frame_skip, stride=stride)

    if out_csv:
        out_dir = Path(out_csv).parent ; out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "datalist.csv", index=False) ; print(f"Wrote {len(df)} rows to {out_csv}")
        with open(out_dir / "metadata.json", "w") as f: json.dump(meta, f)

    return df if return_df else None



def parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=ROOT_DIR)
    p.add_argument("--out", type=str, default="datalist.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_datalist(args.root, args.out)
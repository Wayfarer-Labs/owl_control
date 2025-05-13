import json, csv, math, pathlib, argparse, multiprocessing as mp
from functools import partial

import cv2                       # only to read fps cheaply
import pandas as pd
from tqdm import tqdm

HEADER = ["timestamp", "event_type", "event_args"]

def _time_to_frame_idx(ts, t0, fps):
    """Convert perf_counter timestamp → integer frame index."""
    return int(round((ts - t0) * fps))

def process_session(session_dir: pathlib.Path,
                    before=2, after=2, stride=1):
    """
    Build an index for ONE recording session:
    returns list[dict] with keys:
        run_dir, action_idx, frame_prev, frame_next, event_type, event_args
    """
    csv_path  = session_dir / "inputs.csv"
    mp4_path  = next(session_dir.glob("*.mp4"))  # assumes 1 mp4 / dir

    # 1. read CSV
    df = pd.read_csv(csv_path)

    # 2. locate alignment anchors
    try:
        t_start  = df.loc[df.event_type == "START", "timestamp"].iloc[0]
    except IndexError:
        raise ValueError(f"No START event in {csv_path}")

    cap  = cv2.VideoCapture(str(mp4_path))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    n_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 3. iterate over *action* rows (ignore mouse move spam etc.)
    ACTION_ROWS = df.event_type.isin(["KEYBOARD", "MOUSE_BUTTON"])
    rows = df[ACTION_ROWS].reset_index(drop=True)

    records = []
    for i, row in rows.iterrows():
        fid = _time_to_frame_idx(row.timestamp, t_start, fps)
        f_prev = max(fid - before, 0)
        f_next = min(fid + after,  n_fr - 1)

        # discard actions that would clip video edges
        if fid - before < 0 or fid + after >= n_fr:
            continue

        records.append(
            dict(
                run_dir=str(session_dir),
                video=str(mp4_path),
                csv=str(csv_path),
                action_idx=i,
                frame_prev=f_prev,
                frame_next=f_next,
                event_type=row.event_type,
                event_args=json.loads(row.event_args),
            )
        )
    return records

def main(root="data_dump", workers=8, before=2, after=2):
    root = pathlib.Path(root)
    sessions = [p for p in root.rglob("*") if p.is_dir()]

    fn = partial(process_session, before=before, after=after)
    with mp.Pool(workers) as pool:
        all_recs = list(tqdm(pool.imap_unordered(fn, sessions),
                             total=len(sessions)))

    flat = [r for sub in all_recs for r in sub]
    out = root / "idm_index.parquet"
    pd.DataFrame(flat).to_parquet(out, index=False)
    print(f"Wrote {len(flat)} samples → {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root",   default="data_dump")
    p.add_argument("--before", type=int, default=2,
                   help="frames before the action to return")
    p.add_argument("--after",  type=int, default=2,
                   help="frames after the action to return")
    p.add_argument("-j", "--workers", type=int, default=8)
    main(**vars(p.parse_args()))

import torch
import os, pathlib
import pandas as pd
from toolz import curry
from data_utils.keybinds import CODE_TO_KEY, KEY_TO_CODE


def seek_video_dirs(root: str | os.PathLike[str]) -> list[pathlib.Path]:
    root = pathlib.Path(root)
    return [p for p in root.rglob("*") if p.is_dir() and (p / "inputs.csv").exists()]


def ascii_from_code(code: int) -> str:
    return CODE_TO_KEY.get(code, f"Unknown<{code}>")

def code_from_ascii(k: str) -> int | None:
    return KEY_TO_CODE.get(k)

# -------- mouse
@curry
def _filter_mouse_moves(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.event_type == "MOUSE_MOVE"].copy()


def _parse_mouse_args(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[["dx", "dy"]] = pd.DataFrame(out.event_args.apply(lambda x: eval(x)).tolist(), index=out.index)
    return out


def _aggregate_mouse_by_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("frame", sort=False)[["dx", "dy"]]
          .mean()
          .reset_index()
    )


def _mouse_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    n_frames = int(df.frame.max()) + 1
    t = torch.zeros((n_frames, 2), dtype=torch.bfloat16)
    for _, row in df.iterrows():
        f = int(row.frame)
        t[f] = torch.tensor([row.dx, row.dy], dtype=torch.bfloat16)
    return t


# -------- keyboard

# -- transformations
def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    start = df.head(1000).loc[lambda d: d.event_type == "START"].iloc[-1].timestamp
    df = df.loc[df.timestamp >= start].copy()
    df.timestamp -= start
    end_rows = df.loc[df.event_type == "END"]
    if not end_rows.empty:
        df = df.loc[df.timestamp <= end_rows.iloc[0].timestamp].copy()
    return df.reset_index(drop=True)

def _filter_event_types(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.event_type.isin(["KEYBOARD", "MOUSE_BUTTON"])].copy()

@curry
def _filter_keys(valid_codes: list[int], df: pd.DataFrame) -> pd.DataFrame:
    k_mask = df.event_type == "KEYBOARD"
    if k_mask.any():
        k_df = df.loc[k_mask].copy()
        k_df["keycode"] = k_df.event_args.apply(lambda x: eval(x)[0])
        df = df.loc[~k_mask | k_df.keycode.isin(valid_codes)].copy()
    
    m_mask = df.event_type == "MOUSE_BUTTON"
    if m_mask.any():
        m_df = df.loc[m_mask].copy()
        m_df["button"] = m_df.event_args.apply(lambda x: eval(x)[0])
        df = df.loc[~m_mask | m_df.button.isin([1, 2])].copy()
    return df.reset_index(drop=True)


def _convert_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # Keyboard
    k_mask = out.event_type == "KEYBOARD"
    if k_mask.any():
        k_df = out.loc[k_mask].copy()
        k_df[["code", "is_pressed"]] = k_df.event_args.apply(lambda x: pd.Series(eval(x)))
        k_df.event_type = k_df.is_pressed.map({True: "KEY_DOWN", False: "KEY_UP"})
        k_df.event_args = k_df.code.apply(ascii_from_code)
        out.loc[k_mask] = k_df
    
    # Mouse
    m_mask = out.event_type == "MOUSE_BUTTON"
    if m_mask.any():
        m_df = out.loc[m_mask].copy()
        m_df[["button", "is_pressed"]] = m_df.event_args.apply(lambda x: pd.Series(eval(x)))
        m_df.event_type = m_df.is_pressed.map({True: "MOUSE_DOWN", False: "MOUSE_UP"})
        m_df.event_args = m_df.button.map({1: "LMB", 2: "RMB"})
        out.loc[m_mask] = m_df
    return out.reset_index(drop=True)


@curry
def _add_frame_column(fps: int, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["frame"] = (out.timestamp * fps).astype(int)
    return out


def _simplify_event_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[out.event_type.isin({"KEY_UP", "MOUSE_UP"}), "event_type"] = "UP"
    out.loc[out.event_type.isin({"KEY_DOWN", "MOUSE_DOWN"}), "event_type"] = "DOWN"
    return out


def _process_frame_events(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp")
    downs, ups = g[g.event_type == "DOWN"], g[g.event_type == "UP"]
    last = g.iloc[-1]

    if downs.any(axis=None) and not ups.any(axis=None): return downs.tail(1)
    if ups.any(axis=None) and not downs.any(axis=None): return ups.tail(1)
    if last.event_type == "DOWN":                       return pd.DataFrame([last])
    last = last.copy(); last.event_type = "TAP";        return pd.DataFrame([last])


def _collapse_by_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["frame", "event_args"], sort=False)
          .apply(_process_frame_events)
          .reset_index(drop=True)
    )


@curry
def _events_to_tensor(keybinds: list[str], df: pd.DataFrame) -> torch.Tensor:
    n_frames = int(df.frame.max()) + 1
    t = torch.zeros((n_frames, len(keybinds)), dtype=torch.bool)

    for _, row in df.iterrows():
        f, k = int(row.frame), keybinds.index(row.event_args)
        et = row.event_type
        if et == "DOWN": t[f:, k] = True
        if et == "UP": t[f:, k] = False
        else: t[f, k] = True; t[f + 1 :, k] = False  # tap

    return t

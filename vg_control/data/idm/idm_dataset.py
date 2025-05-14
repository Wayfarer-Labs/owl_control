from pathlib import Path
from typing import Sequence, Optional, Callable, Literal

import torch, decord, pandas as pd, json
from torch.utils.data import Dataset
import torch.utils.dlpack
from torchvision import transforms

__all__ = ["InverseDynamicsDataset"]

DECORD_CFG = dict(num_threads=2, ctx=decord.cpu(0))

Div255 = transforms.Lambda(lambda x: x.float().div_(255.))

class  _VideoCache(dict):
    def __init__(self, decord_cfg: dict = DECORD_CFG):
        super().__init__() ; self.decord_cfg = decord_cfg

    def __missing__(self, key: Path):
        vr = decord.VideoReader(str(key.absolute()), **self.decord_cfg)
        self[key] = vr ; return vr


class _ActionCache(dict):
    def __missing__(self, vid_dir: Path):
        btn   = torch.load(vid_dir / 'buttons_full.pt', mmap=True)
        mouse = torch.load(vid_dir / 'mouse_full.pt', mmap=True)
        self[vid_dir] = (btn, mouse) ; return (btn, mouse)

class InverseDynamicsDataset(Dataset):
    def __init__(self, datalist_csv: str | Path,
                 root: str | Path,
                 *,
                 transform: Callable = Div255,
                 decode_device: Literal['cpu', 'gpu'] = 'cpu',):
        self.root = Path(root)
        self.df = pd.read_csv(datalist_csv)
        self.meta = json.load(open(datalist_csv.parent / "metadata.json"))
        self.clip_size = self.meta['clip_size']
        
        global DECORD_CFG
        self.transform = transform
        self.decode_device = decord.gpu(0) if decode_device == 'gpu' else decord.cpu(0)
        # process-local caches
        self._videos  = _VideoCache({**DECORD_CFG,
                                     'ctx': self.decode_device})
        self._actions = _ActionCache()

    def _get_video(self, path_rel: str) -> decord.VideoReader: 
        return self._videos[self.root / path_rel]

    def _get_actions(self, path_rel: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._actions[(self.root / path_rel).parent]

    def __len__(self): return len(self.df)


    def _to_torch(self, nd: decord.ndarray.NDArray) -> torch.Tensor:
        # if we are on gpu, no memory transfers
        if nd.ctx.device_type == self.decode_device.device_type:
            return torch.utils.dlpack.from_dlpack(nd.to_dlpack())
        # otherwise, use numpy view
        return torch.from_numpy(nd.asnumpy())


    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | dict]:
        row       = self.df.iloc[idx]
        video_rel = row.video
        start_f   = int(row.start)
        
        vr = self._get_video(video_rel)

        frame_idxs: Sequence[int] = range(start_f, start_f + self.clip_size) # (t,h,w,3) uint8
        frames_nd: decord.ndarray.NDArray = vr.get_batch(frame_idxs)
        frames: torch.Tensor = self._to_torch(frames_nd)
        frames = frames.permute(0,3,1,2) # (t,h,w,c) -> (t,c,h,w)

        if self.transform: frames = self.transform(frames)

        btn, mouse = self._get_actions(video_rel)
        target_f   = start_f + self.clip_size - 1
        act_vec = torch.cat([btn[target_f].float(), mouse[target_f].float()], dim=-1)

        return {
            'obs': frames, # (t,c,h,w)
            'action': act_vec, # (action_dim,), which is like 8 or something
            'meta': {'video': video_rel, 'start': start_f, 'clip_size': self.clip_size}
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from data_utils.constants import ROOT_DIR
    ds = InverseDynamicsDataset(
        datalist_csv='datalist.csv', root=ROOT_DIR + '/data_dump/games/MCC-Win64-Shipping/', transform=None, decode_device=None
    )
    print(ds[0])

    dl = DataLoader(ds, batch_size=3, shuffle=True)
    for batch in dl:
        print(batch)
        break
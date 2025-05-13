"""Multiple dots generation dataset"""
from typing import List, Any, Optional

import torch

from .single import ContinuousMotionDataset, DeterministicMotionDataset, Sample


def create_three_datasets(
    size: int,
    batch_size: int,
    n_steps: int,
    std: float = 1.3,
    noise: float = 0.0,
    static_noise: float = 0.0,
    static_noise_speed: float = 0.0,
    structured_noise: bool = False,
    structured_dataset_path: Optional[str] = "/tmp/cifar10",
    img_size: int = 28,
    normalize: bool = False,
    sum_image: bool = False,
    device: Any = torch.device("cpu"),
    train: bool = False,
):
    d1 = ContinuousMotionDataset(
        size,
        batch_size=batch_size,
        n_steps=n_steps,
        std=std,
        noise=noise,
        static_noise=static_noise,
        static_noise_speed=static_noise_speed,
        structured_noise=structured_noise,
        structured_dataset_path=structured_dataset_path,
        img_size=img_size,
        normalize=normalize,
        device=device,
        train=train,
    )
    d2 = ContinuousMotionDataset(
        size,
        batch_size=batch_size,
        n_steps=n_steps,
        std=std,
        noise=noise,
        static_noise=static_noise,
        static_noise_speed=static_noise_speed,
        structured_noise=structured_noise,
        structured_dataset_path=structured_dataset_path,
        img_size=img_size,
        normalize=normalize,
        device=device,
        train=train,
    )
    d3 = DeterministicMotionDataset(
        size,
        batch_size=batch_size,
        n_steps=n_steps,
        std=std,
        noise=noise,
        static_noise=static_noise,
        static_noise_speed=static_noise_speed,
        structured_noise=structured_noise,
        structured_dataset_path=structured_dataset_path,
        img_size=img_size,
        normalize=normalize,
        max_step=0,
        device=device,
        train=train,
    )
    return MultiDotDataset([d1, d2, d3], sum_image=sum_image)


class MultiDotDataset(ContinuousMotionDataset):
    def __init__(
        self, datasets: List[torch.utils.data.Dataset], sum_image: bool = False
    ):
        self.datasets = datasets

        self.size = len(self.datasets[0])
        self.batch_size = self.datasets[0].batch_size
        self.sum_image = sum_image
        for d in self.datasets:
            assert (
                len(d) == self.size
            ), "Datasets in multi dot dataset must have equal sizes"

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

    def __getitem__(self, idx):
        samples = [x[idx] for x in self.datasets]
        states = torch.cat([x.states for x in samples], dim=-3)
        if self.sum_image:
            states = states.sum(dim=-3, keepdim=True)
        locations = torch.cat([x.locations for x in samples], dim=-2)
        actions = torch.cat([x.actions for x in samples], dim=-2)
        return Sample(states=states, locations=locations, actions=actions)

    def unnormalize_mse(self, x):
        # Doesn't matter which dataset gets to unnormalize
        return self.datasets[0].unnormalize_mse(x)

    def unnormalize_location(self, loc):
        # Doesn't matter which dataset gets to unnormalize
        return self.datasets[0].unnormalize_location(loc)

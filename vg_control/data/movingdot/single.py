
from typing import NamedTuple, Any, Optional
import math
import torch
import torchvision
from torchvision import transforms
import numpy as np

import importlib
import vg_control.data.movingdot.cifar as cifar

DATA_PATH = cifar.__path__[0]


class Sample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor  # [(batch_size), T, 2]
    actions: torch.Tensor  # [(batch_size), T, 2]


class ContinuousMotionDataset:
    def __init__(
        self,
        size,
        batch_size,
        n_steps=2,
        concentration: float = 0.2,
        max_step: float = 4.0,
        std: float = 1.3,
        noise: float = 0.0,
        static_noise: float = 0.0,
        structured_noise: bool = False,
        structured_dataset_path: Optional[str] = DATA_PATH,
        static_noise_speed: float = 0.0,
        img_size: int = 28,
        normalize: bool = False,
        device: Any = torch.device("cpu"),
        train: bool = True,
    ):
        self.size = size
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.concentration = concentration
        self.std = std
        self.max_step = max_step
        self.noise = noise
        self.static_noise = static_noise
        self.static_noise_speed = static_noise_speed
        self.img_size = img_size
        self.device = device
        self.structured_noise = structured_noise

        if self.structured_noise > 0:
            self.structured_dataset = torchvision.datasets.CIFAR10(
                root=structured_dataset_path,
                train=train,
                download=True,
                transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
            )
            
            # Pre-load and cache ALL images in VRAM (only ~180MB)
            all_images = []
            temp_loader = torch.utils.data.DataLoader(
                self.structured_dataset, 
                batch_size=1000,  # Load in large batches
                num_workers=0     # No workers needed for one-time load
            )
            for images, _ in temp_loader:
                # Process to 28x28 and move to GPU
                images = images[:, :, :28, :28].to(self.device)
                all_images.append(images)
            
            # Concatenate all batches into one tensor in VRAM
            self.cached_images = torch.cat(all_images, 0)
            self.num_images = self.cached_images.shape[0]

        self.normalize = False

        self.padding = 2 * self.std

        if normalize:
            self.n_mean, self.n_std = self.estimate_mean_std()
            self.normalize = True

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.generate_multistep_sample()

    def __iter__(self):
        for _ in range(self.size):
            yield self.generate_multistep_sample()

    def render_location(self, locations: torch.Tensor):
        x = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        y = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        c = torch.stack([xx, yy], dim=-1)
        c = c.view(*([1] * (len(locations.shape) - 1)), *c.shape).repeat(
            *locations.shape[:-1], *([1] * len(c.shape))
        )  # repeat the number of times required for locations.
        locations = locations.unsqueeze(-2).unsqueeze(
            -2
        )  # add dims for compatibility with c
        img = torch.exp(
            -(c - locations).norm(dim=-1).pow(2) / (2 * self.std * self.std)
        ) / (2 * math.pi * self.std * self.std)
        return img

    def generate_state(self):
        # We leave 2 * self.std margin when generating state, and don't let the
        # dot approach the border.
        effective_range = (self.img_size - 1) - 2 * self.padding
        location = (
            torch.rand(size=(self.batch_size, 2), device=self.device) * effective_range
            + self.padding
        )
        return location

    def get_images_for_overlay(self):
        # Randomly sample from pre-cached images
        indices = torch.randint(0, self.num_images, (self.batch_size,), device=self.device)
        return self.cached_images[indices]

    def generate_static_overlay(
        self,
        shape,
    ):
        # Shape is BS x T x H x W
        if self.structured_noise:
            images = self.get_images_for_overlay()
            images = images.unsqueeze(1)
            repeat = [1] * len(images.shape)
            repeat[1] = shape[1]
            overlay = images.repeat(*repeat)
        else:
            static_noise_overlay = (
                torch.rand(shape[0], *shape[2:], device=self.device)
            ).unsqueeze(1)
            repeat = [1] * len(static_noise_overlay.shape)
            repeat[1] = shape[1]
            overlay = static_noise_overlay.repeat(*repeat)
        return overlay

    def generate_rnd_overlay(self, shape):
        if self.structured_noise:
            images = []
            for i in range(shape[1]):
                images.append(self.get_images_for_overlay())
            overlay = torch.stack(images, dim=1)
        else:
            overlay = torch.rand(shape, device=self.device)
        return overlay

    def generate_multistep_sample(
        self,
    ):
        actions = self.generate_actions(self.n_steps)
        start_location = self.generate_state()
        sample = self.generate_transitions(start_location, actions)
        if self.static_noise > 0 or self.noise > 0:
            # static noise means just one noise overlay for all timesteps
            static_noise_overlay = (
                self.generate_static_overlay(sample.states.shape) * sample.states.max()
            )
            if self.static_noise_speed > 0:
                for i in range(static_noise_overlay.shape[1]):
                    static_noise_overlay[:, i] = torch.roll(
                        static_noise_overlay[:, i],
                        shifts=int(i * self.static_noise_speed),
                        dims=-1,
                    )
            rnd_noise_overlay = (
                self.generate_rnd_overlay(sample.states.shape) * sample.states.max()
            )
            static_noised_states = (
                sample.states
                + static_noise_overlay * self.static_noise
                + rnd_noise_overlay * self.noise
            )
            sample = Sample(
                states=static_noised_states,
                locations=sample.locations,
                actions=sample.actions,
            )
        if self.normalize:
            sample = Sample(
                states=(sample.states - self.n_mean) / self.n_std,
                locations=(
                    (sample.locations - self.padding)
                    / (self.img_size - 1 - 2 * self.padding)
                    - 0.5
                )
                / np.sqrt(1 / 12),
                actions=sample.actions / (0.4102 * self.max_step + 1e-7),
            )
        return sample

    def normalize_location(self, location):
        return (
            (location - self.padding) / (self.img_size - 1 - 2 * self.padding) - 0.5
        ) / np.sqrt(1 / 12)

    def generate_transitions(
        self,
        location,
        actions,
    ):
        locations = [location]
        for i in range(actions.shape[1]):
            next_location = self.generate_transition(locations[-1], actions[:, i])
            locations.append(next_location)

        # Unsqueeze for compatibility with multi-dot dataset
        locations = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)
        states = self.render_location(locations)
        return Sample(states, locations, actions)

    def generate_transition(self, location, action):
        next_location = location + action  # [..., :-1] * action[..., -1]
        # don't let the dot get closer to the border than 2 * self.std
        next_location = torch.clamp(
            next_location, min=self.padding, max=self.img_size - 1 - self.padding
        )
        return next_location

    def generate_actions(self, n_steps: int):
        x = torch.rand(self.batch_size, device=self.device) * 2 * math.pi
        d = torch.distributions.VonMises(
            x, concentration=self.concentration * torch.ones_like(x)
        )
        a = (
            d.sample((n_steps - 1,)).permute(1, 0).to(self.device)
        )  # put the batch dim first
        step_sizes = (
            torch.rand(self.batch_size, n_steps - 1, 1, device=self.device)
            * self.max_step
        )
        vec = ContinuousMotionDataset.angle_to_vec(a)
        actions = vec * step_sizes
        return actions

    @staticmethod
    def angle_to_vec(a):
        return torch.stack([torch.cos(a), torch.sin(a)], dim=-1)

    @torch.no_grad()
    def estimate_mean_std(self):
        N = 100
        means = []
        stds = []
        for i in range(N):
            imgs = self[i].states.flatten()
            means.append(imgs.mean().item())
            stds.append(imgs.std().item())

        return np.mean(means), np.mean(stds)

    @torch.no_grad()
    def unnormalize_mse(self, mse):
        if not self.normalize:
            return mse
        else:
            return mse * (
                (np.sqrt(1 / 12) * (self.img_size - 1 - 2 * self.padding)) ** 2
            )

    @torch.no_grad()
    def normalize_mse(self, mse):
        if self.normalize:
            return mse
        else:
            return mse / ((np.sqrt(1 / 12) * (self.img_size - 1 - 2 * self.padding)) ** 2)

    @torch.no_grad()
    def unnormalize_location(self, locations):
        if not self.normalize:
            return locations
        else:
            return ((locations * np.sqrt(1 / 12)) + 0.5) * (
                self.img_size - 1 - 2 * self.padding
            ) + self.padding


class DeterministicMotionDataset(ContinuousMotionDataset):
    def generate_actions(self, n_steps: int):
        """We sample just one action and then repeat it for the whole episode."""
        x = torch.rand(self.batch_size, 1, device=self.device) * 2 * math.pi
        a = x.repeat(1, n_steps - 1)
        vec = ContinuousMotionDataset.angle_to_vec(a)
        step_sizes = (
            torch.rand(self.batch_size, n_steps - 1, 1, device=self.device)
            * self.max_step
        )
        actions = vec * step_sizes
        return actions

if __name__ == "__main__":
    dataset = ContinuousMotionDataset(size=1000, batch_size=10, n_steps=2, concentration=0.2, max_step=4.0, std=1.3, noise=0.0, static_noise=0.0, structured_noise=False, structured_dataset_path="users/sboughan/scratch/ssl/cifar", static_noise_speed=0.0, img_size=28, normalize=False, device=torch.device("cpu"), train=True)
    sample = dataset[0]
    print(sample.states.shape)
    print(sample.locations.shape)
    print(sample.actions.shape)
    torchvision.datasets.CIFAR10(root="/users/sboughan/scratch/ssl/cifar", train=True, download=True, transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
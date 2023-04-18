## Begin model

import torch
import numpy as np
from typing import Tuple, Dict, Optional

from utils import device

# main parameters to tune
batch_size: int = 8  # number of training instances happening at once (in parallel)
seed: int = 1  # to fix the randomness
torch.manual_seed(seed)
epochs: int = 1
eval_iter: int = 50
lr: float = 0.001
n_layer: int = 8
dropout: float = 0.2  # percent of indermediate calculations that are disabled
dim: int = 2**4


class GeoGuesser(torch.nn.Module):
    # inspiration from https://arxiv.org/pdf/1512.03385.pdf

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        im_res: torch.Size,
        train_idxs: torch.Tensor,
        test_idxs: torch.Tensor,
    ):
        super().__init__()
        self.dataset = dataset
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs

        self.im_res = im_res

        # create the network
        self.out_size = 6  # x, y, z, lat, lon, compass
        self.network = torch.nn.Sequential(
            *[
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1, 1)),  # use maxpool?
                torch.nn.Flatten(),
                # torch.nn.LayerNorm(),
                torch.nn.Linear(64, self.out_size),  # final FC layer
            ]
        )

        self.l2_loss = torch.nn.MSELoss()

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if len(x.shape) == 3:  # no batch, single image
            x = torch.unsqueeze(x, dim=0)  # batch of 1
        B, C, H, W = x.shape  # (batch, channels, height, width)
        logits = self.network.forward(x)
        loss = self.loss_function(logits, y) if y is not None else None
        return logits, loss

    def loss_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.shape == (batch_size, self.out_size)
        assert y.shape == x.shape
        xyz_pred = x[:, :3]
        gps_pred = x[:, 3:]
        xyz_target = y[:, :3]
        gps_target = y[:, 3:]
        distance_xyz = self.l2_loss(xyz_pred, xyz_target)
        # TODO: find a better metric for distance of GPS
        distance_gps = self.l2_loss(gps_pred, gps_target)
        return distance_xyz + distance_gps

    def sample_batch(
        self, type: str = "train"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # return a randomized batch of data from the corresponding dataset
        data = self.train_idxs if type == "train" else self.test_idxs
        start_idx = torch.randint(low=1, high=max(data) - 1, size=(batch_size, 1))
        data = [self.dataset[int(i)] for i in start_idx]
        images = torch.stack([image for image, _, _ in data])
        xyz = torch.stack([xyz for _, xyz, _ in data])
        gps = torch.stack([gps for _, _, gps in data])
        assert images.shape == (batch_size, *self.im_res)
        assert xyz.shape == (batch_size, 3)
        assert gps.shape == (batch_size, 3)
        return images, xyz, gps

    def begin_training(self):
        self.train()

    # create a loss estimator for averaging training and val loss
    def estimate_loss(self, num_iters: int = 20) -> Tuple[float, float]:
        losses: Dict[str, float] = {}
        with torch.no_grad():  # no need to track gradients (lower memory footprint)
            self.eval()  # switch to evaluation mode
            for split in ["train", "valid"]:
                cumulative_loss: float = 0
                for i in range(num_iters):
                    img, xyz, gps = self.sample_batch(split)
                    xyzgps = torch.hstack((xyz, gps))
                    _, loss = self.forward(img, xyzgps)
                    cumulative_loss += loss.item()
                    print(
                        f"({split}) Eval: {100 * i / num_iters:.0f}%",
                        end="\r",
                        flush=True,
                    )
                losses[split] = cumulative_loss / num_iters
            self.train()  # back to training phase
        return losses["train"], losses["valid"]

## Begin model

import torch
import os
from typing import Tuple, Dict, Optional

# main parameters to tune
batch_size: int = 8  # number of training instances happening at once (in parallel)
seed: int = 1  # to fix the randomness
torch.manual_seed(seed)
epochs: int = 2000
eval_iter: int = 50
lr: float = 0.01
n_layer: int = 8
dropout: float = 0.2  # percent of indermediate calculations that are disabled
ckpt_dir: str = "ckpt"
os.makedirs(ckpt_dir, exist_ok=True)


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
            # first set of CONV->RELU->POOL
            torch.nn.Conv2d(
                in_channels=self.im_res[0],  # colour channels
                out_channels=50,
                kernel_size=5,
                stride=1,
                bias=False,
                padding=1,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # next set of CONV->RELU->POOL
            torch.nn.Conv2d(
                in_channels=50,
                out_channels=20,  # number of filters to learn
                kernel_size=5,  # (3x3) filter size
                stride=1,
                bias=False,
                padding=1,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),
            # finall FC7 (layer)
            torch.nn.Flatten(),
            torch.nn.Linear(700, 50),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(50, self.out_size),  # final FC layer
        )

        self.l2_loss = torch.nn.MSELoss()
        self.l2_loss = torch.nn.L1Loss()

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
        distance_gps = 0  # self.l2_loss(gps_pred, gps_target)
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

    def get_ckpt(self, id: int = -1):
        if id is None or id < 0:
            id = max([l for l in os.listdir(ckpt_dir)])
        return os.path.join(ckpt_dir, f"ckpt_{id}.pt")

    def load(self, ckpt: int = -1) -> None:
        ckpt_path: str = self.get_ckpt(ckpt)
        with open(ckpt_path, "rb") as f:
            self.load_state_dict(torch.load(f))
            print(f'Loaded state dict from "{ckpt_path}" successfully!')
            print()

    def begin_training(self) -> None:
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        train_loss = float("nan")
        val_loss = float("nan")
        for epoch in range(epochs):
            img, xyz, gps = self.sample_batch()
            xyzgps = torch.hstack((xyz, gps))
            _, loss = self.forward(img, xyzgps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            est_loss: bool = epoch % eval_iter == eval_iter - 1
            if est_loss:
                train_loss, val_loss = self.estimate_loss(num_iters=eval_iter)
                torch.save(self.state_dict(), self.get_ckpt(epoch))
                scheduler.step(loss)
            print(
                f"Training {epoch:>4}/{epochs} \t ({100 * epoch / epochs:.1f}%) \t Train loss: {train_loss:.2f} \t Val loss: {val_loss:.2f}",
                end="\r",
                flush=True,
            )
            if est_loss:
                print()

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

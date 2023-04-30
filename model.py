## Begin model

import torch
import os
from typing import Tuple, Dict, Optional, List

# main parameters to tune
batch_size: int = 32  # number of training instances happening at once (in parallel)
seed: int = 1  # to fix the randomness
torch.manual_seed(seed)
epochs: int = 6000
eval_iter: int = 50
lr: float = 0.001
fc_dim: int = 5000  # dimensionality of the final fully connected layer
dropout: float = 0.2  # percent of indermediate calculations that are disabled
regularization: float = 0.01  # regularization strength
momentum: float = 0.9
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
        # don't track these buffers as "model parameters"
        self.register_buffer("train_idxs", train_idxs, persistent=False)
        self.register_buffer("test_idxs", test_idxs, persistent=False)

        self.im_res = im_res
        c, h, w = self.im_res

        # create the network
        self.out_size = 3  # x, y, z, lat, lon, compass

        class ConvReluBlock(torch.nn.Module):
            def __init__(self, in_size: int, out_size: int, kernel_size: int):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=in_size,
                        out_channels=out_size,
                        kernel_size=kernel_size,
                        stride=1,
                        bias=False,
                        padding=kernel_size // 2,  # maintain shape!
                    ),
                    torch.nn.ReLU(inplace=True),
                )

            def forward(self, x):
                # print(x.shape)
                return self.network.forward(x)

        self.network = torch.nn.Sequential(
            ConvReluBlock(in_size=c, out_size=24, kernel_size=3),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ConvReluBlock(in_size=24, out_size=30, kernel_size=3),
            torch.nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),
            torch.nn.ReLU(inplace=True),
            # finall FC7 (layer)
            torch.nn.Flatten(),  # convert to latent vector space for FC layer
            torch.nn.Linear(23520, fc_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(fc_dim, self.out_size),  # final FC layer
        )

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

        print(f"State dict: {list(self.state_dict().keys())}")

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if len(x.shape) == 3:  # no batch, single image
            x = torch.unsqueeze(x, dim=0)  # batch of 1
        B, C, H, W = x.shape  # (batch, channels, height, width)
        logits = self.network.forward(x)
        if y is not None and len(y.shape) == 1:  # no batch, single sample
            y = torch.unsqueeze(y, dim=0)  # batch of 1
        loss = self.loss_function(logits, y) if y is not None else None
        return logits, loss

    def loss_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]  # batch size
        assert x.shape == (B, self.out_size)
        assert y.shape == x.shape
        # xyz_pred = x[:, :3]
        # gps_pred = x[:, 3:]
        # xyz_target = y[:, :3]
        # gps_target = y[:, 3:]
        gps_pred = x
        gps_target = y
        # TODO: are we using xyz at all?
        distance_xyz = 0  # self.l2_loss(xyz_pred, xyz_target)
        # TODO: find a better metric for distance of GPS
        distance_gps = self.l1_loss(gps_pred, gps_target)
        return distance_xyz + distance_gps

    def sample_batch(
        self, type: str = "train"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # return a randomized batch of data from the corresponding dataset
        data = self.train_idxs if type == "train" else self.test_idxs
        start_idx = torch.randint(low=1, high=data.max() - 1, size=(batch_size, 1))
        data = [self.dataset[int(i)] for i in start_idx]
        images = torch.stack([image for image, _, _ in data])
        # xyz = torch.stack([xyz for _, xyz, _ in data])
        gps = torch.stack([gps for _, _, gps in data])
        assert images.shape == (batch_size, *self.im_res)
        # assert xyz.shape == (batch_size, 3)
        assert gps.shape == (batch_size, 3)
        return images, gps

    def get_ckpt(self, id: int = -1) -> str:
        if id is None or id < 0:
            try:
                # only take the int (epoch) component
                files = os.listdir(ckpt_dir)
                id = max([int(f.strip("ckpt_").strip(".pt")) for f in files])
            except Exception as e:
                id = 0
        return os.path.join(ckpt_dir, f"ckpt_{id}.pt")

    def load(self, ckpt: Optional[int] = -1) -> None:
        ckpt_path: str = self.get_ckpt(ckpt)
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "rb") as f:
                self.load_state_dict(torch.load(f))
                print(f'Loaded state dict from "{ckpt_path}" successfully!')
                print()
        else:
            print(f'No ckpt found @ "{ckpt_path}"')

    def begin_training(self) -> None:
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=regularization,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        train_loss, val_loss = self.estimate_loss()  # initial losses
        for epoch in range(epochs):
            img, gps = self.sample_batch()
            # xyzgps = torch.hstack((xyz, gps))
            _, loss = self.forward(img, gps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            print(
                f"Epoch {epoch:>4}/{epochs} \t ({100 * epoch / epochs:.1f}%) \t Train loss: {train_loss:.2f} \t Val loss: {val_loss:.2f}",
                end="\r",
                flush=True,
            )
            if epoch % eval_iter == 0 and epoch > 0:
                print()
                train_loss, val_loss = self.estimate_loss(num_iters=eval_iter)
                torch.save(self.state_dict(), self.get_ckpt(epoch))
                scheduler.step(loss)
        print()
        print(
            f"Epoch {epochs}/{epochs} \t ({100:.1f}%) \t Train loss: {train_loss:.2f} \t Val loss: {val_loss:.2f}"
        )
        torch.save(self.state_dict(), self.get_ckpt(epochs))

    # create a loss estimator for averaging training and val loss
    def estimate_loss(self, num_iters: int = 20) -> Tuple[float, float]:
        losses: Dict[str, float] = {}
        with torch.no_grad():  # no need to track gradients (lower memory footprint)
            self.eval()  # switch to evaluation mode
            for split in ["train", "valid"]:
                cumulative_loss: float = 0
                for i in range(num_iters):
                    img, gps = self.sample_batch(split)
                    _, loss = self.forward(img, gps)
                    cumulative_loss += loss.item()
                    print(
                        f"({split}) Eval: {100 * i / num_iters:.0f}%",
                        end="\r",
                        flush=True,
                    )
                losses[split] = cumulative_loss / num_iters
            self.train()  # back to training phase
        # https://stackoverflow.com/questions/5419389/how-to-overwrite-the-previous-print-to-stdout
        for x in range(75):  # line clearing
            print("*" * (75 - x), x, end="\x1b[1K\r")
        return losses["train"], losses["valid"]

## ImageDataset to (lazily) load images and associated metadata

import torch
import numpy as np
from typing import Tuple
from PIL import Image
import os

# get this for MacOS: https://discuss.pytorch.org/t/failed-to-load-image-python-extension-could-not-find-module/140278/8
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # no more nasty torch warnings
from torchvision import transforms

from utils import device


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = None, res: float = 1):
        super().__init__()
        self.data_dir = data_dir
        # self.image_dir = "images"
        self.image_dir = os.path.join("images", "lowres")  # use low-res
        self.dataset_size = len(os.listdir(os.path.join(data_dir, self.image_dir)))
        self.xyz_cartesian = np.loadtxt(os.path.join(data_dir, "xyz_cartesian.txt"))
        self.gps_compass = np.loadtxt(os.path.join(data_dir, "gps_compass.txt"))
        assert len(self.xyz_cartesian) == len(self.gps_compass) >= self.dataset_size
        # scale factor for images (resolution scale)
        assert 0 < res <= 1
        # get the resolution for the images
        example_im: str = os.path.join(data_dir, self.image_dir, "000001_4.jpg")
        assert os.path.exists(example_im)
        self.im_res = (transforms.ToTensor()(Image.open(example_im))).shape
        # initialize transformations
        self.to_tensor = transforms.ToTensor()
        self.resize_im = transforms.Resize(
            (int(self.im_res[1] * res), int(self.im_res[2] * res))
        )

    def __len__(self) -> int:
        return self.dataset_size

    # given an image index, return the image itself as well as its associated metadata
    def __getitem__(
        self, idx: int, view: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert 0 <= idx < self.dataset_size
        assert 0 <= view <= 5
        # TODO: keep a running batch of images loaded in memory (limited to some number) and
        # use this as a running queue for fast img access time
        img_path = os.path.join(self.data_dir, self.image_dir, f"{idx:06d}_{view}.jpg")
        if not os.path.exists(img_path):
            return None, None, None
        im: torch.Tensor = self.to_tensor(self.resize_im(Image.open(img_path)))
        # can't compute gradients with uint8 :((
        # im = (255 * im).type(torch.uint8)  # to uint8 to save memory (vs float32)
        # return image (tensor), tuple of cartesian coords (x, y, z), and tuple of GPS & compass (lat, long, compass)
        return (
            im.to(device),
            torch.from_numpy(self.xyz_cartesian[idx]).type(torch.float32).to(device),
            torch.from_numpy(self.gps_compass[idx]).type(torch.float32).to(device),
        )

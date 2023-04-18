import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
from PIL import Image
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # no more nasty torch warnings


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_size = len(os.listdir(os.path.join(self.data_dir, "images")))
        self.xyz_cartesian = np.loadtxt(
            os.path.join(self.data_dir, "xyz_cartesian.txt")
        )
        self.gps_compass = np.loadtxt(os.path.join(self.data_dir, "gps_compass.txt"))
        assert len(self.xyz_cartesian) == len(self.gps_compass) >= self.dataset_size

    def __len__(self) -> int:
        return self.dataset_size

    # given an image index, return the image itself as well as its associated metadata
    def __getitem__(
        self, idx: int, view: int = 4
    ) -> Tuple[torch.Tensor, Tuple[float, float, float], Tuple[float, float, float]]:
        assert 0 <= idx < self.dataset_size
        assert 0 <= view <= 5
        img_path: str = os.path.join(self.data_dir, "images", f"{idx:06d}_{view}.jpg")
        if not os.path.exists(img_path):
            return None, None, None
        im: torch.Tensor = transforms.ToTensor()(Image.open(img_path))
        return (
            torch.Tensor(im),
            tuple(self.xyz_cartesian[idx]),
            tuple(self.gps_compass[idx]),
        )

import torch
import numpy as np
from typing import Tuple
from PIL import Image
import os

# get this for MacOS: https://discuss.pytorch.org/t/failed-to-load-image-python-extension-could-not-find-module/140278/8
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # no more nasty torch warnings
from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = None, res: float = 1):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_size = len(os.listdir(os.path.join(data_dir, "images")))
        self.xyz_cartesian = np.loadtxt(os.path.join(data_dir, "xyz_cartesian.txt"))
        self.gps_compass = np.loadtxt(os.path.join(data_dir, "gps_compass.txt"))
        assert len(self.xyz_cartesian) == len(self.gps_compass) >= self.dataset_size
        # scale factor for images (resolution scale)
        assert 0 < res <= 1
        # get the resolution for the images
        example_im: str = os.path.join(data_dir, "images", "000001_4.jpg")
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
    ) -> Tuple[torch.Tensor, Tuple[float, float, float], Tuple[float, float, float]]:
        assert 0 <= idx < self.dataset_size
        assert 0 <= view <= 5
        img_path: str = os.path.join(self.data_dir, "images", f"{idx:06d}_{view}.jpg")
        if not os.path.exists(img_path):
            return None, None, None
        im: torch.Tensor = self.to_tensor(self.resize_im(Image.open(img_path)))
        im = (255 * im).type(torch.uint8)  # to uint8 to save memory (vs float32)
        # return image (tensor), tuple of cartesian coords (x, y, z), and tuple of GPS & compass (lat, long, compass)
        return (im, tuple(self.xyz_cartesian[idx]), tuple(self.gps_compass[idx]))

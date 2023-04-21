## Begin main

import os
import torch

from model import GeoGuesser
from gsv_query import download_gsv
from pit_orl_manh import download_pom
from dataloader import ImageDataset
from utils import device


BIG_DATASET: bool = False  # flip this to use all available images from the dataset

# load data
dataset_dir: str = "dataset"
if not (
    os.path.exists(dataset_dir)
    and os.path.exists(os.path.join(dataset_dir, "xyz_cartesian.txt"))
    and os.path.exists(os.path.join(dataset_dir, "gps_compass.txt"))
):
    # whether or not we need to download the dataset
    download_pom(dataset_dir, num_images=-1 if BIG_DATASET else 100)

dataset = ImageDataset(dataset_dir, res=1)  # tune im_scale to change resolution scale
dataset.preload()  # load everything in memory to be super fast!
# examples
image, xyz, gps = dataset[0]  # happens to be all None!
image, xyz, gps = dataset[56]  # non-None example

torch.manual_seed(1)
idxs = torch.randperm(len(dataset))

# use 90% for training, 10% for test
test_train_split: int = int(0.9 * len(idxs))
train = idxs[:test_train_split]
test = idxs[test_train_split:]

m = GeoGuesser(dataset, image.shape, train, test)
m = m.to(device)
print(m)

pred, _ = m.forward(image)
print(pred)

m.load()  # try to load from the latest ckpt
m.begin_training()

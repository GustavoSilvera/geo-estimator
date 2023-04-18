import os
import torch

from model import GeoGuesser
from gsv_query import download_gsv
from pit_orl_manh import download_pom
from dataloader import ImageDataset


# load data
dataset_dir: str = "dataset"
if not (
    os.path.exists(dataset_dir)
    and os.path.exists(os.path.join(dataset_dir, "xyz_cartesian.txt"))
    and os.path.exists(os.path.join(dataset_dir, "gps_compass.txt"))
):
    # whether or not we need to download the dataset
    download_pom(dataset_dir, num_images=-1)  # begin downloading dataset

dataset = ImageDataset(dataset_dir)

# examples
image, xyz, gps = dataset[56]
image, xyz, gps = dataset[0]  # happens to be all None!

torch.manual_seed(1)
idxs = torch.randperm(len(dataset))

# use 90% for training, 10% for test
test_train_split: int = int(0.9 * len(idxs))
train = idxs[:test_train_split]
test = idxs[test_train_split:]

m = GeoGuesser(dataset, test, train)

print(m)

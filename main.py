## Begin main

import os
import torch

from model import GeoGuesser
from gsv_query import download_gsv
from pit_orl_manh import download_pom
from dataloader import ImageDataset
from utils import device

import argparse

BIS_DATASET: bool = False
TRAIN_MODEL: bool = False
PRELOAD_DATA: bool = False
LOAD_CKPT: bool = False

parser = argparse.ArgumentParser()
parser.add_argument("--big_data", action="store_true", default=BIS_DATASET)
parser.add_argument("--train", action="store_true", default=TRAIN_MODEL)
parser.add_argument("--preload", action="store_true", default=PRELOAD_DATA)
parser.add_argument("--ckpt", default=None)
parser.add_argument("--quality", default=1)
args = parser.parse_args()

print(f"Parameters: {args}")

# load data
dataset_dir: str = "dataset"
if not (
    os.path.exists(dataset_dir)
    and os.path.exists(os.path.join(dataset_dir, "xyz_cartesian.txt"))
    and os.path.exists(os.path.join(dataset_dir, "gps_compass.txt"))
):
    # whether or not we need to download the dataset
    download_pom(dataset_dir, num_images=-1 if args.big_data else 100)

dataset = ImageDataset(dataset_dir, res=args.quality)
if args.preload:
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

if args.ckpt is not None:
    m.load(int(args.ckpt))  # try to load from the latest ckpt
if args.train:
    m.begin_training()

while True:
    try:
        prompt: str = input("> ")
        if prompt[0] == "~":  # load new ckpt
            new_ckpt = int(prompt[1:])
            print(f"[Loading ckpt {new_ckpt}]")
            m.load(new_ckpt)
            continue
        image, xyz, gps = dataset[int(prompt)]
        if image is None or xyz is None or gps is None:
            print(f"Dataset does not contrain entry @ {prompt}")
            continue
        pred, loss = m.forward(image, gps)
        pred = pred.cpu().detach().numpy().tolist()
        gps = gps.cpu().detach().numpy().tolist()
        print(f"Prediction: {pred}")
        print(f"Actual: {gps}")
        print(f"Loss: {loss}")
    except KeyboardInterrupt:  # ctrl+c
        print("Goodbye!")
        break
    except EOFError:  # ctrl+D
        print("Goodbye!")
        break

import os

from model import GeoGuesser
from gsv_query import download_gsv
from pit_orl_manh import download_pom


# load data
dataset_dir: str = "dataset"
if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
    download_pom(dataset_dir)  # begin downloading dataset

m = GeoGuesser()

print(m)

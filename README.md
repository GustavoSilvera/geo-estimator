# geo-guesser-ai
## Final (mini) project for CMU 10-315
## Team: Gustavo S, Alan L, Lawrence C

Its an AI for geo-guesser... give it an input image location and it'll estimate where it thinks it is (latitude & longitude). This is a regression problem where the networks processes the input image using various convolutional layers, relu activations, and pooling until a final sequence of fully-connected layers to the desired format. 

Python requirements:
```python
torch
torchvision
PIL
numpy

# optional
google_streetview # if downloading dataset from google_streetview_query.py
jupytext          # if want to create a .ipynb for jupyter notebook
```

## Dataset
We had thought about using the [google-streetview developer API](https://pypi.org/project/google-streetview/) for downloading the dataset, but this proved to be too much hassle so instead we took a look at [this simpler](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) (and already organized) dataset from UCF. The downloading of this dataset is handled in []`pit_orl_manh.py`](pit_orl_manh.py) and pulls down their entire dataset in ~10 minutes thanks to parallelizing the `curl`/`wget` commands in batches. 

## Model Architecture:
TODO

```
State dict: ['network.0.network.0.weight', 'network.2.network.0.weight', 'network.6.weight', 'network.6.bias', 'network.9.weight', 'network.9.bias']
GeoGuesser(
  (network): Sequential(
    (0): ConvReluBlock(
      (network): Sequential(
        (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
    )
    (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (2): ConvReluBlock(
      (network): Sequential(
        (0): Conv2d(24, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ReLU(inplace=True)
      )
    )
    (3): MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=0, dilation=1, ceil_mode=False)
    (4): ReLU(inplace=True)
    (5): Flatten(start_dim=1, end_dim=-1)
    (6): Linear(in_features=23520, out_features=5000, bias=True)
    (7): Dropout(p=0.2, inplace=False)
    (8): ReLU(inplace=True)
    (9): Linear(in_features=5000, out_features=2, bias=True)
    (10): Tanh()
    (11): ScaleToLatLong()
  )
)
```

## Getting started
Begin by downloading the dataset of images/coordinates from either `gsv_query` or `pit_orl_manh` (recommended) and running the `main` file

```sh
python main.py --preload --train --big_data
# downloads dataset (takes ~10m)

# [args] --preload : used to load the entire dataset in memory, takes ~6G
# [args] --train : whether or not to train the model
# [args] --big_data : whether or not to download the entire dataset or only 100 samples (demo purposes)
# [args] --ckpt X: (Optional) load the model from a specific ckpt at ckpt/ckpt_{x}.pt
```

The image data is now located in `dataset/images/` with ~10000 images from the 4'th viewpoint and with associated XYZ and GPS metadata in `dataset/xyz_cartesian.txt` and `dataset/gps_compass.txt` respectively. This is what we will use for our primary dataset (images + metadata).

## To Google Colab (.ipynb)
For ease of working in local Python files but uploading a submission via .ipynb, there is a helper script (`to_colab.py`) that you can run:
```sh
python to_colab.py
# Reading input py file: "utils"
# Reading input py file: "gsv_query"
# Reading input py file: "pit_orl_manh"
# Reading input py file: "model"
# Reading input py file: "main"
# Output .ipynb file to geo-estimator.ipynb
```


## Acknowledgements
- *Image Geo-localization based on Multiple Nearest Neighbor Feature Matching using Generalized Graphs*. Amir Roshan Zamir and Mubarak Shah. **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, 2014.
    - For their excellent dataset provided [here](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) ([README](http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/Readme.pdf))
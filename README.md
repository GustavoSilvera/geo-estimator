# geo-guesser-ai
## Final (mini) project for 10-315
## Team: Lawrence C, Gustavo S, Alan L

Its an AI for geo-guesser... give it an input image location and it'll estimate where it thinks it is.

Python requirements:
```python
torch
numpy

# optional
google_streetview # if downloading dataset from google_streetview_query.py
jupytext          # if want to create a .ipynb for jupyter notebook
```

Using the [google-streetview developer API](https://pypi.org/project/google-streetview/) for downloading the dataset

## Getting started
Begin by downloading the dataset of images/coordinates from either `gsv_query` or `pit_orl_manh` (recommended) and running the `main` file

```sh
python main.py
# downloads dataset (takes ~10m)
```

The data is now located in `dataset/` with ~10000 images from the 4'th viewpoint and with associated GPS coordinates in `dataset/coords.txt`. This is what we will use for our primary dataset (images + labels).

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
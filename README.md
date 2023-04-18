# geo-guesser-ai
## Final (mini) project for 10-315
## Team: Lawrence C, Gustavo S, Alan L

Its an AI for geo-guesser... give it an input image location and it'll estimate where it thinks it is.

Python requirements:
```python
torch

# optional
google_streetview # if downloading dataset from google_streetview_query.py

jupytext          # if want to create a .ipynb for jupyter notebook
```

Using the [google-streetview developer API](https://pypi.org/project/google-streetview/) for downloading the dataset

## Getting started
Begin by downloading the dataset of images/coordinates from either `gsv_query` or 
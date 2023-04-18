### Function to download Google Street View images and metadata for use as a dataset of images and locations.


def download_gsv():
    import os
    import numpy as np
    import google_streetview.api

    data_dir: str = "data"
    os.makedirs(data_dir, exist_ok=True)

    np.random.seed(1)

    num_download: int = 1000

    # constraining random bounds to US (for now)
    lat_bounds = (-124.84, -66.88)
    lon_bounds = (31.37, 49.38)
    pitch_bounds = (-5, 5)

    # randomly selecting latitude, longitude, heading, and pitch
    random = np.random.random(size=(4, num_download))
    random[0] = random[0] * (max(lat_bounds) - min(lat_bounds)) + min(lat_bounds)
    random[1] = random[1] * (max(lon_bounds) - min(lon_bounds)) + min(lon_bounds)
    random[2] *= 360  # [0,1] -> degrees
    random[3] = random[3] * (max(pitch_bounds) - min(pitch_bounds)) + min(pitch_bounds)
    random = random.T

    key: str = None  # PUT YOUR KEY HERE
    if key is None:
        key = input("Google Maps Dev Key: ")
    params = [
        {
            "size": "512x512",
            "location": f"{longitude:.3f},{latitude:.3f}",
            "heading": f"{heading:.3f}",
            "pitch": f"{pitch:.3f}",
            "key": "PUT YOUR KEY HERE",
        }
        for (latitude, longitude, heading, pitch) in random
    ]

    # see https://pypi.org/project/google-streetview/
    # and https://rrwen.github.io/google_streetview/
    print(f"Downloading {num_download} samples using google streeview API")
    results = google_streetview.api.results(params)
    results.preview()
    results.download_links(data_dir)

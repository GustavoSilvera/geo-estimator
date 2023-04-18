def download_pom(view: int = 4):
    # view is either 1 (rear), 2 (right), 3 (left), 4(front), or 5(up)
    assert 1 <= view <= 5

    import os

    # using PitOrlManh dataset from:
    # https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset
    data_dir: str = "dataset-jpg"
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)

    num_download: int = 1000
    # use simple single-threaded downloader like curl or wget
    for i in range(num_download):
        url: str = f"http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/images/{i:06d}_{view}.jpg"
        os.system(f"wget {url}")
        print(f"Finished downloading image {i}")

    coords: str = "http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/#:~:text=Cartesian_Location_Coordinates.mat"
    os.system(f"wget {coords}")  # download metadata for coordinates

    os.chdir("..")  # back to main dir

    print(f'Downloads successful in "{data_dir}"')


def download_pom_raw():
    import os

    # using PitOrlManh dataset from:
    # https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset
    data_dir: str = "dataset"
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)
    urls = [
        f"http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/zipped%20images/part{i+1}.zip"
        for i in range(10)
    ]

    use_aria2c: bool = os.system("which aria2c") == 0  # no error code!

    if use_aria2c:
        print("Using aria2c to download dataset fast!")
        # if aria2c is available, use it
        aria_cmd = f"aria2c -Z"
        for url in urls:
            aria_cmd += f" {url}"
        os.system(aria_cmd)  # runs much faster (in parallal)
    else:
        # use simple single-threaded downloader like curl or wget
        downloader: str = "wget"
        downloader: str = "curl"
        for url in urls:
            os.system(f"{downloader} {url}")

    for i in range(10):
        os.system(f"unzip part{i+1}.zip")

    os.chdir("..")  # back to main dir

    print(f'Downloads successful in "{data_dir}"')

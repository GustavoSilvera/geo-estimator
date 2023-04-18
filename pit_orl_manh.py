### Function to download individual files from the PitOrlManh dataset from a particular view


def download_pom(data_dir: str, view: int = 4):
    # view is either 1 (rear), 2 (right), 3 (left), 4(front), or 5(up)
    assert 1 <= view <= 5

    import os

    # using PitOrlManh dataset from:
    # https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)

    num_download: int = 10
    # use simple single-threaded downloader like curl or wget
    for i in range(num_download):
        url: str = f"http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/images/{i:06d}_{view}.jpg"
        os.system(f"wget {url}")
        print(f"Finished downloading image {i}")

    import scipy
    import numpy as np

    coord_file: str = "Cartesian_Location_Coordinates.mat"
    coords: str = (
        f"http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh/{coord_file}"
    )
    os.system(f"wget {coords}")  # download metadata for coordinates
    mat = scipy.io.loadmat(coord_file)
    np.savetxt("coords.txt", mat["XYZ_Cartesian"])  # indices match with image index
    os.system(f"rm {coord_file}")  # gross matlab file ew

    os.chdir("..")  # back to main dir

    print(f'Downloads successful in "{data_dir}"')


### Function to download the entire PitOrlManh dataset (~50Gb)


def download_pom_raw(data_dir: str):
    import os

    # using PitOrlManh dataset from:
    # https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset
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

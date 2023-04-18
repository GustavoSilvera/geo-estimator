### Function to download individual files from the PitOrlManh dataset from a particular view


def download_pom(data_dir: str, view: int = 4, num_images: int = -1):
    # view is either 1 (rear), 2 (right), 3 (left), 4(front), or 5(up)
    assert 1 <= view <= 5

    import os, subprocess

    # using PitOrlManh dataset from:
    # https://www.crcv.ucf.edu/data/GMCP_Geolocalization/#Dataset
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)
    img_dir: str = "images"
    os.makedirs(img_dir, exist_ok=True)  # for downloading the images
    os.chdir(img_dir)

    max_dataset: int = 0  # total dataset is approx 10343
    num_download: int = num_images if num_images > 0 else max_dataset
    num_download = min(num_download, max_dataset)

    # use simple single-threaded downloader like curl or wget
    statuses = []
    batch: int = 100  # number of parallel downloads at once

    dataset_home: str = "http://www.cs.ucf.edu/~aroshan/index_files/Dataset_PitOrlManh"

    with open(os.devnull, "wb") as devnull:
        cumulative = 0
        for i in range(num_download // batch):
            for b in range(batch):
                idx = i * batch + b + 1
                url: str = f"{dataset_home}/images/{idx:06d}_{view}.jpg"
                statuses.append(
                    subprocess.Popen(
                        ["wget", url], stdout=devnull, stderr=subprocess.STDOUT
                    )
                )
                # print(f"Starting image {idx}", end="\r", flush=True)
            for s in statuses:
                s.communicate()
                cumulative += 1
                print(
                    f"Completed download: {cumulative}/{num_download} ({100 * cumulative / num_download:.2f}%)",
                    end="\r",
                    flush=True,
                )
            statuses = []  # reset
    print()
    os.system("rm wget-log.*")  # cleanup
    os.chdir("..")  # back to data_dir

    import scipy
    import numpy as np

    # download cartesian location (X, Y, Z) locations
    def download_mat(coord_file: str) -> np.ndarray:
        coords: str = f"{dataset_home}/{coord_file}"
        os.system(f"wget {coords}")  # download metadata for coordinates
        mat = scipy.io.loadmat(coord_file)
        os.system(f"rm {coord_file}")  # gross matlab file ew
        return mat

    # X, Y, Z locations for each of the images
    XYZ = download_mat("Cartesian_Location_Coordinates.mat")
    np.savetxt("xyz_cartesian.txt", XYZ["XYZ_Cartesian"])
    # latitude, longitude, and compass of each of the images
    LLC = download_mat("GPS_Long_Lat_Compass.mat")
    np.savetxt("gps_compass.txt", LLC["GPS_Compass"])

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

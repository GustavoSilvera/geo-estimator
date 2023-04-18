def download_pom():
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

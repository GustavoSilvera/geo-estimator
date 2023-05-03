### Function to download Google Street View images and metadata for use as a dataset of images and locations.

def download_cities(data_dir: str):
    import os
    import numpy as np
    import google_streetview.api
    
    # city: (min long, min lat, max long, max lat)
    city_bounding_boxes = {
        "reykjavik": (-21.955261,64.127984,-21.879559,64.148863),
        "madrid": (-3.759613,40.375335,-3.612671,40.471982),
        "lisbon": (-9.242935,38.712352,-9.122772,38.778452),
        "paris": (2.263641,48.815099,2.431183,48.900432),
        "london": (-0.165253,51.473872,-0.032387,51.537089),
        "berlin": (13.329391,52.480629,13.472214,52.557046),
        "rome": (12.428970,41.849854,12.560806,41.936185),
        "stockholm": (18.011284,59.299835,18.120804,59.348850),
        "oslo": (10.712013,59.906915,10.783081,59.928500),
        "istanbul": (28.932610,41.003164,28.981705,41.022841),
        "copenhagen": (12.529907,55.644757,12.638397,55.712083),
        "helsinki": (24.907265,60.151972,24.972496,60.195319),
        "bucharest": (26.053505,44.397874,26.149979,44.462071),
        "moscow": (37.502518,55.682255,37.738724,55.823215),
        "melbourne": (144.755859,-37.976435,145.207672,-37.668538),
        "sydney": (151.032486,-33.948339,151.282425,-33.827569),
        "perth": (115.752869,-32.116140,116.002808,-31.826230),
        "auckland": (174.682617,-36.923890,174.866638,-36.858029),
        "mumbai": (72.816925,19.051430,72.954254,19.221290),
        "dubai": (55.269585,25.226330,55.405540,25.281878),
        "tokyo": (139.664154,35.659606,139.831696,35.737628),
        "seoul": (126.918640,37.478449,127.107468,37.600881),
        "bangkok": (100.350952,13.599937,100.743713,13.885241),
        "singapore": (103.703384,1.303237,103.910065,1.417121),
        "jakarta": (106.725082,-6.261335,106.940002,-6.124887),
        "manila": (120.971603,14.571562,121.052971,14.642285),
        "taipei": (121.459579,24.961080,121.614761,25.097254),
        "cape town": (18.405304,-34.066685,18.638077,-33.921600),
        "mombasa": (39.652748,-4.067258,39.674377,-4.037139),
        "accra": (-0.257034,5.544049,-0.188370,5.586737),
        "nairobi": (36.773186,-1.316832,36.862450,-1.252341),
        "dakar": (-17.495155,14.663254,-17.426491,14.748880),
        "rio": (-43.292313,-22.925846,-43.220558,-22.877487),
        "lima": (-77.073212,-12.121825,-76.977425,-12.020494),
        "mexico city": (-99.242706,19.311139,-99.025726,19.533773),
        "bogota": (-74.192047,4.516599,-74.026566,4.760102),
        "buenos aires": (-58.536072,-34.666281,-58.376083,-34.574229),
        "toronto": (-79.485512,43.645653,-79.355049,43.701492),
        "quebec": (-71.385498,46.745096,-71.147919,46.858775),
        "vancouver": (-123.177795,49.096779,-122.745209,49.270818),
        "new york": (-74.048309,40.694273,-73.857422,40.856411),
        "pittsburgh": (-80.012398,40.426326,-79.916267,40.471759),
        "san francisco": (-122.495270,37.627431,-122.386093,37.810908),
        "houston": (-95.596161,29.583284,-95.123749,29.925255)
    }

    os.makedirs(data_dir, exist_ok=True)

    np.random.seed(1)

    num_download: int = 300

    for (city, bbox) in city_bounding_boxes.items():
        lat_bounds = (bbox[1], bbox[3])
        lon_bounds = (bbox[0], bbox[2])
        pitch_bounds = (-5, 5)

        # randomly selecting latitude, longitude, heading, and pitch
        random = np.random.random(size=(4, num_download))
        random[0] = random[0] * (max(lat_bounds) - min(lat_bounds)) + min(lat_bounds)
        random[1] = random[1] * (max(lon_bounds) - min(lon_bounds)) + min(lon_bounds)
        random[2] *= 360  # [0,1] -> degrees
        random[3] = random[3] * (max(pitch_bounds) - min(pitch_bounds)) + min(pitch_bounds)
        random = random.T

        key: str = "AIzaSyCqgpflulg7J0TJq2zsdanmKMXqFU-3zK0"  # PUT YOUR KEY HERE
        if key is None:
            key = input("Google Maps Dev Key: ")
        params = [
            {
                "size": "512x512",
                "location": f"{latitude:.3f},{longitude:.3f}",
                "heading": f"{heading:.3f}",
                "pitch": f"{pitch:.3f}",
                "key": key,
            }
            for (latitude, longitude, heading, pitch) in random
        ]

        # see https://pypi.org/project/google-streetview/
        # and https://rrwen.github.io/google_streetview/
        print(f"Downloading {num_download} samples for {city}")
        results = google_streetview.api.results(params)
        # results.preview()
        results.download_links(f"{data_dir}/{city}")

# Formats output from download_cities in the same way as pit_orl_manh dataset
# XXXXXX_4.jpg for images
# gps_compass.txt:
#  lat0 long0 compass
#  lat1 long1 compass
#  ...
# We do not use compass data. But for backwards-compatibility reasons, we will make them all 0's.
def process_cities_dataset(input_data_dir: str, output_data_dir: str):
    import os
    import glob
    import json
    import shutil
    
    # Read in all the images and their coordinates
    all_images = []
    all_coords = []
    city_names = os.listdir(input_data_dir)
    for city in city_names:
        with open(os.path.join(input_data_dir, city, 'metadata.json')) as f:
            json_data = json.load(f)
            for item in json_data:
                if item['status'] == 'OK':
                    image_file = item['_file']
                    lat = item['location']['lat']
                    long = item['location']['lng']
                    all_images.append(os.path.join(input_data_dir, city, image_file))
                    all_coords.append((lat, long))
    
    # Copy them over to new directory
    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)
    if not os.path.exists(os.path.join(output_data_dir, "images")):
        os.mkdir(os.path.join(output_data_dir, "images"))
    for (i, image) in enumerate(all_images):
        shutil.copyfile(image, os.path.join(output_data_dir, "images", f"{i:06d}_4.jpg"))
    with open(os.path.join(output_data_dir, "gps_compass.txt"), 'w') as f:
        f.write('\n'.join([f"{lat} {lng} 0" for (lat, lng) in all_coords]))

#download_cities("cities_dataset_raw")
process_cities_dataset("cities_dataset_raw", "cities_dataset")
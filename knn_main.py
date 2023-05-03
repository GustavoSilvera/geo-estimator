import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
import glob

dataset_dir: str = "cities_dataset"
gps_compass = np.loadtxt(os.path.join(dataset_dir, "gps_compass.txt"))

def get_data(idx: int):
    preprocess = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])
    img_path = os.path.join(dataset_dir, "images", f"{idx:06d}_4.jpg")
    
    if not os.path.exists(img_path):
        return None, None
    
    img = torch.permute(preprocess(Image.open(img_path)), (1,2,0)).numpy()
    latlng = gps_compass[idx,:2]
    
    return (img, latlng)

X = []
y = []
dataset_size = len(glob.glob(os.path.join(dataset_dir, "images", "*.jpg")))
for i in range(dataset_size):
    img, latlng = get_data(i)
    if img is None:
        continue
    img = img.flatten()
    X.append(img)
    y.append(latlng)

N = len(X)

X_train = X[:int(0.9*N)]
y_train = y[:int(0.9*N)]
X_test = X[int(0.9*N):]
y_test = y[int(0.9*N):]

neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(X_train, y_train)
y_hat = neigh.predict(X_test)

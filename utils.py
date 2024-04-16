import os
import numpy as np
import json
import torch
import math
from torch.utils.data import Dataset
from skimage import io
import cv2
from matplotlib import pyplot as plt

import os

class NeRFDatasetLoader(Dataset):
    """
    NeRF Dataset loader class
    tutorial - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self, dataset_path, mode):
        """
        dataset_path - path to the dataset, 
            assumes the following folder structure
            .
            ├── test
            ├── train
            │   ├── r_0.png
            │   ├── r_1.png
            │   ├── r_10.png
            ├── transforms_test.json
            ├── transforms_train.json
            ├── transforms_val.json
            └── val
        mode - train/test/val
        """
        self.root_dir = dataset_path
        if mode == "test":
            file = "transforms_test.json"
        elif mode == "train":
            file = "transforms_train.json"
        elif mode == "val":
            file = "transforms_val.json"

        transforms_path = os.path.join(dataset_path + os.sep + file)
        print(f"Loading {transforms_path} for {mode}")
        with open(transforms_path) as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data["frames"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        img_file = self.data["frames"][idx]["file_path"] + ".png"
        img_name = os.path.join(self.root_dir + os.sep + img_file)
        image = cv2.imread(img_name)

        image = cv2.resize(image, (100,100), interpolation=cv2.INTER_AREA)

        # get transforms of that image
        transforms = self.data["frames"][idx]["transform_matrix"]
        transforms = torch.tensor(transforms)

        # get focal length
        # https://github.com/NVlabs/instant-ngp/issues/332
        camera_angle_x = self.data["camera_angle_x"]
        focal_length = 0.5*image.shape[0] / math.tan(0.5 * camera_angle_x) # TODO need to verify math

        return {'focal_length': focal_length, 'image': image, 'transforms': transforms}

    def get_full_data(self, device):
        """
        inputs:
            path
        outputs:
            focal_length - 1,
            transformations - List[N; 4 x 4]
            images - List[N; W x H]
        """
        transforms = []
        images = []
        focal_length = self[0]['focal_length']
        for i in range(len(self.data["frames"])):
            data = self[i]
            transforms.append(torch.tensor(data["transforms"]))
            images.append(torch.tensor(data["image"]))

        return torch.tensor(focal_length).to(device), torch.stack(transforms).to(device), torch.stack(images).to(device)
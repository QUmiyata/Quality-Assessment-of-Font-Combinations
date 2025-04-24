import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx]).astype(np.float32)
        img = torch.from_numpy(img/255)
        return img
"""
Data utils
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
import os

class RawImageDataset(Dataset):
    def __init__(self, img_dir: str, transform=None, device=None):
        self.img_filenames = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        if device is not None:
            print(f'Reading dataset into {device} memory...', end=' ', flush=True)
            self.data = [x.to(device) for x in self]
            print('done')


    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            image = self.data[idx]
        else:
            img_path = os.path.join(self.img_dir, self.img_filenames[idx])
            image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image



class AnimeFaces(RawImageDataset):
    """Wrapper for https://www.kaggle.com/splcher/animefacedataset"""
    img_shape = (3, 64, 64)

    def __init__(self, img_dir: str, **kwargs):
        super().__init__(img_dir, transform=transforms.Compose([
            transforms.CenterCrop(self.img_shape[1:]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(0, 1),
        ]), **kwargs)




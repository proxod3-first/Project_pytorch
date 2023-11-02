# load_data


import torch
from torch.utils.data import *
import torchvision
import torchvision.transforms as transforms
import lightning as L


class LightningDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.batch_size = 4
        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=2)
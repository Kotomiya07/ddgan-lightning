import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lmdb_datasets import LMDBDataset
from datasets_prep.lsun import LSUN
from datasets_prep.stackmnist_data import StackedMNIST, _data_transforms_stacked_mnist

class DDGANDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_name = args.dataset
        self.batch_size = args.batch_size
        self.num_workers = 4
        self.image_size = args.image_size
        
    def setup(self, stage=None):
        if self.dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.train_dataset = CIFAR10('./data', train=True, transform=transform, download=True)
            self.val_dataset = CIFAR10('./data', train=False, transform=transform, download=True)

        elif self.dataset_name == 'stackmnist':
            train_transform, valid_transform = _data_transforms_stacked_mnist()
            self.train_dataset = StackedMNIST(root='./data', train=True, download=True, transform=train_transform)
            self.val_dataset = StackedMNIST(root='./data', train=False, download=True, transform=valid_transform)

        elif self.dataset_name == 'lsun':
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            train_data = LSUN(root='/datasets/LSUN/', classes=['church_outdoor_train'], transform=transform)
            subset = list(range(0, 120000))
            self.train_dataset = torch.utils.data.Subset(train_data, subset)
            # LSUNはテストセットが提供されていないため、訓練データの一部を検証用として使用
            val_size = int(len(self.train_dataset) * 0.1)
            train_size = len(self.train_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        elif self.dataset_name == 'celeba_256':
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.train_dataset = LMDBDataset(root='/datasets/celeba-lmdb/', name='celeba', train=True, transform=transform)
            self.val_dataset = LMDBDataset(root='/datasets/celeba-lmdb/', name='celeba', train=False, transform=transform)
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

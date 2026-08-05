import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class ImagesOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self._data = base_dataset

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        x, _ = self._data[idx]
        return x


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str = "datasets/", batch_size=64, num_workers=0, seed=42, images_only=False
    ):
        super().__init__()
        self._data_dir = data_dir
        self._transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self._num_workers = num_workers
        self._seed = seed
        self._batch_size = batch_size
        self._images_only = images_only

    def prepare_data(self):
        CIFAR10(self._data_dir, train=True, download=True)
        CIFAR10(self._data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            cifar10 = CIFAR10(self._data_dir, train=True, transform=self._transform)
            self._train_set, self._val_set = random_split(
                ImagesOnlyDataset(cifar10) if self._images_only else cifar10,
                [0.9, 0.1],
                torch.Generator().manual_seed(self._seed),
            )

        if stage == "test":
            test_set = CIFAR10(self.data_dir, train=False, transform=self.transform)
            self._test_set = ImagesOnlyDataset(test_set)

    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_set,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=True,
        )

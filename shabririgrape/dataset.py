import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset

from . import preprocess as aug


class BCIDataContainer:
    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray):
        """
        BCI data container.

        Args:
            train_x (np.ndarray): training input data x
            train_y (np.ndarray): training data label y
            test_x (np.ndarray): testing input data x
            test_y (np.ndarray): testing data label y
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y


class AugmentedDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0], 'invalid dataset'
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        x = aug.jitter(x)
        x = aug.scaling(x)
        x = aug.random_flip(x)
        x = aug.random_shift(x)
        x = aug.random_window_slice(x)
        return tensor(x, dtype=torch.float), tensor(self.y[idx], dtype=torch.long)


def gen_loader(data_container: BCIDataContainer, batch_size: int = 64, use_aug: bool = False) -> DataLoader:
    """
    Generate data loader.

    Args:
        data_container (BCIDataContainer): data container
        batch_size (int, optional): batch size. Defaults to 64.
        use_aug (bool, optional): use augmentation. Defaults to False.

    Returns:
        DataLoader: data loader
    """
    if use_aug:
        train_dataset = AugmentedDataset(
            tensor(data_container.train_x, dtype=torch.float),
            tensor(data_container.train_y, dtype=torch.long)
        )
    else:
        train_dataset = TensorDataset(
            tensor(data_container.train_x, dtype=torch.float),
            tensor(data_container.train_y, dtype=torch.long)
        )
    test_dataset = TensorDataset(
        tensor(data_container.test_x, dtype=torch.float),
        tensor(data_container.test_y, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, test_loader

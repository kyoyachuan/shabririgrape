import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler

from . import DATASET, MODE


class DRContainer:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Diabetic retinopathy data container.

        Args:
            x (np.ndarray): input data x filenames.
            y (np.ndarray): data label y.
        """
        self.x = x
        self.y = y
        self._prepend_directory()

    def _prepend_directory(self):
        """
        Prepend directory to path.
        """
        self.x = np.array([os.path.join(DATASET.IMGS, f'{path}{DATASET.EXT}') for path in self.x])


def get_data(mode: str) -> DRContainer:
    """
    Get data from csv file.

    Args:
        mode (str): train or test.

    Raises:
        ValueError: If mode is not train or test.

    Returns:
        DRContainer: Data from csv file.
    """
    if mode == MODE.TRAIN:
        img = pd.read_csv(DATASET.TRAIN_IMG)
        label = pd.read_csv(DATASET.TRAIN_LABEL)
    elif mode == MODE.TEST:
        img = pd.read_csv(DATASET.TEST_IMG)
        label = pd.read_csv(DATASET.TEST_LABEL)
    else:
        raise ValueError(f'Should be either {MODE.TRAIN} or {MODE.TEST}')
    return DRContainer(
        np.squeeze(img.values),
        np.squeeze(label.values)
    )


class RetinopathyDataset(Dataset):
    def __init__(self, data_container: DRContainer, mode: str, use_aug: bool = False):
        """
        Initialize the dataset.

        Args:
            data_container (DBContainer): Data container.
            mode (str): train or test.
            use_aug (bool): Use augmentation or not. Default is False.
        """
        self.img_name = data_container.x
        self.label = tensor(data_container.y, dtype=torch.long)
        self.mode = mode
        if use_aug and mode == MODE.TRAIN:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(512),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_name)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of image and label.
        """
        img_path = self.img_name[index]
        label = self.label[index]
        img = self._load_and_preprocess_image(img_path)

        img_t = self.transform(img)

        return img_t, label

    def _load_and_preprocess_image(self, img_path) -> tensor:
        """
        Load and preprocess image.
        1. Load image by using PIL.
        2. Convert the pixel value to [0, 1]
        3. Transpose the image shape from [H, W, C] to [C, H, W]

        Args:
            img_path (str): Image path.

        Returns:
            tensor: Preprocessed image.
        """
        img = np.asarray(Image.open(img_path))
        img_t = tensor(img)
        img_t = img_t.type(torch.FloatTensor)
        img_t /= 255.0
        img_t = torch.transpose(img_t, 0, 2)
        return img_t

    def get_labels(self) -> torch.Tensor:
        """
        Get labels.

        Returns:
            torch.Tensor: Labels.
        """
        return self.label


def gen_loader(
    data_container: DRContainer,
    batch_size: int = 64,
    mode: str = MODE.TRAIN,
    use_aug: bool = False,
    use_imbalanced_sampler: bool = False
) -> DataLoader:
    """
    Generate data loader.

    Args:
        data_container (DRContainer): data container
        batch_size (int, optional): batch size. Defaults to 64.
        mode (str, optional): train or test. Defaults to MODE.TRAIN.
        use_aug (bool, optional): use augmentation. Defaults to False.
        use_imbalanced_sampler (bool, optional): use imbalanced sampler. Defaults to False.

    Returns:
        DataLoader: data loader
    """
    dataset = RetinopathyDataset(data_container, mode, use_aug)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if use_imbalanced_sampler or mode == MODE.TEST else True,
        sampler=None if not use_imbalanced_sampler else ImbalancedDatasetSampler(dataset)
    )

    return dataloader

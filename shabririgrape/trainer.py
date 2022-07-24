import numpy as np
from numpy import ndarray
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from .model import get_model
from .utils import convert_to_soft_label

from . import DATASET, MODEL_TYPE, DEVICE_TYPE


def get_device() -> torch.device:
    """
    Get the device.

    Returns:
        torch.device: device
    """
    device = DEVICE_TYPE.CUDA if torch.cuda.is_available() else DEVICE_TYPE.CPU
    print("Using {} device".format(device))
    return device


def get_loss(loss: str) -> nn.Module:
    """
    Get the loss function.

    Args:
        loss (str): string of loss function

    Raises:
        ValueError: We only allow the following losses for experiment:
            - 'cross_entropy'

    Returns:
        nn.Module: loss function
    """
    if loss == MODEL_TYPE.CROSS_ENTROPY:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Not included loss {loss}')


def get_optimizer(optimizer: str, model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """
    Get the optimizer.

    Args:
        optimizer (str): string of optimizer
        model (nn.Module): model
        lr (float): learning rate
        weight_decay (float, optional): weight decay. Defaults to 0.0.

    Raises:
        ValueError: We only allow the following optimizers for experiment:
            - 'adam'
            - 'sgd'

    Returns:
        nn.optim.Optimizer: optimizer
    """
    if optimizer == MODEL_TYPE.ADAM:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == MODEL_TYPE.SGD:
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Not included optimizer {optimizer}')


class LossAndAccCollector:
    def __init__(self, device: torch.device):
        """
        Initialize the loss and acc collector.

        Args:
            device (torch.device): device
        """
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        self.empty_test_data(device)

    def empty_test_data(self, device: torch.device):
        """
        Empty the test data
        """
        self.test_output = torch.tensor([]).to(device)
        self.test_target = torch.tensor([]).to(device)

    def add_test_data(self, output: torch.Tensor, target: torch.Tensor):
        """
        Add test data to the collector.

        Args:
            output (torch.Tensor): output
            target (torch.Tensor): target
        """
        self.test_output = torch.cat((self.test_output, output))
        self.test_target = torch.cat((self.test_target, target))

    def add(self, loss: float, accuracy: float):
        """
        Add loss and accuracy to the collector.

        Args:
            loss (float): loss
            accuracy (float): accuracy
        """
        self.train_loss.append(loss)
        self.train_acc.append(accuracy)

    def add_test(self, loss: float, accuracy: float):
        """
        Add loss and accuracy to the collector.

        Args:
            loss (float): loss
            accuracy (float): accuracy
        """
        self.test_loss.append(loss)
        self.test_acc.append(accuracy)


class Trainer:
    def __init__(
        self,
        model_name: str,
        pretrained_mode: str,
        lr: float,
        optimizer: str = MODEL_TYPE.ADAM,
        loss: str = MODEL_TYPE.CROSS_ENTROPY,
        weight_decay: float = 0.0,
        use_soft_label: bool = False,
        **kwargs
    ):
        """
        Initialize the trainer.

        Args:
            model_name (str): string of model name
            pretrained_mode (str): string of pretrained mode
            lr (float): learning rate
            optimizer (str): string of optimizer. Defaults to 'adam'.
            loss (str): string of loss function. Defaults to 'cross_entropy'.
            weight_decay (float, optional): weight decay. Defaults to 0.0.
            use_soft_label (bool, optional): use soft label for ordinal regression. Defaults to False.
            **kwargs: keyword arguments
        """
        self.model_name = model_name
        self.pretrained_mode = pretrained_mode
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_soft_label = use_soft_label

        self.device = get_device()
        self.collector = LossAndAccCollector(self.device)
        self.model = get_model(model_name, pretrained_mode).to(self.device)
        self.loss_fn = get_loss(loss)
        self.optimizer = get_optimizer(optimizer, self.model, lr, weight_decay)

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int = 300):
        """
        Train the model.

        Args:
            train_loader (torch.utils.data.DataLoader): train loader
            val_loader (torch.utils.data.DataLoader): validation loader
            epochs (int, optional): number of epochs. Defaults to 300.
        """
        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch+1)
            self.validate_epoch(val_loader)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int):
        """
        Train the model for one epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): train loader
            epoch (int): epoch
        """
        self.model.train()
        train_loss = 0
        train_accuracy = 0
        with tqdm(train_loader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch}')
            for data, target in tepoch:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.compute_loss(output, target)
                accuracy = self.compute_accuracy(output, target)
                loss.backward()
                self.optimizer.step()
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)

                train_loss += loss.item()
                train_accuracy += accuracy

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        self.collector.add(train_loss, train_accuracy)
        print('Training set: Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(
            train_loss, 100 * train_accuracy))

    def validate_epoch(self, val_loader: torch.utils.data.DataLoader):
        """
        Validate the model for one epoch.

        Args:
            val_loader (torch.utils.data.DataLoader): validation loader
        """
        self.model.eval()
        val_loss = 0
        val_accuracy = 0
        self.collector.empty_test_data(self.device)
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.compute_loss(output, target).item()
                self.collector.add_test_data(output, target)
        val_loss /= len(val_loader)
        val_accuracy = self.compute_accuracy(self.collector.test_output, self.collector.test_target)
        self.collector.add_test(val_loss, val_accuracy)
        print('Testing set: Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(
            val_loss, 100 * val_accuracy))

    def compute_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the accuracy.

        Args:
            output (torch.Tensor): output
            target (torch.Tensor): target

        Returns:
            float: accuracy
        """
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct / len(target)

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute the loss.

        Args:
            output (torch.Tensor): output
            target (torch.Tensor): target

        Returns:
            float: loss
        """
        if self.use_soft_label:
            soft_target = convert_to_soft_label(target)
            loss = self.loss_fn(output, soft_target)
        else:
            loss = self.loss_fn(output, target)
        return loss

    def compute_confusion_matrix(self, output: torch.Tensor, target: torch.Tensor) -> ndarray:
        """
        Compute the confusion matrix.

        Args:
            output (torch.Tensor): output
            target (torch.Tensor): target

        Returns:
            ndarray: confusion matrix
        """
        pred = output.argmax(dim=1, keepdim=True).cpu().numpy()
        gt = target.cpu().numpy()
        cm = confusion_matrix(gt, pred, labels=np.arange(DATASET.NUM_CLASSES), normalize='true')
        return cm

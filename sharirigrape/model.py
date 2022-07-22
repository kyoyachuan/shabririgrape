from torch import nn, tensor
from torchsummary import summary


def get_model(model_name: str, activation: str, **kwargs) -> nn.Module:
    """
    Get the model.

    Args:
        model_name (str): string of model name
        activation (str): string of activation function
        **kwargs: keyword arguments

    Raises:
        ValueError: We only allow the following models for experiment:
            - 'EEGNet'
            - 'DeepConvNet'

    Returns:
        nn.Module: model
    """
    if model_name == 'EEGNet':
        model = EEGNet(activation=activation, **kwargs)
    elif model_name == 'DeepConvNet':
        model = DeepConvNet(activation=activation, **kwargs)
    else:
        raise ValueError(f'Not included model {model_name}')

    return model


def get_activation(activation: str) -> nn.Module:
    """
    Get the activation function.

    Args:
        activation (str): string of activation function

    Raises:
        ValueError: We only allow the following activations for experiment:
            - 'relu'
            - 'elu'
            - 'leaky_relu'

    Returns:
        nn.Module: activation function
    """
    if activation == 'elu':
        return nn.ELU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f'Not included activation {activation}')


class ExtendedModule(nn.Module):
    def summary(self, input_size: tuple, device='cpu') -> str:
        """
        Get the summary of the model.

        Args:
            input_size (tuple): input size of the model
            device (str, optional): device. Defaults to 'cpu'.

        Returns:
            str: summary of the model
        """
        return summary(self, input_size=input_size, device=device)


class EEGNet(ExtendedModule):
    def __init__(
        self,
        dropout_prob: float = 0.5,
        temporal_filter_size_1: int = 16,
        temporal_filter_size_2: int = 32,
        spatial_filter_depth: int = 2,
        activation: str = 'elu'
    ):
        """
        EEGNet

        Args:
            dropout_prob (float, optional): dropout probabilty. Defaults to 0.5.
            temporal_filter_size_1 (int, optional): temporal filter size in depthwise convolution. Defaults to 16.
            temporal_filter_size_2 (int, optional): temporal filter size in separable convolution. Defaults to 32.
            spatial_filter_depth (int, optional): depth of spatial filter. Defaults to 2.
            activation (str, optional): activation function use in hidden layer. Defaults to 'elu'.
        """
        super(EEGNet, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, temporal_filter_size_1, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(temporal_filter_size_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                temporal_filter_size_1, spatial_filter_depth * temporal_filter_size_1, kernel_size=(2, 1),
                stride=(1, 1),
                groups=temporal_filter_size_1, bias=False
            ),
            nn.BatchNorm2d(
                spatial_filter_depth * temporal_filter_size_1, eps=1e-05, momentum=0.1, affine=True,
                track_running_stats=True
            ),
            get_activation(activation),
            nn.AvgPool2d(
                kernel_size=(1, 4),
                stride=(1, 4),
                padding=0
            ),
            nn.Dropout(p=dropout_prob)
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                spatial_filter_depth * temporal_filter_size_1, spatial_filter_depth * temporal_filter_size_1,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                groups=spatial_filter_depth * temporal_filter_size_1, bias=False
            ),
            nn.Conv2d(
                spatial_filter_depth * temporal_filter_size_1, temporal_filter_size_2, kernel_size=(1, 1),
                stride=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(temporal_filter_size_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(activation),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0
            ),
            nn.Dropout(p=dropout_prob),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(temporal_filter_size_2 * 23, 2, bias=True)
        )

    def forward(self, x: tensor) -> tensor:
        """
        Forward pass of the model.

        Args:
            x (tensor): input data

        Returns:
            tensor: output data
        """
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.classifier(x)
        return x


class DeepConvNet(ExtendedModule):
    def __init__(
        self,
        channel_list: list = [25, 50, 100, 200],
        dropout_prob: float = 0.5,
        activation: str = 'elu'
    ):
        """
        DeepConvNet

        Args:
            channel_list (list, optional): channel list. Defaults to [25, 50, 100, 200].
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            activation (str, optional): activation function use in hidden layer. Defaults to 'elu'.
        """
        super(DeepConvNet, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, channel_list[0], kernel_size=(1, 5)),
            nn.Conv2d(channel_list[0], channel_list[0], kernel_size=(2, 1)),
            nn.BatchNorm2d(channel_list[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_prob)
        )
        self.second_conv = self._conv_block(channel_list[0], channel_list[1], dropout_prob, activation)
        self.third_conv = self._conv_block(channel_list[1], channel_list[2], dropout_prob, activation)
        self.fourth_conv = self._conv_block(channel_list[2], channel_list[3], dropout_prob, activation)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8600, 2, bias=True)
        )

    def _conv_block(
            self, in_channels: int, out_channels: int, dropout_prob: float = 0.5, activation: str = 'elu'
    ) -> nn.Sequential:
        """
        Convolutional block.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            dropout_prob (float, optional): dropout probability. Defaults to 0.5.
            activation (str, optional): activation function in hidden layer. Defaults to 'elu'.

        Returns:
            nn.Sequential: convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5)),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            get_activation(activation),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x: tensor) -> tensor:
        """
        Forward pass of the model.

        Args:
            x (tensor): input data

        Returns:
            tensor: output data
        """
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.third_conv(x)
        x = self.fourth_conv(x)
        x = self.classifier(x)
        return x

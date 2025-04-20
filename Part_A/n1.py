import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    def __init__(
        self,
        input_dimension: tuple,
        number_of_filters: int,
        filter_size: tuple,
        stride: int,
        padding: int,
        max_pooling_size: tuple,
        n_neurons: int,
        n_classes: int,
        conv_activation: nn.Module,
        dense_activation: nn.Module,
        dropout_rate: float,
        use_batchnorm: bool,
        factor: float,
        dropout_organisation: int,
    ):
        super(CNN, self).__init__()

        self.conv_blocks = nn.ModuleList()
        in_channels = input_dimension[0]

        for i in range(5):
            out_channels = int((factor ** i) * number_of_filters)
            out_channels = max(out_channels, 3)

            add_dropout = (i % dropout_organisation) > 0

            conv_block = self.create_conv_block(
                in_channels,
                out_channels,
                filter_size,
                max_pooling_size,
                stride,
                padding,
                conv_activation,
                dropout_rate,
                use_batchnorm,
                add_dropout,
            )
            self.conv_blocks.append(conv_block)
            in_channels = out_channels

        self.flatten = nn.Flatten()

        # Compute the size after conv layers
        dummy_input = torch.ones(1, *input_dimension)
        with torch.no_grad():
            x = dummy_input
            for block in self.conv_blocks:
                x = block(x)
        in_features = x.view(1, -1).shape[1]

        # Define dense layers
        self.dense_block1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_neurons),
            dense_activation,
            nn.Linear(n_neurons, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def create_conv_block(
        self,
        in_c,
        out_c,
        kernel_size,
        max_pooling_size,
        stride,
        padding,
        conv_activation,
        dropout_rate,
        use_batchnorm,
        add_dropout,
    ):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            conv_activation
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.MaxPool2d(kernel_size=max_pooling_size))
        if add_dropout:
            layers.append(nn.Dropout(p=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.flatten(x)
        return self.dense_block1(x)

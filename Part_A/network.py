import numpy as np
import pandas as pd
from torch import nn

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
        super().__init__()
        
        # Define convolutional blocks based on input parameters

        self.conv_blocks = nn.ModuleList([])
        in_c = input_dimension[0]
        for i in range(0, 5):
            add_dropout = i % dropout_organisation > 0  # Determine whether to add dropout
            out_c = int((factor**i) * number_of_filters)  # Calculate number of output channels
            if out_c <= 0:
                out_c = 3  # Set minimum output channels to 3 if it goes below
            # Create convolutional block

            conv_block = self.create_conv_block(
                in_c,
                out_c,
                filter_size,
                max_pooling_size,
                stride,
                padding,
                conv_activation,
                dropout_rate,
                use_batchnorm,
                add_dropout,
            )
            self.conv_blocks.append(conv_block)  # Add the convolutional block to the list
            in_c = out_c  # Update input channels for the next block
        self.flatten = nn.Flatten()  # Flatten the output of convolutional layers
        
        # Determine the number of input features for the dense layers

        r = torch.ones(1, *input_dimension)
        for block in self.conv_blocks:
            block.eval()
            r = block(r)
        in_features = int(np.prod(r.size()[1:]))  # Compute the total number of features

        # Define the dense block

        self.dense_block1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=n_neurons),  # Dense layer
            dense_activation,  # Activation function for the dense layer
            nn.Linear(in_features=n_neurons, out_features=n_classes),  # Output layer
            nn.LogSoftmax(dim=1),  # Log-softmax activation for classification
        )

    def create_conv_block(
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
            nn.Conv2d(
                in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding
            ),  # Convolutional layer
            conv_activation,  # Activation function for convolutional layer
        ]
        if use_batchnorm:  # Optionally add batch normalization
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.MaxPool2d(kernel_size=max_pooling_size))  # Max pooling layer
        if add_dropout:  # Optionally add dropout
            layers.append(nn.Dropout(p=dropout_rate))
        return nn.Sequential(*layers)  # Return the sequential block of layers

    def __call__(self, x):
        r = x
        for block in self.conv_blocks:
            r = block(r)  # Pass input through each convolutional block
        r = self.flatten(r)  # Flatten the output
        return self.dense_block1(r)
    
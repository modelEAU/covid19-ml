"""
This module contains the neural network modules used in the models.
"""

import math
from typing import Union

import torch
import torch.nn as nn

from types_ml import TimeSeriesType, Variable


class SequenceTooShortError(Exception): ...


class TooMuchPaddingError(Exception): ...


def check_padding(padding: int, kernel_size):
    if padding > kernel_size // 2:
        raise TooMuchPaddingError(
            f"Padding {padding} is bigger than half of the kernel size ({kernel_size})"
        )


def conv_length_out(
    lin: int, padding: int, dilation: int, kernel: int, stride: int
) -> int:
    return math.floor((lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def conv_transpose_length_out(
    lin: int,
    padding: int,
    dilation: int,
    kernel: int,
    stride: int,
    output_padding: int = 0,
) -> int:
    return (
        (stride * (lin - 1))
        - 2 * padding
        + dilation * (kernel - 1)
        + output_padding
        + 1
    )


def get_new_activation(name: str, leaking_rate=None):
    if name is None:
        activation = nn.LeakyReLU(leaking_rate)
    else:
        activation = {
            "leaky_relu": nn.LeakyReLU(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }[name]
    return activation


class CityHeadModule(nn.Module):
    """This module is used to extract features from the city data."""

    def __init__(
        self,
        city: str,
        n_days_back: int,
        n_in_channels: int,
        n_middle_channels: int,
        n_out_channels: int,
        dropout_rate: float,
        kernel_size: int,
        n_layers: int,
        leaking_rate: float,
        conv_padding: int,
        conv_dilation: int,
        conv_stride: int,
        pool_type: int,
        pool_kernel_size: int,
        pool_padding: int,
        pool_stride: int,
        pool_dilation: int,
        activation: str,
    ):
        super().__init__()
        self.city = city
        self.n_days_back = int(n_days_back)
        self.n_in_channels = int(n_in_channels)
        self.n_middle_channels = int(n_middle_channels)
        self.n_out_channels = int(n_out_channels)
        self.kernel_size = int(kernel_size)
        self.n_layers = int(n_layers)
        self.dropout_rate = dropout_rate
        self.leaking_rate = leaking_rate
        self.conv_padding = int(conv_padding)
        self.conv_dilation = int(conv_dilation)
        self.conv_stride = int(conv_stride)
        self.pool_type = pool_type
        self.pool_kernel_size = int(pool_kernel_size)
        self.pool_padding = int(pool_padding)
        self.pool_stride = int(pool_stride)
        self.pool_dilation = int(pool_dilation)

        self.activation = activation

        layers = []  # type: ignore
        if pool_type is not None:
            pool_class = {"average": nn.AvgPool1d, "max": nn.MaxPool1d}[pool_type]
        else:
            pool_class = None
        for i in range(self.n_layers):
            if i == 0:
                layers.extend(
                    (
                        nn.Flatten(start_dim=1, end_dim=-1),
                        nn.Linear(
                            in_features=self.n_days_back * self.n_in_channels,
                            out_features=self.n_days_back * self.n_in_channels,
                        ),
                        # reshape into a 3D tensor
                        nn.Unflatten(1, (self.n_in_channels, self.n_days_back)),
                    )
                )
                if n_layers == 1:
                    # model has only one layer
                    layers.append(
                        nn.Conv1d(
                            in_channels=self.n_in_channels,
                            out_channels=self.n_out_channels,
                            kernel_size=self.kernel_size,
                            stride=self.conv_stride,
                            padding=self.conv_padding,
                            dilation=self.conv_dilation,
                        )
                    )
                else:
                    # model has more than one layer and this is the first layer
                    layers.append(
                        nn.Conv1d(
                            in_channels=self.n_in_channels,
                            out_channels=self.n_middle_channels,
                            kernel_size=self.kernel_size,
                            stride=self.conv_stride,
                            padding=self.conv_padding,
                            dilation=self.conv_dilation,
                        )
                    )
            elif i == self.n_layers - 1:
                # model has more than one layer and this is the last layer
                layers.append(
                    nn.Conv1d(
                        in_channels=self.n_middle_channels,
                        out_channels=self.n_out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.conv_stride,
                        padding=self.conv_padding,
                        dilation=self.conv_dilation,
                    )
                )
            else:
                # model has more than one layer and this is a middle layer
                layers.append(
                    nn.Conv1d(
                        in_channels=self.n_middle_channels,
                        out_channels=self.n_middle_channels,
                        kernel_size=self.kernel_size,
                        stride=self.conv_stride,
                        padding=self.conv_padding,
                        dilation=self.conv_dilation,
                    )
                )
            layers.append(get_new_activation(activation, leaking_rate))  # type: ignore
            if pool_class:
                if self.pool_type == "average":
                    layers.append(
                        pool_class(
                            kernel_size=self.pool_kernel_size,
                            stride=self.pool_stride,
                            padding=self.pool_padding,
                        )
                    )  # type: ignore # Feb 2

                else:
                    layers.append(
                        pool_class(
                            kernel_size=self.pool_kernel_size,
                            stride=self.pool_stride,
                            padding=self.pool_padding,
                            dilation=self.pool_dilation,
                        )
                    )

        layers.extend((nn.Dropout(self.dropout_rate),))  # type: ignore
        self.model = nn.Sequential(*layers)
        self.init_weights()
        if self.output_length < 1:
            raise ValueError("The channels have shrunk into non-existence")

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def init_weights(self):
        # Initialize weights and biases for Conv1d and Linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def output_length(self):
        output = self.n_days_back
        for _ in range(self.n_layers):
            output = conv_length_out(
                lin=output,
                padding=self.conv_padding,
                dilation=self.conv_dilation,
                kernel=self.kernel_size,
                stride=self.conv_stride,
            )
            if self.pool_type == "average":
                output = conv_length_out(
                    lin=output,
                    padding=self.pool_padding,
                    dilation=1,
                    kernel=self.pool_kernel_size,
                    stride=self.pool_stride,
                )
            else:
                output = conv_length_out(
                    lin=output,
                    padding=self.pool_padding,
                    dilation=self.pool_dilation,
                    kernel=self.pool_kernel_size,
                    stride=self.pool_stride,
                )
        return output


class MiddleModule(nn.Module):
    """This module is used to extract general features from the processed city data."""

    def __init__(
        self,
        input_length: int,
        n_in_channels: int,
        n_middle_channels: int,
        n_out_channels: int,
        kernel_size: int,
        n_layers: int,
        dropout_rate: float,
        leaking_rate: float,
        conv_padding: int,
        conv_dilation: int,
        conv_stride: int,
        pool_type: str,
        pool_kernel_size: int,
        pool_padding: int,
        pool_stride: int,
        pool_dilation: int,
        activation: str = None,
        entry_nn: bool = True,
    ):
        super().__init__()
        self.input_length = input_length
        self.n_in_channels = n_in_channels
        self.n_middle_channels = n_middle_channels
        self.n_out_channels = n_out_channels
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.leaking_rate = leaking_rate
        self.conv_padding = conv_padding
        self.conv_dilation = conv_dilation
        self.conv_stride = conv_stride
        self.pool_padding = pool_padding
        self.pool_type = pool_type
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_dilation = pool_dilation
        self.activation = activation
        self.entry_nn = entry_nn
        layers = []  # type: ignore

        if pool_type is not None:
            pool_class = {"average": nn.AvgPool1d, "max": nn.MaxPool1d}[pool_type]
        else:
            pool_class = None
        sequence_length = self.input_length
        check_padding(self.conv_padding, self.kernel_size)
        if pool_class:
            check_padding(self.pool_padding, self.pool_kernel_size)
        for i in range(self.n_layers):
            if i == 0:
                if self.entry_nn:
                    # model has only one layer
                    layers.extend(
                        (
                            nn.Flatten(),
                            nn.Linear(
                                in_features=self.input_length * self.n_in_channels,
                                out_features=self.input_length * self.n_in_channels,
                            ),
                            # reshape into a 3D tensor
                            nn.Unflatten(1, (self.n_in_channels, self.input_length)),
                        )
                    )
                if self.n_layers == 1:
                    layers.append(
                        nn.Conv1d(
                            in_channels=self.n_in_channels,
                            out_channels=self.n_out_channels,
                            kernel_size=self.kernel_size,
                            stride=self.conv_stride,
                            padding=self.conv_padding,
                            dilation=self.conv_dilation,
                        )
                    )
                    sequence_length = conv_length_out(
                        sequence_length,
                        self.conv_padding,
                        self.conv_dilation,
                        self.kernel_size,
                        self.conv_stride,
                    )
                else:
                    layers.append(
                        nn.Conv1d(
                            self.n_in_channels,
                            self.n_middle_channels,
                            kernel_size=self.kernel_size,
                            stride=self.conv_stride,
                            padding=self.conv_padding,
                            dilation=self.conv_dilation,
                        )
                    )
                    sequence_length = conv_length_out(
                        sequence_length,
                        self.conv_padding,
                        self.conv_dilation,
                        self.kernel_size,
                        self.conv_stride,
                    )

            elif i == self.n_layers - 1:
                # model has more than one layer and this is the last layer
                layers.append(
                    nn.Conv1d(
                        in_channels=n_middle_channels,
                        out_channels=n_out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.conv_stride,
                        padding=self.conv_padding,
                        dilation=self.conv_dilation,
                    )
                )
                sequence_length = conv_length_out(
                    sequence_length,
                    self.conv_padding,
                    self.conv_dilation,
                    self.kernel_size,
                    self.conv_stride,
                )
            else:
                # model has more than one layer and this is a middle layer
                layers.append(
                    nn.Conv1d(
                        in_channels=n_middle_channels,
                        out_channels=n_middle_channels,
                        kernel_size=self.kernel_size,
                        stride=self.conv_stride,
                        padding=self.conv_padding,
                        dilation=self.conv_dilation,
                    )
                )
                sequence_length = conv_length_out(
                    sequence_length,
                    self.conv_padding,
                    self.conv_dilation,
                    self.kernel_size,
                    self.conv_stride,
                )
                # append a dropout layer after each middle layer
                layers.append(nn.Dropout(self.dropout_rate))  # type: ignore
            layers.append(get_new_activation(activation, leaking_rate))  # type: ignore # Feb 2
            if pool_class:
                if self.pool_type == "average":
                    layers.append(
                        pool_class(
                            kernel_size=self.pool_kernel_size,
                            stride=self.pool_stride,
                            padding=self.pool_padding,
                        )
                    )  # type: ignore # Feb 2
                    sequence_length = conv_length_out(
                        sequence_length,
                        self.pool_padding,
                        1,
                        self.pool_kernel_size,
                        self.pool_stride,
                    )
                else:
                    layers.append(
                        pool_class(
                            kernel_size=self.pool_kernel_size,
                            stride=self.pool_stride,
                            padding=self.pool_padding,
                            dilation=self.pool_dilation,
                        )
                    )
                    sequence_length = conv_length_out(
                        sequence_length,
                        self.pool_padding,
                        self.pool_dilation,
                        self.pool_kernel_size,
                        self.pool_stride,
                    )
            if sequence_length < 1:
                raise SequenceTooShortError(
                    f"Ran out of items in the sequence at layer {i}"
                )

        layers.extend((nn.Dropout(self.dropout_rate),))  # type: ignore
        self.model = nn.Sequential(*layers)
        self.init_weights()
        if self.output_length < 1:
            raise ValueError("The channels have shrunk into non-existence")

    @torch.no_grad()
    def init_weights(self):
        # Initialize weights and biases for Conv1d and Linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

    @property
    def output_length(self):
        output = self.input_length
        for _ in range(self.n_layers):
            output = conv_length_out(
                lin=output,
                padding=self.conv_padding,
                dilation=self.conv_dilation,
                kernel=self.kernel_size,
                stride=self.conv_stride,
            )
            if self.pool_type == "average":
                output = conv_length_out(
                    lin=output,
                    padding=self.pool_padding,
                    dilation=1,
                    kernel=self.pool_kernel_size,
                    stride=self.pool_stride,
                )
            else:
                output = conv_length_out(
                    lin=output,
                    padding=self.pool_padding,
                    dilation=self.pool_dilation,
                    kernel=self.pool_kernel_size,
                    stride=self.pool_stride,
                )
        return output


class CityClassifier(nn.Module):
    """This module is used to output a classification of city labels"""

    def __init__(
        self,
        n_in_channels: int,
        n_middle_channels: int,
        n_out_channels: int,
        n_per_in_channel: int,
        n_out_neurons: int,
        n_layers: int,
        leaking_rate: float,
        kernel_size: int,
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_per_in_channel = n_per_in_channel
        self.n_middle_channels = n_middle_channels
        self.n_out_channels = n_out_channels
        self.n_out_neurons = n_out_neurons
        self.n_layers = n_layers
        self.leaking_rate = leaking_rate
        self.kernel_size = kernel_size

        layers = []  # type: ignore
        for i in range(self.n_layers):
            if i == 0:
                layers.extend(
                    (
                        nn.Flatten(),
                        nn.Linear(
                            in_features=self.n_in_channels * self.n_per_in_channel,
                            out_features=self.n_in_channels * self.n_per_in_channel,
                        ),
                        # reshape into a 3D tensor
                        nn.Unflatten(1, (self.n_in_channels, self.n_per_in_channel)),
                    )
                )
                if self.n_layers == 1:
                    # model has only one layer
                    layers.append(
                        nn.ConvTranspose1d(
                            self.n_in_channels, self.n_out_channels, kernel_size
                        )
                    )
                else:
                    # model has more than one layer and this is the first layer
                    layers.append(
                        nn.ConvTranspose1d(
                            self.n_in_channels, self.n_middle_channels, kernel_size
                        )
                    )
            elif i == self.n_layers - 1:
                # model has more than one layer and this is the last layer
                layers.append(
                    nn.ConvTranspose1d(n_middle_channels, n_out_channels, kernel_size)
                )
            else:
                # model has more than one layer and this is a middle layer
                layers.append(
                    nn.ConvTranspose1d(
                        n_middle_channels, n_middle_channels, kernel_size
                    )
                )
            layers.append(nn.LeakyReLU(self.leaking_rate))  # type: ignore
        n_linear_neurons = n_out_channels * (
            n_per_in_channel + ((self.kernel_size - 1) * self.n_layers)
        )
        layers.extend(
            (  # type: ignore
                nn.Flatten(),
                nn.Linear(
                    in_features=n_linear_neurons,
                    out_features=self.n_out_neurons,
                ),
            )
        )
        self.model = nn.Sequential(*layers)

        self.init_weights()

    def forward(self, x):
        y = self.model(x)
        return y

    @torch.no_grad()
    def init_weights(self):
        # Initialize weights and biases for Conv1d and Linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class PredictionHead(nn.Module):
    """This module is used to output a number of predictions for a given variable"""

    def __init__(
        self,
        n_in_channels: int,
        n_middle_channels: int,
        n_out_channels: int,
        n_per_in_channel: int,
        n_out_neurons: int,
        variable: Variable,
        ts_type: TimeSeriesType,
        n_layers: int,
        leaking_rate: float,
        kernel_size: int,
        dropout_rate: float,
        conv_padding: int,
        conv_dilation: int,
        conv_stride: int,
        pool_type: int,
        pool_kernel_size: int,
        pool_padding: int,
        pool_stride: int,
        pool_dilation: int,
        activation: str,
        entry_nn: bool = True,
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_per_in_channel = n_per_in_channel
        self.n_middle_channels = n_middle_channels
        self.n_out_channels = n_out_channels
        self.n_out_neurons = n_out_neurons
        self.variable = variable
        self.ts_type = ts_type
        self.n_layers = n_layers
        self.leaking_rate = leaking_rate
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.conv_padding = conv_padding
        self.conv_dilation = conv_dilation
        self.conv_stride = conv_stride
        self.pool_type = pool_type
        self.pool_kernel_size = pool_kernel_size
        self.pool_padding = pool_padding
        self.pool_stride = pool_stride
        self.pool_dilation = pool_dilation
        self.activation = activation
        self.entry_nn = entry_nn
        sequence_length = self.n_per_in_channel
        layers = []  # type: ignore
        if pool_type is not None:
            pool_class = {"average": nn.AvgPool1d, "max": nn.MaxPool1d}[pool_type]
        else:
            pool_class = None
        check_padding(self.conv_padding, self.kernel_size)
        if pool_class:
            check_padding(self.pool_padding, self.pool_kernel_size)
        for i in range(self.n_layers):
            if i == 0:
                if entry_nn:
                    layers.extend(
                        (
                            nn.Flatten(),
                            nn.Linear(
                                in_features=self.n_in_channels * self.n_per_in_channel,
                                out_features=self.n_in_channels * self.n_per_in_channel,
                            ),
                            # reshape into a 3D tensor
                            nn.Unflatten(
                                1, (self.n_in_channels, self.n_per_in_channel)
                            ),
                        )
                    )
                # model has only one layer
                if self.n_layers == 1:
                    layers.append(
                        nn.ConvTranspose1d(
                            in_channels=self.n_in_channels,
                            out_channels=self.n_out_channels,
                            kernel_size=kernel_size,
                            padding=self.conv_padding,
                            dilation=self.conv_dilation,
                            stride=self.conv_stride,
                        )
                    )
                    sequence_length = conv_transpose_length_out(
                        sequence_length,
                        self.conv_padding,
                        self.conv_dilation,
                        self.kernel_size,
                        self.conv_stride,
                    )
                else:
                    # model has more than one layer and this is the first layer
                    layers.append(
                        nn.ConvTranspose1d(
                            in_channels=self.n_in_channels,
                            out_channels=self.n_middle_channels,
                            kernel_size=kernel_size,
                            padding=self.conv_padding,
                            dilation=self.conv_dilation,
                            stride=self.conv_stride,
                        )
                    )
                    sequence_length = conv_transpose_length_out(
                        sequence_length,
                        self.conv_padding,
                        self.conv_dilation,
                        self.kernel_size,
                        self.conv_stride,
                    )
            elif i == self.n_layers - 1:
                # model has more than one layer and this is the last layer
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=n_middle_channels,
                        out_channels=n_out_channels,
                        kernel_size=kernel_size,
                        padding=self.conv_padding,
                        dilation=self.conv_dilation,
                        stride=self.conv_stride,
                    )
                )
                sequence_length = conv_transpose_length_out(
                    sequence_length,
                    self.conv_padding,
                    self.conv_dilation,
                    self.kernel_size,
                    self.conv_stride,
                )
            else:
                # model has more than one layer and this is a middle layer
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=n_middle_channels,
                        out_channels=n_middle_channels,
                        kernel_size=kernel_size,
                        padding=self.conv_padding,
                        dilation=self.conv_dilation,
                        stride=self.conv_stride,
                    )
                )
                sequence_length = conv_transpose_length_out(
                    sequence_length,
                    self.conv_padding,
                    self.conv_dilation,
                    self.kernel_size,
                    self.conv_stride,
                )
                # append a dropout layer after each middle layer
                layers.append(nn.Dropout(self.dropout_rate))  # type: ignore
            layers.append(get_new_activation(activation, leaking_rate))  # type: ignore
            if pool_class:
                if self.pool_type == "average":
                    layers.append(
                        pool_class(
                            kernel_size=self.pool_kernel_size,
                            stride=self.pool_stride,
                            padding=self.pool_padding,
                        )
                    )  # type: ignore # Feb 2
                    sequence_length = conv_length_out(
                        sequence_length,
                        self.pool_padding,
                        1,
                        self.pool_kernel_size,
                        self.pool_stride,
                    )
                else:
                    layers.append(
                        pool_class(
                            kernel_size=self.pool_kernel_size,
                            stride=self.pool_stride,
                            padding=self.pool_padding,
                            dilation=self.pool_dilation,
                        )
                    )
                    sequence_length = conv_length_out(
                        sequence_length,
                        self.pool_padding,
                        self.pool_dilation,
                        self.pool_kernel_size,
                        self.pool_stride,
                    )
            if sequence_length < 1:
                raise SequenceTooShortError(
                    f"Ran out of items in the sequence at layer {i}"
                )
        n_linear_neurons = (
            n_out_channels * sequence_length
            if self.n_layers
            else sequence_length * self.n_in_channels
        )
        layers.extend(
            (  # type: ignore
                nn.Flatten(),
                nn.Linear(
                    in_features=n_linear_neurons,
                    out_features=self.n_out_neurons,
                ),
            )
        )
        self.model = nn.Sequential(*layers)

        self.init_weights()

    @property
    def output_length(self):
        output = self.input_length
        for _ in range(self.n_layers):
            output = conv_length_out(
                lin=output,
                padding=self.conv_padding,
                dilation=self.conv_dilation,
                kernel=self.kernel_size,
                stride=self.conv_stride,
            )
            if self.pool_type == "average":
                output = conv_length_out(
                    lin=output,
                    padding=self.pool_padding,
                    dilation=1,
                    kernel=self.pool_kernel_size,
                    stride=self.pool_stride,
                )
            else:
                output = conv_length_out(
                    lin=output,
                    padding=self.pool_padding,
                    dilation=self.pool_dilation,
                    kernel=self.pool_kernel_size,
                    stride=self.pool_stride,
                )
        return output

    def forward(self, x):
        y = self.model(x)
        return y

    @torch.no_grad()
    def init_weights(self):
        # Initialize weights and biases for Conv1d and Linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class SmallNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_out, n_hidden_layers) -> None:
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.n_hidden_layers = int(n_hidden_layers)

        layers = []
        if self.n_hidden_layers == 0:
            layers.append(nn.Linear(self.n_inputs, self.n_out))
            layers.append(nn.ReLU())  # type: ignore

        else:
            for i in range(self.n_hidden_layers):
                layers.append(nn.Linear(self.n_inputs, self.n_hidden))
                layers.append(nn.ReLU())  # type: ignore
                self.n_inputs = self.n_hidden
            layers.append(nn.Linear(self.n_hidden, self.n_out))

        self.model = nn.Sequential(*layers)
        self.model.apply(self.weights_init)

    @torch.no_grad()
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )  # He Initialization for ReLU
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class SmallConvDecreasingKernels(nn.Module):
    def __init__(
        self,
        n_per_in_channels,
        n_in_channels,
        n_middle_channels,
        n_out_channels,
        n_per_out_channel,
        n_layers,
        kernel_size,
    ) -> None:
        super().__init__()
        self.n_per_in_channels = n_per_in_channels
        self.n_layers = n_layers
        self.n_in_channels = n_in_channels
        self.n_middle_channels = n_middle_channels
        self.n_out_channels = n_out_channels
        self.n_per_out_channel = n_per_out_channel
        self.kernel_size = kernel_size

        layers: list[Union[torch.Tensor, nn.Module, nn.Sequential]] = []
        sequence_length = self.n_per_in_channels
        kernel_size = self.kernel_size
        for i in range(self.n_layers):
            if i == 0 and self.n_layers == 1:
                # model has only one layer
                layers.extend(
                    [
                        nn.Conv1d(self.n_in_channels, self.n_out_channels, kernel_size),
                        nn.Dropout(0.2),
                    ]
                )
            elif i == 0:
                # model has more than one layer and this is the first layer
                layers.extend(
                    [
                        nn.Conv1d(
                            self.n_in_channels, self.n_middle_channels, kernel_size
                        ),
                        nn.Dropout(0.2),
                    ]
                )
            elif i == self.n_layers - 1:
                # model has more than one layer and this is the last layer
                layers.extend(
                    [
                        nn.Conv1d(
                            self.n_middle_channels, self.n_out_channels, kernel_size
                        ),
                        nn.Dropout(0.2),
                    ]
                )
            else:
                # model has more than one layer and this is a middle layer
                layers.extend(
                    [
                        nn.Conv1d(
                            self.n_middle_channels,
                            self.n_middle_channels,
                            kernel_size,
                        ),
                        nn.Dropout(0.2),
                    ]
                )
            layers.append(nn.ReLU())
            sequence_length = sequence_length - kernel_size + 1
            kernel_size -= 2
        layers.extend(
            (
                nn.Flatten(),
                nn.Linear(
                    self.n_out_channels * sequence_length,
                    self.n_per_out_channel,
                ),
            )
        )
        self.model = nn.Sequential(*layers)  # type: ignore
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Initialize weights and biases for Conv1d and Linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)


class SmallLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, n_prediction_out):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_prediction_out = n_prediction_out

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(n_hidden, n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_features * n_prediction_out)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Initialize LSTM weights and biases
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        # Initialize Linear layer weights with Xavier initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        initial_shape = x.shape
        if len(initial_shape) == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        for_prediction = lstm_out[:, -1]
        output = self.linear2(self.relu(self.linear(for_prediction)))

        if len(initial_shape) == 2:
            return output.reshape(self.n_prediction_out, self.n_features)
        return output.reshape(initial_shape[0], self.n_prediction_out, self.n_features)

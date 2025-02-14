import sys

import torch
import torch.nn as nn

sys.path.append("src")
from modules import (
    CityHeadModule,
    MiddleModule,  # noqa: E402
    PredictionHead,
    SmallNN,
)
from types_ml import City, TimeSeriesType, Variable


def test_smal_nn_instantiation():
    model = SmallNN(n_inputs=4, n_hidden=10, n_out=2, n_hidden_layers=2)
    assert isinstance(model, SmallNN)


def test_small_nn_output_shape():
    model = SmallNN(n_inputs=4, n_hidden=10, n_out=2, n_hidden_layers=2)
    input_tensor = torch.randn(8, 4)  # Batch size of 8
    output = model(input_tensor)
    assert output.shape == (8, 2)
    model = SmallNN(n_inputs=32, n_hidden=10, n_out=2, n_hidden_layers=2)
    input_tensor = torch.randn(1, 8, 4)
    output = model(input_tensor.view(1, -1))
    assert output.shape == (1, 2)
    # Adjust based on your expected output shape


def test_small_nn_training():
    model = SmallNN(n_inputs=4, n_hidden=10, n_out=2, n_hidden_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    input_tensor = torch.randn(8, 4)
    target = torch.randn(8, 2)

    # Simple training loop for a few epochs
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        if epoch == 0:
            first_loss = loss.clone().detach()
        loss.backward()

        optimizer.step()

    # Assuming the model should overfit the random data in a few epochs
    new_loss = loss.clone().detach()
    assert new_loss < first_loss


def test_weights_initialization():
    model = SmallNN(n_inputs=4, n_hidden=10, n_out=2, n_hidden_layers=2)

    for layer in model.model:
        if isinstance(layer, nn.Linear):
            # Check weights initialization using He initialization
            # Check weights aren't all zeros
            assert (layer.weight.data != 0).any()

            # Check bias initialization
            assert (layer.bias.data == 0).all()
            # Ensure no NaN values in weights
            assert not torch.isnan(layer.weight.data).any()

            # Ensure no NaN values in biases
            assert not torch.isnan(layer.bias.data).any()


def test_prediction_head():
    # Dummy classes for the types, you can replace these with actual types if available.
    variable = Variable("COD")
    ts_type = TimeSeriesType("TREND_CURVATURE")

    # 1. Test instantiation
    model = PredictionHead(
        n_in_channels=3,
        n_middle_channels=4,
        n_out_channels=5,
        n_per_in_channel=10,
        n_out_neurons=7,
        variable=variable,
        ts_type=ts_type,
        n_layers=2,
        leaking_rate=0.01,
        kernel_size=3,
        dropout_rate=0.1,
        conv_padding=0,
        conv_dilation=1,
        conv_stride=1,
        pool_type="average",
        pool_kernel_size=1,
        pool_padding=0,
        pool_stride=1,
        pool_dilation=1,
        activation="leaky_relu",
        entry_nn=True,
    )

    # 2. Test forward pass with dummy data
    x = torch.rand((5, 3, 10))
    y = model(x)

    # 3. Test output shape
    assert len(y.shape) == 2  # Batch_size, n_out_neurons (no n_classes dimension)

    # 4. Task Regression
    assert y.shape[-1] == 7  # n_out_neurons


def test_prediction_head_learns_regression():
    # Dummy classes for the types, you can replace these with actual types if available.
    variable = Variable("COD")
    ts_type = TimeSeriesType("TREND_CURVATURE")

    # 1. Test instantiation
    model = PredictionHead(
        n_in_channels=3,
        n_middle_channels=4,
        n_out_channels=5,
        n_per_in_channel=10,
        n_out_neurons=7,
        variable=variable,
        ts_type=ts_type,
        n_layers=2,
        leaking_rate=0.01,
        kernel_size=3,
        dropout_rate=0.1,
        conv_padding=0,
        conv_dilation=1,
        conv_stride=1,
        pool_type="average",
        pool_kernel_size=1,
        pool_padding=0,
        pool_stride=1,
        pool_dilation=1,
        activation="leaky_relu",
        entry_nn=True,
    )

    # 2. Test forward pass with dummy data
    x = torch.rand((5, 3, 10))
    y = model(x)
    target = torch.rand((5, 7))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Simple training loop for a few epochs
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        if epoch == 0:
            first_loss = loss.clone().detach()
        loss.backward()

        optimizer.step()

    # Assuming the model should overfit the random data in a few epochs
    new_loss = loss.clone().detach()
    assert new_loss < first_loss


def test_middle_module():
    model = MiddleModule(
        input_length=14,
        n_in_channels=3,
        n_middle_channels=4,
        n_out_channels=5,
        kernel_size=3,
        n_layers=2,
        dropout_rate=0.2,
        leaking_rate=0.01,
        conv_padding=0,
        conv_dilation=1,
        conv_stride=1,
        pool_type="average",
        pool_kernel_size=1,
        pool_padding=0,
        pool_stride=1,
        pool_dilation=1,
        activation="leaky_relu",
        entry_nn=True,
    )
    assert isinstance(model, MiddleModule)

    x = torch.rand((5, 3, 14))
    y = model(x)
    assert y.shape == (5, 5, 10)
    target = torch.rand((5, 5, 10))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Simple training loop for a few epochs
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        # output_classes = torch.argmax(output, dim=1)
        loss = criterion(output, target)
        if epoch == 0:
            first_loss = loss.clone().detach()
        loss.backward()

        optimizer.step()

    # Assuming the model should overfit the random data in a few epochs
    new_loss = loss.clone().detach()
    assert new_loss < first_loss


def test_city_head_module():
    model = CityHeadModule(
        city=City.QUEBEC_EAST.value,
        n_days_back=14,
        n_in_channels=3,
        n_middle_channels=4,
        n_out_channels=5,
        dropout_rate=0.2,
        kernel_size=3,
        n_layers=2,
        leaking_rate=0.01,
        conv_padding=0,
        conv_dilation=1,
        conv_stride=1,
        pool_type="average",
        pool_kernel_size=1,
        pool_padding=0,
        pool_stride=1,
        pool_dilation=1,
        activation="leaky_relu",
    )
    # Create dummy input tensor
    x = torch.randn(32, model.n_in_channels, 14)  #  sequence length of 14

    # Forward pass
    output = model(x)
    expected_length = 10
    target = torch.randn(32, model.n_out_channels, expected_length)
    # Check output dimensions
    expected_channels = model.n_out_channels
    assert output.shape == (
        32,
        expected_channels,
        expected_length,
    ), f"Output shape mismatch: {output.shape}"

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Simple training loop for a few epochs
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(x)
        # output_classes = torch.argmax(output, dim=1)
        loss = criterion(output, target)
        if epoch == 0:
            first_loss = loss.clone().detach()
        loss.backward()

        optimizer.step()

    # Assuming the model should overfit the random data in a few epochs
    new_loss = loss.clone().detach()
    assert new_loss < first_loss

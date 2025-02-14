import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append("src")
from datamodules import CityDataModule, CombinedDataModule  # noqa: E402
from datasets import CityDataSet  # noqa: E402
from models import CityConvModel  # noqa: E402
from types_config import (  # noqa: E402
    CityClassifierConfig,
    CityConvConfig,
    HyperParameters,  # noqa: E402
    OptimizerConfig,  # noqa: E402
    Recipe,  # noqa: E402
)
from types_ml import City, Task, TensorConfig, TimeSeriesType, Variable  # noqa: E402


@pytest.fixture
def parquet_path_short():
    test_path = "./tests/test_data"

    # create a temporary parquet file for testing
    df = pd.DataFrame(
        {
            "Calculated_timestamp": pd.date_range("2020-01-01", "2020-01-12"),
            "CASES": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "HOSPITALIZED": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "BOD": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "COD": [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "TEMP": [x for x in range(12)],
        }
    )
    df = df.set_index("Calculated_timestamp")
    file_path = Path(test_path) / "test.parquet"
    df.to_parquet(file_path)
    return file_path


@pytest.fixture
def parquet_path_long():
    test_path = "./tests/test_data"

    # create a temporary parquet file for testing
    df = pd.DataFrame(
        {
            "Calculated_timestamp": pd.date_range("2020-01-01", "2020-02-19"),
            "CASES": list(range(1, 51)),
            "HOSPITALIZED": list(range(50, 0, -1)),
            "BOD": list(range(1, 51)),
            "COD": list(range(50, 0, -1)),
            "TEMP": [x / 2 for x in range(50)],
        }
    )
    df = df.set_index("Calculated_timestamp")
    file_path = Path(test_path) / "test.parquet"
    df.to_parquet(file_path)
    return file_path


@pytest.fixture
def sample_tensor_config_regression():
    return TensorConfig(
        input_variables=[Variable("CASES"), Variable("HOSPITALIZED")],
        target_variables=[Variable("COD"), Variable("BOD")],
        input_ts_types=[TimeSeriesType("RAW"), TimeSeriesType("SMOOTH")],
        target_ts_types=[
            TimeSeriesType("TREND_SLOPE"),
            TimeSeriesType("TREND_CURVATURE"),
        ],
        n_timesteps_back=5,
        n_timesteps_forward=4,
        task=Task.REGRESSION,
        trend_model_order=2,
        trend_model_window=3,
        days_before=-1,
        insert_dummy_variable=False,
        artificial_noise=False,
    )


@pytest.fixture
def short_sample_citydataset_regression(
    parquet_path_short: Path, sample_tensor_config_regression: TensorConfig
):
    return CityDataSet(
        path=parquet_path_short,
        city=City.EASYVILLE_1,
        start_date="2020-01-01",
        end_date="2020-01-08",
        tensor_config=sample_tensor_config_regression,
        validation_split=2,
    )


@pytest.fixture
def long_sample_citydataset_regression1(
    parquet_path_long: Path, sample_tensor_config_regression: TensorConfig
):
    return CityDataSet(
        path=parquet_path_long,
        city=City.EASYVILLE_1,
        start_date="2020-01-01",
        end_date="2020-02-15",
        tensor_config=sample_tensor_config_regression,
    )


@pytest.fixture
def long_sample_citydataset_regression2(
    parquet_path_long: Path, sample_tensor_config_regression: TensorConfig
):
    return CityDataSet(
        path=parquet_path_long,
        city=City.EASYVILLE_2,
        start_date="2020-01-05",
        end_date="2020-02-15",
        tensor_config=sample_tensor_config_regression,
    )


@pytest.fixture
def sample_regression_citydatamodule1(long_sample_citydataset_regression1: CityDataSet):
    return CityDataModule(
        city_dataset=long_sample_citydataset_regression1,
        test_split_date="2020-02-05",
        batch_size=2,
        task=Task.REGRESSION,
        n_splits=5,
        validation_split=2,
    )


@pytest.fixture
def sample_regression_citydatamodule2(long_sample_citydataset_regression2: CityDataSet):
    return CityDataModule(
        city_dataset=long_sample_citydataset_regression2,
        test_split_date="2020-02-05",
        batch_size=2,
        n_splits=5,
        task=Task.REGRESSION,
    )


@pytest.fixture
def sample_classification_citydatamodule1(
    long_sample_citydataset_classification1: CityDataSet,
):
    return CityDataModule(
        city_dataset=long_sample_citydataset_classification1,
        test_split_date="2020-02-05",
        batch_size=2,
        n_splits=5,
        task=Task.CLASSIFICATION,
    )


@pytest.fixture
def sample_classification_citydatamodule2(
    long_sample_citydataset_classification2: CityDataSet,
):
    return CityDataModule(
        city_dataset=long_sample_citydataset_classification2,
        test_split_date="2020-02-05",
        batch_size=2,
        n_splits=5,
        task=Task.CLASSIFICATION,
    )


@pytest.fixture
def combined_datamodule_regression(
    sample_regression_citydatamodule1: CityDataModule,
    sample_regression_citydatamodule2: CityDataModule,
):
    datamodules_list = [
        sample_regression_citydatamodule1,
        sample_regression_citydatamodule2,
    ]
    return CombinedDataModule(
        datamodules=datamodules_list,
        batch_size=2,
        city_balancer=None,
    )


@pytest.fixture
def recipe():
    return Recipe(
        inputs=["CASES", "HOSPITALIZED"],
        input_ts=["RAW", "SMOOTH"],
        targets=["CASES", "HOSPITALIZED"],
        target_ts=["RAW", "SMOOTH"],
    )


@pytest.fixture
def regression_tensor_config():
    return TensorConfig(
        input_variables=[Variable("CASES"), Variable("HOSPITALIZED")],
        input_ts_types=[TimeSeriesType("RAW"), TimeSeriesType("SMOOTH")],
        target_variables=[Variable("CASES"), Variable("HOSPITALIZED")],
        target_ts_types=[TimeSeriesType("RAW"), TimeSeriesType("SMOOTH")],
        n_timesteps_back=5,
        n_timesteps_forward=4,
        task=Task.REGRESSION,
        trend_model_order=2,
        trend_model_window=3,
        insert_dummy_variable=False,
        days_before=-1,
        artificial_noise=False,
    )


@pytest.fixture
def classification_tensor_config():
    return TensorConfig(
        input_variables=[Variable("CASES"), Variable("HOSPITALIZED")],
        input_ts_types=[
            TimeSeriesType("SMOOTH"),
            TimeSeriesType("TREND_SLOPE"),
            TimeSeriesType("TREND_CURVATURE"),
        ],
        target_variables=[Variable("CASES"), Variable("HOSPITALIZED")],
        target_ts_types=[
            TimeSeriesType("TREND_SLOPE"),
            TimeSeriesType("TREND_CURVATURE"),
        ],
        n_timesteps_back=5,
        n_timesteps_forward=4,
        task=Task.CLASSIFICATION,
        trend_model_order=2,
        trend_model_window=3,
    )


@pytest.fixture
def conv_conf():
    return CityConvConfig(
        n_middle_channels_city=1,
        n_out_channels_city=16,
        n_layers_city=1,
        city_pooling_padding=1,
        city_pooling_kernel_size=5,
        city_pooling_stride=1,
        city_pooling_dilation=2,
        city_kernel_size=3,
        city_conv_padding=1,
        city_conv_dilation=1,
        city_conv_stride=1,
        n_layers_middle=1,
        entry_nn_middle=True,
        entry_nn_prediction=False,
        n_middle_channels_middle=4,
        n_out_channels_middle=8,
        middle_pooling_padding=0,
        middle_pooling_kernel_size=3,
        middle_pooling_stride=1,
        middle_pooling_dilation=1,
        middle_kernel_size=2,
        middle_conv_padding=0,
        middle_conv_dilation=1,
        middle_conv_stride=3,
        n_layers_prediction=3,
        n_middle_channels_prediction=8,
        n_out_channels_prediction=4,
        prediction_kernel_size=3,
        prediction_pooling_padding=1,
        prediction_pooling_kernel_size=5,
        prediction_pooling_stride=2,
        prediction_pooling_dilation=1,
        prediction_conv_padding=0,
        prediction_conv_dilation=1,
        prediction_conv_stride=2,
        pooling_type="max",
        dropout_rate=0.3,
        leaking_rate=0.001,
        activation="leaky_relu",
    )


@pytest.fixture
def opt_conf():
    return OptimizerConfig(
        type="SGD",
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.01,
        learning_rate_decay=0.01,
    )


@pytest.fixture
def city_classifier_conf():
    return CityClassifierConfig(
        n_layers=2,
        n_hidden=10,
        dropout_rate=0.1,
    )


@pytest.fixture
def hparams(
    opt_conf: OptimizerConfig,
    conv_conf: CityConvConfig,
    city_classifier_conf: CityClassifierConfig,
):
    return HyperParameters(
        recipe=None,
        random_seed=42,
        task="REGRESSION",
        trend_model_window=7,
        trend_model_order=1,
        denoise_window=3,
        insert_dummy_variable=False,
        model_type="city_conv",
        train_classification=False,
        train_prediction=True,
        use_identity_for_city_heads=True,
        loss_fn="MAE",
        optimizer=opt_conf,
        classifier=city_classifier_conf,
        city_conv=conv_conf,
        n_montecarlo_samples=25,
        batch_size=4,
        classification_batch_size=8,
        n_back=14,
        n_forward=7,
        n_splits=5,
        patience=10,
        target_type="delta",
        weighted_loss=True,
        kl_weight=-1,
        classifier_regularization=0.5,
        fake_ww_shift=-1,
        artificial_noise=False,
        log_transform=False,
        small_nn=None,
    )


@pytest.fixture
def regression_city_conv_model(
    hparams: HyperParameters,
    opt_conf: OptimizerConfig,
    conv_conf: CityConvConfig,
    city_classifier_conf: CityClassifierConfig,
    regression_tensor_config: TensorConfig,
):
    return CityConvModel(
        cities=["EASYVILLE_1"],
        hparams=hparams,
        optimizer_config=opt_conf,
        conv_config=conv_conf,
        city_classifier_config=city_classifier_conf,
        tensor_config=regression_tensor_config,
        trial=None,
    )


@pytest.fixture
def classification_city_conv_model(
    hparams: HyperParameters,
    opt_conf: OptimizerConfig,
    conv_conf: CityConvConfig,
    city_classifier_conf: CityClassifierConfig,
    classification_tensor_config: TensorConfig,
):
    return CityConvModel(
        cities=["EASYVILLE_1"],
        hparams=hparams,
        optimizer_config=opt_conf,
        conv_config=conv_conf,
        city_classifier_config=city_classifier_conf,
        tensor_config=classification_tensor_config,
        trial=None,
    )

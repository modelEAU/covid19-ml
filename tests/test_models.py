import itertools
import sys

import pandas as pd
import pytest
import torch
from fixtures import (  # noqa: E402 F401
    city_classifier_conf,
    combined_datamodule_regression,
    conv_conf,
    hparams,
    long_sample_citydataset_regression1,
    long_sample_citydataset_regression2,
    opt_conf,
    parquet_path_long,
    recipe,
    regression_city_conv_model,
    regression_tensor_config,
    sample_classification_citydatamodule1,
    sample_classification_citydatamodule2,
    sample_regression_citydatamodule1,
    sample_regression_citydatamodule2,
    sample_tensor_config_regression,
)

sys.path.append("src")

from models import CityConvModel  # noqa: E402
from types_ml import (
    HeadTensors,  # noqa: E402
    Stage,  # noqa: E402
    StepResult,  # noqa: E402
    Task,  # noqa: E402
    TimeSeriesType,  # noqa: E402
    Variable,  # noqa: E402
)


def test_check_for_nan():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1, 2, float("nan")])
    assert CityConvModel._check_for_nan(x, "x") is None

    with pytest.raises(ValueError):
        CityConvModel._check_for_nan(y, "y")


def test_ensure_3d():
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    z = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    assert CityConvModel._ensure_3d(x).shape == (1, 1, 3)
    assert CityConvModel._ensure_3d(y).shape == (1, 2, 3)
    assert CityConvModel._ensure_3d(z).shape == (1, 2, 3)


def test_lookup_key():
    var = Variable.CASES
    time_series_type = TimeSeriesType.RAW
    expected = "CASES_RAW"
    assert CityConvModel._lookup_key(var, time_series_type) == expected


def test_prediction_key():
    var = Variable.CASES
    time_series_type = TimeSeriesType.RAW
    expected = "CASES,RAW"
    assert CityConvModel._prediction_key(var, time_series_type) == expected


def test_normalize_input():
    x = torch.ones(4, 1)
    lookup = {
        "CASES_RAW": 0,
        "DEATHS_RAW": 1,
        "CASES_SMOOTH": 2,
        "DEATHS_SMOOTH": 3,
    }
    norm_stats = {
        "CASES_RAW": (0.0, 1.0),
        "DEATHS_RAW": (2.0, 2.0),
        "CASES_SMOOTH": (3.0, 1.0),
        "DEATHS_SMOOTH": (2.0, 0.0),
    }
    expected = torch.tensor([[0.9999], [-0.5000], [-2.0000], [-100000.0]])
    result = CityConvModel._normalize_input(x, lookup, norm_stats, log_transform=False)

    assert result.shape == (4, 1)
    assert torch.allclose(result - expected, torch.zeros(4, 1), atol=1e-4)


def test_normalize_output():
    y = torch.ones(4, 1)
    lookup = {
        "CASES_RAW": 0,
        "DEATHS_RAW": 1,
        "CASES_SMOOTH": 2,
        "DEATHS_SMOOTH": 3,
    }
    norm_stats = {
        "CASES_RAW": (0.0, 1.0),
        "DEATHS_RAW": (2.0, 2.0),
        "CASES_SMOOTH": (3.0, 1.0),
        "DEATHS_SMOOTH": (2.0, 0.0),
    }
    expected = torch.tensor([[0.9999], [-0.5000], [-2.0000], [-100000.0]])
    result = CityConvModel._normalize_output(y, lookup, norm_stats, log_transform=False)
    assert result.shape == (4, 1)
    assert torch.allclose(result - expected, torch.zeros(4, 1), atol=1e-4)


def test_build_comparison_df():
    step_result_example = StepResult(
        city="city_name",
        task="some_task",
        date_t=pd.Timestamp("2020-01-10"),
        x_item=torch.rand(1),
        y_item=torch.rand(1),
        info_item={},
        head_results={
            "some_head": HeadTensors(
                input=torch.ones(7),
                target=torch.ones(5) * 2,
                prediction=torch.ones(5) * 1.5,
            )
        },
        city_output=torch.randn(1, 1),
        city_predicted_label=torch.tensor(1),
        stage=Stage.TRAIN,
        mc_results=None,
    )
    df = CityConvModel._build_comparison_df(
        city_results=[step_result_example],
        name="some_head",
        n_days_back=7,
        n_days_ahead=5,
    )
    expected_index = pd.date_range(pd.Timestamp("2020-01-04"), periods=5 + 7, freq="D")
    expected = pd.DataFrame(
        index=expected_index,
        data={
            "true": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ],
            "t+1": [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                1.5,
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            ],
            "t+2": [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                1.5,
                float("nan"),
                float("nan"),
                float("nan"),
            ],
            "t+3": [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                1.5,
                float("nan"),
                float("nan"),
            ],
            "t+4": [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                1.5,
                float("nan"),
            ],
            "t+5": [
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                1.5,
            ],
        },
    )
    # check that dfs match, nan == nan is True
    assert df.shape == expected.shape, f"Shapes differ: {df.shape} vs {expected.shape}"

    assert all(df.columns == expected.columns), "Columns differ"

    assert all(df.index == expected.index), "Indices differ"

    nan_diff = df.isna() != expected.isna()
    if nan_diff.any().any():
        print("NaN values are different at:")
        print(nan_diff[nan_diff.any(axis=1)])

    value_diff = df.fillna(0) != expected.fillna(0)
    if value_diff.any().any():
        print("Values are different at:")
        print(value_diff[value_diff.any(axis=1)])


def test_compute_losses_and_predictions(regression_city_conv_model):  # noqa: F811
    # 1. Setup Necessary Prerequisites
    model = regression_city_conv_model

    norm_y_hat_deltas = {
        "CASES,RAW": torch.ones(4) * 1,  # predicted gap 1
        "HOSPITALIZED,RAW": torch.ones(4) * 2,  # predicted gap 2
        "CASES,SMOOTH": torch.ones(4) * 3,  # predicted gap 3
        "HOSPITALIZED,SMOOTH": torch.ones(4) * 4,  # predicted gap 4
    }

    x_normalized = torch.tensor(
        [
            [3.0, 3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )  # mock tensor with appropriate shape
    y_targets_normalized = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],  # target gap -3, absolute err 1
            [1.0, 1.0, 1.0, 1.0],  # target gap -1, absolute err 1
            [2.0, 2.0, 2.0, 2.0],  # target gap 1, absolute err 1
            [3.0, 3.0, 3.0, 3.0],  # target gap 3, absolute err 1
        ]
    )

    y_norm_stats = {
        "CASES_RAW": (0.0, 1.0),
        "HOSPITALIZED_RAW": (0.0, 1.0),
        "CASES_SMOOTH": (0.0, 1.0),
        "HOSPITALIZED_SMOOTH": (0.0, 1.0),
    }
    input_indices = {
        "CASES_RAW": 0,
        "HOSPITALIZED_RAW": 1,
        "CASES_SMOOTH": 2,
        "HOSPITALIZED_SMOOTH": 3,
    }
    output_indices = {
        "CASES_RAW": 0,
        "HOSPITALIZED_RAW": 1,
        "CASES_SMOOTH": 2,
        "HOSPITALIZED_SMOOTH": 3,
    }
    city_classifier_loss = torch.tensor([2.0])

    use_delta = True  # Or False depending on what you want to test
    weighted_loss = False
    # 3. Run the Function
    head_losses, dico_y_hat = model._compute_losses_and_predictions(
        model_output_dico=norm_y_hat_deltas,
        x_normalized=x_normalized,
        y_targets_normalized=y_targets_normalized,
        y_normalization_stats=y_norm_stats,
        input_indices=input_indices,
        output_indices=output_indices,
        predicted_variables=model.predicted_variables,
        predicted_ts_types=model.predicted_ts_types,
        use_delta=use_delta,
        weighted_loss=weighted_loss,
        city_classifier_loss=city_classifier_loss,
    )
    expected_head_losses = {
        "CASES,RAW": torch.tensor(4.0),
        "HOSPITALIZED,RAW": torch.tensor(3.0),
        "CASES,SMOOTH": torch.tensor(2.0),  # (3 - 1)^2
        "HOSPITALIZED,SMOOTH": torch.tensor(1.0),  # (4 - 3)^2
        "city_classifier": torch.tensor(2.0),
        "total": torch.tensor(12.0),
    }
    expected_dico_y_hat = {
        "CASES,RAW": [torch.tensor([4.0, 4.0, 4.0, 4.0])],  # 3 + 1
        "HOSPITALIZED,RAW": [torch.tensor([4.0, 4.0, 4.0, 4.0])],  # 2 + 2
        "CASES,SMOOTH": [torch.tensor([4.0, 4.0, 4.0, 4.0])],  # 1 + 3
        "HOSPITALIZED,SMOOTH": [torch.tensor([4.0, 4.0, 4.0, 4.0])],  # 0 + 4
    }

    # 4. Assertions
    assert isinstance(head_losses, dict)
    assert "total" in head_losses
    assert isinstance(dico_y_hat, dict)
    assert all(
        [
            torch.allclose(head_losses[key], expected_head_losses[key])
            for key in head_losses
        ]
    )
    assert all(
        [
            torch.allclose(
                dico_y_hat[key][0], expected_dico_y_hat[key][0], rtol=1e-4, atol=1e-4
            )
            for key in dico_y_hat
        ]
    )

    use_delta = False
    weighted_loss = False
    head_losses, dico_y_hat = model._compute_losses_and_predictions(
        model_output_dico=norm_y_hat_deltas,
        x_normalized=x_normalized,
        y_targets_normalized=y_targets_normalized,
        y_normalization_stats=y_norm_stats,
        input_indices=input_indices,
        output_indices=output_indices,
        predicted_variables=model.predicted_variables,
        predicted_ts_types=model.predicted_ts_types,
        use_delta=use_delta,
        weighted_loss=weighted_loss,
        city_classifier_loss=city_classifier_loss,
    )
    expected_head_losses = {
        "CASES,RAW": torch.tensor(1.0),  # (0 - 1)^2
        "HOSPITALIZED,RAW": torch.tensor(1.0),  # (1 - 2)^2
        "CASES,SMOOTH": torch.tensor(1.0),  # (2 - 3)^2
        "HOSPITALIZED,SMOOTH": torch.tensor(1.0),  # (3 - 4)^2
        "city_classifier": torch.tensor(2.0),
        "total": torch.tensor(6.0),
    }
    expected_dico_y_hat = {
        "CASES,RAW": [torch.tensor([1.0, 1.0, 1.0, 1.0])],  # y_hat denormalized
        "HOSPITALIZED,RAW": [torch.tensor([2.0, 2.0, 2.0, 2.0])],
        "CASES,SMOOTH": [torch.tensor([3.0, 3.0, 3.0, 3.0])],
        "HOSPITALIZED,SMOOTH": [torch.tensor([4.0, 4.0, 4.0, 4.0])],
    }
    assert all(
        [
            torch.allclose(head_losses[key], expected_head_losses[key])
            for key in head_losses
        ]
    )
    assert all(
        [
            torch.allclose(
                dico_y_hat[key][0], expected_dico_y_hat[key][0], rtol=1e-4, atol=1e-4
            )
            for key in dico_y_hat
        ]
    )

    use_delta = False
    weighted_loss = True
    head_losses, dico_y_hat = model._compute_losses_and_predictions(
        model_output_dico=norm_y_hat_deltas,
        x_normalized=x_normalized,
        y_targets_normalized=y_targets_normalized,
        y_normalization_stats=y_norm_stats,
        input_indices=input_indices,
        output_indices=output_indices,
        predicted_variables=model.predicted_variables,
        predicted_ts_types=model.predicted_ts_types,
        use_delta=use_delta,
        weighted_loss=weighted_loss,
        city_classifier_loss=city_classifier_loss,
    )
    expected_head_losses = {
        "CASES,RAW": torch.tensor(1.5),  # (0 - 1)^2 * torch.linspace(1, 4 / 2, 4)
        "HOSPITALIZED,RAW": torch.tensor(1.5),  # (1 - 2)^2
        "CASES,SMOOTH": torch.tensor(1.5),  # (2 - 3)^2
        "HOSPITALIZED,SMOOTH": torch.tensor(1.5),  # (3 - 4)^2
        "city_classifier": torch.tensor(2.0),
        "total": torch.tensor(8.0),
    }
    expected_dico_y_hat = {
        "CASES,RAW": [torch.tensor([1.0, 1.0, 1.0, 1.0])],  # y_hat denormalized
        "HOSPITALIZED,RAW": [torch.tensor([2.0, 2.0, 2.0, 2.0])],
        "CASES,SMOOTH": [torch.tensor([3.0, 3.0, 3.0, 3.0])],
        "HOSPITALIZED,SMOOTH": [torch.tensor([4.0, 4.0, 4.0, 4.0])],
    }
    assert all(
        [
            torch.allclose(head_losses[key], expected_head_losses[key])
            for key in head_losses
        ]
    )
    assert all(
        [
            torch.allclose(
                dico_y_hat[key][0], expected_dico_y_hat[key][0], rtol=1e-4, atol=1e-4
            )
            for key in dico_y_hat
        ]
    )

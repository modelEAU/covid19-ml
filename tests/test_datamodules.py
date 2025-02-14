import sys

import pytest
import torch

sys.path.append("src")
from fixtures import (  # noqa: E402 F401
    long_sample_citydataset_regression1,
    long_sample_citydataset_regression2,
    parquet_path_long,
    sample_classification_citydatamodule1,
    sample_classification_citydatamodule2,
    sample_regression_citydatamodule1,
    sample_regression_citydatamodule2,
    sample_tensor_config_regression,
)

from datamodules import get_batch_item_info, recombine_batch_item_info  # noqa: E402


def test_get_batch_item_info(sample_regression_citydatamodule1):  # noqa: F811
    datamodule = sample_regression_citydatamodule1
    datamodule.setup(stage="fit")
    info = list(iter(datamodule.train_dataset))[0][2]
    assert info == {
        "city": "EASYVILLE_1",
        "first_input_date": "2020-01-01",
        "last_input_date": "2020-01-05",
        "first_target_date": "2020-01-06",
        "last_target_date": "2020-01-09",
        "augmented": torch.tensor(False),
        "x_column_lookup": {
            "CASES_RAW": torch.tensor(0),
            "CASES_SMOOTH": torch.tensor(1),
            "HOSPITALIZED_RAW": torch.tensor(2),
            "HOSPITALIZED_SMOOTH": torch.tensor(3),
        },
        "y_column_lookup": {
            "BOD_TREND_CURVATURE": torch.tensor(0),
            "BOD_TREND_SLOPE": torch.tensor(1),
            "COD_TREND_CURVATURE": torch.tensor(2),
            "COD_TREND_SLOPE": torch.tensor(3),
        },
    }


def test_recombine_batch_item_info(sample_regression_citydatamodule1):  # noqa: F811
    datamodule = sample_regression_citydatamodule1
    datamodule.setup(stage="fit")
    _, _, info = next(iter(datamodule.val_dataloader()))
    info_1 = get_batch_item_info(info, 0)
    info_2 = get_batch_item_info(info, 1)
    recombined = recombine_batch_item_info([info_1, info_2])
    for key, value in recombined.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, info[key])
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v, info[key][k])
                else:
                    assert v == info[key][k]
        else:
            assert value == info[key]


# noqa: E402
def test_city_datamodule_regression(sample_regression_citydatamodule1):  # noqa: F811
    datamodule = sample_regression_citydatamodule1
    datamodule.setup(stage="fit")
    assert len(datamodule.train_dataset) == 17
    assert len(datamodule.val_dataset) == 6
    with pytest.raises(AttributeError):
        datamodule.test_dataset
    datamodule.setup(stage="test")
    assert len(datamodule.test_dataset) == 7

    assert datamodule.train_loader.batch_size == 2
    assert len(iter(datamodule.train_loader)) == 9
    assert len(iter(datamodule.val_loader)) == 3
    assert len(iter(datamodule.test_loader)) == 4

    assert datamodule.train_dataset[0][0].shape == (4, 5)
    assert datamodule.train_dataset[0][1].shape == (4, 4)

    assert datamodule.val_dataset[0][0].shape == (4, 5)
    assert datamodule.val_dataset[0][1].shape == (4, 4)

    assert datamodule.test_dataset[0][0].shape == (4, 5)
    assert datamodule.test_dataset[0][1].shape == (4, 4)

    assert datamodule.x_norm["CASES_RAW"] == (
        pytest.approx(17.5882358),
        pytest.approx(11.6499481),
    )

    assert datamodule.y_norm["BOD_TREND_CURVATURE"] == (
        pytest.approx(1.0),
        pytest.approx(0.0),
    )

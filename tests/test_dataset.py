import sys

import pytest

sys.path.append("src")
from datasets import CityDataSet  # noqa: E402
from fixtures import (  # noqa: E402 F401
    parquet_path_short,
    sample_tensor_config_regression,
)
from types_ml import City  # noqa: E402


def test_city_dataset(
    parquet_path_short, sample_tensor_config_regression  # noqa: F811
):
    dataset = CityDataSet(
        path=parquet_path_short,
        city=City.EASYVILLE_1,
        start_date="2020-01-05",
        end_date="2020-01-08",
        tensor_config=sample_tensor_config_regression,
    )
    assert len(dataset) == 4
    assert dataset[0][0].shape == (4, 5)
    assert dataset[0][1].shape == (4, 4)

    assert dataset.date_index("2020-01-05") == 0
    assert dataset.date_index("2020-01-08") == 3
    with pytest.raises(IndexError):
        dataset.date_index("2020-01-04")

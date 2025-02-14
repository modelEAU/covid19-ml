import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append("src")

from sklearn.linear_model import LinearRegression

from qualitative import (
    calculate_curvature,
    calculate_monotonic_classes,
    calculate_slope,
    calculate_triangular_classes,
    classify_monotonic_letter,
    classify_monotonic_numeric,
    classify_triangluar_number,
    classify_triangular_letter,
    datetimes_to_epoch_days,
    get_fitted_model,
    multi_smooth,
    prepare_features,
    prepare_variable,
    rolling_classification_denoise,
    rolling_curvature_model,
    rolling_slope_model,
)


def test_prepare_variable():
    variable = pd.Series([1, 2, 3])
    expected = np.array([[1], [2], [3]])
    np.testing.assert_allclose(prepare_variable(variable), expected)


def test_datetime_to_epoch_days():
    time = pd.Series(
        pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
        )
    )

    expected = pd.Series([18262, 18263, 18264, 18265, 18266])
    np.testing.assert_allclose(datetimes_to_epoch_days(time), expected)


def test_prepare_features():
    time = pd.Series(
        pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
        )
    )
    expected = np.array([[1], [1], [1], [1], [1]])
    np.testing.assert_allclose(prepare_features(time, 0), expected)

    expected = np.array([[18262], [18263], [18264], [18265], [18266]])
    np.testing.assert_allclose(prepare_features(time, 1), expected)

    expected = np.array(
        [
            [18262, 333500644],
            [18263, 333537169],
            [18264, 333573696],
            [18265, 333610225],
            [18266, 333646756],
        ]
    )
    np.testing.assert_allclose(prepare_features(time, 2), expected)


def test_get_fitted_model():
    # Create a test dataset
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([4, 8, 12])

    # Test fitting a linear regression model
    model = get_fitted_model(X, y)
    assert isinstance(model, LinearRegression)
    assert np.allclose(model.coef_, [1.0, 1.0])
    assert np.allclose(model.intercept_, 1.0)

    X = np.array([[1], [3], [5]])
    y = np.array([3, 7, 11])
    model = get_fitted_model(X, y)
    assert isinstance(model, LinearRegression)
    assert np.allclose(model.coef_, [2.0])
    assert np.allclose(model.intercept_, 1.0)


def test_rolling_curvature_model():
    # Create a test dataset
    data = {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],
    }
    df = pd.DataFrame(data["y"], index=data["x"])

    # Test rolling curvature model with an invalid subset
    N = 3
    subset = df.iloc[:N]
    curvature = rolling_curvature_model(subset)
    assert isinstance(curvature, float)
    assert np.isclose(curvature, 0.0)

    # Test rolling curvature model with a valid subset that has a curvature
    N = 5
    subset = df.iloc[:5]
    curvature = rolling_curvature_model(subset)
    assert isinstance(curvature, float)
    assert np.isclose(curvature, 2.0)

    # Test rolling curvature model with a valid subset that has no curvature
    N = 5
    data = {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    }
    df = pd.DataFrame(data["y"], index=data["x"])
    subset = df.iloc[:N]
    curvature = rolling_curvature_model(subset)
    assert isinstance(curvature, float)
    assert curvature == 0.0


def test_rolling_slope():
    # Create a test dataset
    data = {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    }
    df = pd.DataFrame(data["y"], index=data["x"])

    # Test rolling slope model with an invalid subset
    N = 2
    subset = df.iloc[:N]
    slope = rolling_slope_model(subset)
    assert isinstance(slope, float)
    assert np.isclose(slope, 0.0)

    # Test rolling slope model with a valid subset that has a slope
    N = 5
    subset = df.iloc[:5]
    slope = rolling_slope_model(subset)
    assert isinstance(slope, float)
    assert np.isclose(slope, 2.0)

    # Test rolling slope model with a valid subset that has no slope
    N = 5
    data = {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [10, 20, 10, 20, 10, 20, 10, 20, 10, 20],
    }
    df = pd.DataFrame(data["y"], index=data["x"])
    subset = df.iloc[:N]
    slope = rolling_slope_model(subset)
    assert isinstance(slope, float)
    assert slope == 0.0


def test_rolling_classification_denoise():
    # Create a test dataset
    df = pd.Series(index=[1, 2, 3, 4, 5], data=[1, 1, 2, 2, 0])
    classification = rolling_classification_denoise(df)
    assert isinstance(classification, int)
    assert classification == 2

    df = pd.Series(index=[1, 2, 3, 4, 5], data=[0, 0, 2, 2, 1])
    classification = rolling_classification_denoise(df)
    assert isinstance(classification, int)
    assert classification == 1

    df = pd.Series(index=[1, 2, 3, 4, 5], data=[0, 0, 2, 2, 0])
    classification = rolling_classification_denoise(df)
    assert isinstance(classification, int)
    assert classification == 0

    # Test rolling classification denoise with an invalid subset
    df = pd.Series(index=[], data=[])
    # should raise a ValueError
    with pytest.raises(ValueError, match="Subset must not be empty"):
        classification = rolling_classification_denoise(df)

    df = pd.Series(index=[1, 2, 3], data=[1, 2, np.nan])
    # should raise a ValueError
    with pytest.raises(ValueError, match="Subset must not contain nan values"):
        classification = rolling_classification_denoise(df)

    df = pd.Series(index=[1, 2, 3, 4], data=[1, 2, 3, 4])
    # should raise a ValueError
    with pytest.raises(ValueError, match="Subset must have odd length"):
        classification = rolling_classification_denoise(df)


def test_multi_smooth():
    # Create a test dataset
    data = {
        "x": [1, 2, 3, 4, 5],
        "y": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame({"y": data["y"]}, index=data["x"])

    # Test multi_smooth with an invalid window
    with pytest.raises(ValueError):
        multi_smooth(df, "y", 0, 2)

    # Test multi_smooth with an invalid window
    with pytest.raises(ValueError, match="Window size must be odd"):
        multi_smooth(df, "y", 2, 2)

    # multi_smooth with 0 times should return the original series
    smooth = multi_smooth(df, "y", 3, 0)
    assert isinstance(smooth, pd.Series)
    assert smooth.name == "y"
    assert smooth.index.equals(df.index)
    assert smooth.equals(df["y"])

    expected = pd.Series(
        [
            1.269387755,
            1.909387755,
            3,
            4.090612245,
            4.730612245,
        ],
        index=data["x"],
    )
    # Test multi_smooth with a valid window and degree
    smooth = multi_smooth(df, "y", 5, 2)
    assert isinstance(smooth, pd.Series)
    assert smooth.name == "y"
    assert smooth.index.equals(df.index)
    assert np.allclose(smooth.to_numpy(), expected.to_numpy())


def test_classify_monotonic_letter():
    # Test positive slope
    letter = classify_monotonic_letter(1.0)
    assert isinstance(letter, str)
    assert letter == "U"

    # Test negative slope
    letter = classify_monotonic_letter(-1.0)
    assert isinstance(letter, str)
    assert letter == "D"

    # Test zero slope
    letter = classify_monotonic_letter(0.0)
    assert isinstance(letter, str)
    assert letter == "G"


def test_classify_monotonic_numeric():
    # Test positive slope
    classification = classify_monotonic_numeric(1.0)
    assert isinstance(classification, int)
    assert classification == 2

    # Test negative slope
    classification = classify_monotonic_numeric(-1.0)
    assert isinstance(classification, int)
    assert classification == 0

    # Test zero slope
    classification = classify_monotonic_numeric(0.0)
    assert isinstance(classification, int)
    assert classification == 1


def test_classify_triangular_letter():
    # Test number 6
    letter = classify_triangular_letter(6)
    assert isinstance(letter, str)
    assert letter == "D"

    # Test number 5
    letter = classify_triangular_letter(5)
    assert isinstance(letter, str)
    assert letter == "E"

    # Test number 4
    letter = classify_triangular_letter(4)
    assert isinstance(letter, str)
    assert letter == "A"

    # Test number 3
    letter = classify_triangular_letter(3)
    assert isinstance(letter, str)
    assert letter == "G"

    # Test number 2
    letter = classify_triangular_letter(2)
    assert isinstance(letter, str)
    assert letter == "C"

    # Test number 1
    letter = classify_triangular_letter(1)
    assert isinstance(letter, str)
    assert letter == "F"

    # Test number 0
    letter = classify_triangular_letter(0)
    assert isinstance(letter, str)
    assert letter == "B"


def test_classify_triangluar_number():
    # Test slope decreasing, curvature decreasing
    row = {"SLOPE": 0, "CURVATURE": 0}
    classification = classify_triangluar_number(row)
    assert isinstance(classification, int)
    assert classification == 0

    # Test slope decreasing, curvature stable
    row = {"SLOPE": 0, "CURVATURE": 1}
    classification = classify_triangluar_number(row)
    assert isinstance(classification, int)
    assert classification == 1

    # Test slope decreasing, curvature increasing
    row = {"SLOPE": 0, "CURVATURE": 2}
    classification = classify_triangluar_number(row)
    assert isinstance(classification, int)
    assert classification == 2

    # Test slope stable
    row = {"SLOPE": 1, "CURVATURE": 1}
    classification = classify_triangluar_number(row)
    assert isinstance(classification, int)
    assert classification == 3

    # Test slope increasing, curvature decreasing
    row = {"SLOPE": 2, "CURVATURE": 0}
    classification = classify_triangluar_number(row)
    assert isinstance(classification, int)
    assert classification == 4

    # Test slope increasing, curvature stable
    row = {"SLOPE": 2, "CURVATURE": 1}
    classification = classify_triangluar_number(row)
    assert isinstance(classification, int)
    assert classification == 5

    # Test slope increasing, curvature increasing
    row = {"SLOPE": 2, "CURVATURE": 2}
    classification = classify_triangluar_number(row)
    assert isinstance(classification, int)
    assert classification == 6

    # Test invalid curvature value
    row = {"SLOPE": 0, "CURVATURE": 3}
    try:
        classification = classify_triangluar_number(row)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


def test_calculate_slope():
    data = pd.DataFrame(
        index=[1, 2, 3, 4, 5],
        data={
            "y": [1, 4, 9, 16, 25],
        },
    )
    expected_slope = pd.Series(
        [np.nan, np.nan, 5.462204, np.nan, np.nan], index=data.index
    )
    slope = calculate_slope(data, "y", 5, 5)
    assert isinstance(slope, pd.Series)
    assert slope.name == "y"
    assert slope.index.equals(data.index)
    np.testing.assert_allclose(
        slope.to_numpy(), expected_slope.to_numpy(), equal_nan=True
    )


def test_calculate_curvature():
    data = pd.DataFrame(
        index=[1, 2, 3, 4, 5],
        data={
            "y": [1, 8, 27, 64, 125],
        },
    )
    expected_curvature_p0p5 = pd.Series(
        [np.nan, np.nan, 8.1043032, np.nan, np.nan], index=data.index
    )
    expected_curvature_p0p05 = pd.Series(
        [np.nan, np.nan, 0.0, np.nan, np.nan], index=data.index
    )
    slope1 = calculate_curvature(data, "y", 5, 5, alpha=0.5)
    slope2 = calculate_curvature(data, "y", 5, 5, alpha=0.05)
    assert isinstance(slope1, pd.Series)
    assert slope1.name == "y"
    assert slope1.index.equals(data.index)
    np.testing.assert_allclose(
        slope2.to_numpy(), expected_curvature_p0p05.to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(
        slope1.to_numpy(), expected_curvature_p0p5.to_numpy(), equal_nan=True
    )


def test_calculate_monotonic_classes():
    slope = pd.Series(
        [0.5, 1.0, 0, -1.0, -3.0],
        index=[1, 2, 3, 4, 5],
    )
    slope.name = "y"
    expected_trend_slope = pd.Series([2, 2, 1, 0, 0], index=slope.index)
    expected_trend_slope.name = "y"
    trend_slope = calculate_monotonic_classes(slope)
    assert isinstance(trend_slope, pd.Series)
    assert trend_slope.name == "y"
    assert trend_slope.index.equals(slope.index)
    np.testing.assert_allclose(
        trend_slope.to_numpy(), expected_trend_slope.to_numpy(), equal_nan=True
    )
    letter_mode = True
    expected_trend_slope = pd.Series(["U", "U", "G", "D", "D"], index=slope.index)
    trend_slope = calculate_monotonic_classes(slope, letter_mode)
    assert isinstance(trend_slope, pd.Series)
    assert trend_slope.name == "y"
    assert trend_slope.index.equals(slope.index)
    assert all(
        [x == y for x, y in zip(trend_slope.to_list(), expected_trend_slope.to_list())]
    )


def test_calculate_triangular_classes():
    slope = pd.Series(
        [0.5, 1.0, 1.0, 0, 0, 0, -1, -2.0, -3.0],
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    curvature = pd.Series(
        [0.5, 0, -1.0, 1.0, 0, -2.0, 0.6, 0.0, -2.0],
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    expected_triangular_classes = pd.Series(
        [6, 5, 4, 3, 3, 3, 2, 1, 0], index=slope.index
    )

    triangular_classes = calculate_triangular_classes(slope, curvature)
    assert isinstance(triangular_classes, pd.Series)

    assert triangular_classes.index.equals(slope.index)
    np.testing.assert_allclose(
        triangular_classes.to_numpy(),
        expected_triangular_classes.to_numpy(),
        equal_nan=True,
    )
    letter_mode = True
    expected_triangular_classes = pd.Series(
        ["D", "E", "A", "G", "G", "G", "C", "F", "B"], index=slope.index
    )

    triangular_classes = calculate_triangular_classes(slope, curvature, letter_mode)
    assert isinstance(triangular_classes, pd.Series)

    assert triangular_classes.index.equals(slope.index)
    assert all(
        [
            x == y
            for x, y in zip(
                triangular_classes.to_list(), expected_triangular_classes.to_list()
            )
        ]
    )

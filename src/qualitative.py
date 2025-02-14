import numpy as np
import pandas as pd
from scipy.signal import savgol_filter  # type: ignore
from scipy.stats import f  # type: ignore
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

MONOTONIC_CLASSES = {
    "D": 0,
    "G": 1,
    "U": 2,
}

MONOTONIC_CLASS_WEIGHTS = {
    "D": 0.333,
    "G": 0.333,
    "U": 0.333,
}

TRIANGULAR_CLASS_WEIGHTS = {
    "D": 0.143,
    "E": 0.143,
    "A": 0.143,
    "G": 0.142,
    "C": 0.143,
    "F": 0.143,
    "B": 0.143,
}

TRIANGULAR_CLASSES = {
    "D": 6,
    "E": 5,
    "A": 4,
    "G": 3,
    "C": 2,
    "F": 1,
    "B": 0,
}

MONOTONIC_CLASS_LOOKUP = {v: k for k, v in MONOTONIC_CLASSES.items()}

TRIANGULAR_CLASS_LOOKUP = {v: k for k, v in TRIANGULAR_CLASSES.items()}


def datetimes_to_epoch_days(datetimes: pd.Series) -> pd.Series:
    return datetimes.astype("int64") // (10**9 * 3600 * 24)


def prepare_features(time, degree):
    if time.dtype in ["datetime64[ns]", "<m8[ns]"]:
        time = datetimes_to_epoch_days(time)
    time = time.to_numpy().reshape(-1, 1)
    return PolynomialFeatures(degree=degree, include_bias=not degree).fit_transform(
        time
    )


def prepare_variable(variable):
    variable = variable.to_numpy().reshape(-1, 1)
    return variable


def get_fitted_model(features, variable):
    model = LinearRegression()
    model.fit(features, variable)
    return model


def rolling_curvature_model(subset, alpha=0.05):
    time = subset.index - subset.index[0]
    N = len(time)
    if N <= 3:
        return 0.0
    results = {}
    variable = prepare_variable(subset)
    for order in range(1, 3):
        features = prepare_features(time, order)
        model = get_fitted_model(features, variable)
        predicted = model.predict(features)
        SSR = sum((variable - predicted) ** 2)
        results[order] = {
            "model": model,
            "predicted": predicted,
            "SSR": SSR,
        }
    p2 = 3
    p1 = 2
    SSR_lin = results[1]["SSR"]
    SSR_para = results[2]["SSR"]
    F = (np.abs(SSR_lin - SSR_para) / (p2 - p1)) / (SSR_para / (N - p2))
    F_thresh = f.ppf(1 - alpha, p2 - p1, N - p2)
    quadratic_coeff = results[2]["model"].coef_[0][1]
    tentative_curvature = quadratic_coeff * 2
    return tentative_curvature if F > F_thresh else 0.0


def rolling_slope_model(subset, alpha=0.05):
    time = subset.index - subset.index[0]
    N = len(time)
    if N <= 2:
        return 0.0
    results = {}
    variable = prepare_variable(subset)

    for order in range(2):
        features = prepare_features(time, order)
        model = get_fitted_model(features, variable)
        predicted = model.predict(features)
        SSR = sum((variable - predicted) ** 2)
        results[order] = {
            "model": model,
            "predicted": predicted,
            "SSR": SSR,
        }
    p2 = 2
    p1 = 1
    SSR_flat = results[0]["SSR"]
    SSR_lin = results[1]["SSR"]
    F = (np.abs(SSR_flat - SSR_lin) / (p2 - p1)) / (SSR_lin / (N - p2))
    F_thresh = f.ppf(1 - alpha, p2 - p1, N - p2)
    tentative_slope = results[1]["model"].coef_[0][0]
    return tentative_slope if F > F_thresh else 0.0


def rolling_classification_denoise(subset):
    if subset.isna().any():
        raise ValueError("Subset must not contain nan values")
    if len(subset) == 0:
        raise ValueError("Subset must not be empty")
    if len(subset) % 2 == 0:
        raise ValueError("Subset must have odd length")
    average = subset.mean()
    if average > 1:
        return 2
    elif average < 1:
        return 0
    return 1


def multi_smooth(df, series_name, window_size, n_times):
    if n_times == 0:
        return df[series_name]
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    for _ in range(n_times):
        df[series_name] = savgol_filter(
            x=df[series_name].values,
            window_length=window_size,
            polyorder=3,
            mode="nearest",
        )
    return df[series_name]


def classify_monotonic_letter(slope: float) -> str:
    if slope > 0:
        return "U"
    elif slope < 0:
        return "D"
    else:
        return "G"


def classify_monotonic_numeric(slope: float) -> int:
    if slope > 0:
        return 2
    elif slope < 0:
        return 0
    else:
        return 1


def classify_triangular_letter(number: int) -> str:
    lookup = {
        6: "D",
        5: "E",
        4: "A",
        3: "G",
        2: "C",
        1: "F",
        0: "B",
    }
    return lookup[number]


def classify_triangluar_number(row):
    slope = row["SLOPE"]
    curvature = row["CURVATURE"]
    if slope == 0:  # slope is decreasing
        if curvature == 0:  # curvature is decreasing
            return 0
        elif curvature == 1:  # curvature is stable
            return 1
        elif curvature == 2:  # curvature is increasing
            return 2
        raise ValueError("Curvature must be 0, 1, or 2")
    elif slope == 1:  # slope is stable
        return 3
    elif slope == 2:  # slope is increasing
        if curvature == 0:  # curvature is decreasing
            return 4
        elif curvature == 1:  # curvature is stable
            return 5
        elif curvature == 2:  # curvature is increasing
            return 6
        raise ValueError("Curvature must be 0, 1, or 2")
    raise ValueError("Slope must be 0, 1, or 2")


def calculate_slope(df, column, smooth_window, model_window, alpha=0.05):
    smooth_for_slope = multi_smooth(df, column, smooth_window, 2)
    return smooth_for_slope.rolling(model_window, center=True).apply(
        rolling_slope_model, args=(alpha,)
    )


def calculate_curvature(df, column, smooth_window, model_window, alpha=0.05):
    smooth_for_curvature = multi_smooth(df, column, smooth_window, 3)
    return smooth_for_curvature.rolling(model_window, center=True).apply(
        rolling_curvature_model, args=(alpha,)
    )


def calculate_monotonic_classes(slope, letter_mode=False):
    slope = slope.copy()
    if letter_mode:
        return slope.map(classify_monotonic_letter).copy()
    else:
        trend_slope = slope.apply(classify_monotonic_numeric)
    return trend_slope


def calculate_triangular_classes(slope, curvature, letter_mode=False):
    monotonic_trend = calculate_monotonic_classes(slope, letter_mode=False)
    trend_curvature = curvature.map(
        classify_monotonic_numeric,
    )
    tri_df = pd.DataFrame(
        {
            "SLOPE": monotonic_trend,
            "CURVATURE": trend_curvature,
        },
        index=slope.index,
    )
    classification = tri_df.apply(classify_triangluar_number, axis=1)  # type: ignore
    if letter_mode:
        return classification.map(classify_triangular_letter)
    return classification

"""
This module contains functions for converting city-specific ODM wide datasets to tensors that can be used with the project's models.
Namely, it computes the derived variables, it creates the smoothed and differentiated time series, and it creates the tensors.
The main function is `to_tensor`, which takes a city and a task as input and returns a dictionary of tensors.
"""

import itertools
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from scipy.fft import fft, ifft
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import seasonal_decompose

import qualitative
from types_ml import City, Task, TensorConfig, TimeSeriesType, Variable

HEALTH_PREFIX = "CPHD-"

EvenNumber = int


def find_key(input_dict, value):
    """Return the first key in a dictionary that matches a given value, or None if no key is found."""

    return next((k for k, v in input_dict.items() if v == value), None)


DATASET_COL_NAMES = {
    "Calculated_timestamp": "TIME",
    "CPHD-conf_report_value": "CASES",
    "CPHD-death_report_value": "DEATHS",
    "CPHD-hospcen_report_value": "HOSPITALIZED",
    "CPHD-pposrt_report_value": "POS_TESTS_RATE",
    "CPHD-pctvaccinedose1_report_value": "PCT_VAX1",
    "CPHD-pctvaccinedose2_report_value": "PCT_VAX2",
    "SiteMeasure_wwbod5c_mgl_single-to-mean_value": "BOD",
    "SiteMeasure_wwcod_mgl_single-to-mean_value": "COD",
    "SiteMeasure_wwtss_mgl_single-to-mean_value": "TSS",
    "SiteMeasure_wwnh4n_mgl_single-to-mean_value": "NH4",
    "SiteMeasure_wwtemp_°c_single-to-mean_value": "WATER_TEMP",
    "SiteMeasure_wwtemp_degc_single-to-mean_value": "WATER_TEMP",
    "SiteMeasure_envrnf_mm_single-to-mean_value": "RAINFALL",
    # "SiteMeasure_envRnF_mm_single-to-mean_value": "RAINFALL",
    "SiteMeasure_wwflow_m3d_single-to-mean_value": "FLOW",
    "WWMeasure_covn1_gcml_single-to-mean_value": "COVN1",
}


HEALTH_NAMES_LOOKUP = {
    "CPHD-conf_report_value": "CASES",
    "CPHD-death_report_value": "DEATHS",
    "CPHD-hospcen_report_value": "HOSPITALIZED",
    "CPHD-pposrt_report_value": "POS_TESTS_RATE",
    "CPHD-postest_report_value": "POS_TESTS",
    "CPHD-test_report_value": "TESTS",
    "CPHD-pctvaccinedose1_report_value": "PCT_VAX1",
    "CPHD-pctvaccinedose2_report_value": "PCT_VAX2",
}


def shift_back(data, n):
    """
    Shift the rows of a 1D numpy array backwards by n positions, with the top rows filled with NaN values, creating a 2D array.
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 1:
        raise TypeError("data must be a 1D numpy array")
    if n >= data.shape[0]:
        raise ValueError("n must be less than the number of rows in the data array")

    shifted_matrices = np.empty((data.shape[0], n))

    shifted_matrices[:, -1] = data.T

    for i in range(n):
        shifted_data = np.roll(data, i, axis=0)
        shifted_data[:i] = np.nan
        shifted_matrices[:, n - i - 1] = shifted_data.T

    return shifted_matrices


def shift_forward(data: np.ndarray, n: int) -> np.ndarray:
    """
    Shift the rows of a 1D numpy array forwards by n positions, with the bottom rows filled with NaN values, creating a 2D array.
    """
    if not isinstance(data, np.ndarray) or len(data.shape) != 1:
        raise TypeError("data must be a 1D numpy array")
    if n >= data.shape[0]:
        raise ValueError("n must be less than the number of rows in the data array")

    data = data.astype(np.float32)
    shifted_matrices = np.empty((data.shape[0], n))

    for i in range(1, n + 1):
        shifted_data = np.roll(data, -i, axis=0)

        shifted_data[-i:] = np.nan
        shifted_matrices[:, i - 1] = shifted_data.T

    return shifted_matrices


def extract_poly_population(df: pd.DataFrame, poly_id: str, sewershed=False) -> float:
    """
    Extracts the population of a polygon from a DataFrame containing population data.

    Args:
        df (pd.DataFrame): The DataFrame containing population data.
        poly_id (str): The ID of the polygon whose population is to be extracted.
        sewershed (bool, optional): A flag indicating if the polygon is a sewershed. Default is False.

    Returns:
        float: The population of the polygon.

    Raises:
        ValueError: If the population is not defined for the polygon, or is not constant for the whole time range.
    """
    prefix = "SewershedPoly-" if sewershed else "OverlappingPoly-"
    if sewershed:
        name = f"{prefix}Polygon_pop"
    else:
        name = f"{prefix}{poly_id}-Polygon_pop"
    try:
        series = df[name]
    except KeyError as e:
        raise KeyError(f"Could not locate column {name}") from e
    population_uniques = series.replace("", np.nan).dropna().unique()
    if len(population_uniques) == 0:
        raise ValueError("Population is not defined for the polygon")
    elif len(population_uniques) > 1:
        raise ValueError("Population is not constant for the whole time range")
    pop = float(population_uniques[0])
    if pd.isna(pop):
        raise ValueError("Population is not defined for the polygon")
    try:
        answer = float(pop)
    except ValueError as e:
        raise ValueError(f"Could not parse {pop} to a float") from e
    return float(answer)


def deseasonalize(df: pd.DataFrame, period: int = 7) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        # Use LOESS to remove weekly seasonality if it is stronger than the residuals
        stl_decompose = seasonal_decompose(df[col], period=period, model="additive")
        df[col] = stl_decompose.trend.bfill().ffill()
    return df


def high_pass_filter(
    df: pd.DataFrame, threshold: float = 0.05, col_names: Optional[list[str]] = None
) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        # if col_names and (col not in col_names):
        #     continue
        signal = df[col].bfill().ffill().values
        time = (df.index - df.index[0]).days
        fft_result = fft(signal)
        frequencies = np.fft.fftfreq(len(fft_result), d=(time[1] - time[0]))
        # amplitudes = np.abs(fft_result)  # Amplitude of each frequency component
        # phases = np.angle(fft_result)    # Phase of each frequency component
        threshold_frequency = threshold  # Define a frequency threshold
        fft_result[np.abs(frequencies) > threshold_frequency] = 0
        reconstructed_signal = ifft(fft_result).real
        df[col] = reconstructed_signal
    return df


def compute_smooth(df: pd.DataFrame, n_timesteps_back: int, times=1) -> pd.DataFrame:
    if n_timesteps_back % 2 == 0 or n_timesteps_back < 4:
        raise ValueError(
            "The window size for the Savitzky-Golay filter must be an odd integer greater than or equal to 3"
        )
    df = df.copy()
    for col in df.columns:
        for i in range(times):
            df[col] = savgol_filter(
                x=df[col].values,
                window_length=n_timesteps_back,
                polyorder=3,
                mode="nearest",
            )
            df[col] = df[col].bfill().ffill()

    df = df.add_suffix(f"_{TimeSeriesType.SMOOTH.value}")
    return df


def compute_slope(
    df: pd.DataFrame, smooth_window: int, model_window: int, alpha: float = 0.05
) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = (
            qualitative.calculate_slope(
                df, col, smooth_window, model_window, alpha=alpha
            )
            .bfill()
            .ffill()
        )

    df = df.add_suffix(f"_{TimeSeriesType.SLOPE.value}")
    return df


def compute_curvature(
    df: pd.DataFrame, smooth_window: int, model_window: int, alpha: float = 0.05
) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = (
            qualitative.calculate_curvature(
                df, col, smooth_window, model_window, alpha=alpha
            )
            .bfill()
            .ffill()
        )

    df = df.add_suffix(f"_{TimeSeriesType.CURVATURE.value}")
    return df


def compute_trend_slope(
    df: pd.DataFrame, smooth_window: int, model_window: int, alpha: float = 0.05
) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        slope = qualitative.calculate_slope(
            df, col, smooth_window, model_window, alpha=alpha
        )
        df[col] = qualitative.calculate_monotonic_classes(slope, letter_mode=False)

    df = df.add_suffix(f"_{TimeSeriesType.TREND_SLOPE.value}")
    return df


def compute_trend_curvature(
    df: pd.DataFrame,
    smooth_window: int,
    model_window: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        curvature = qualitative.calculate_curvature(
            df, col, smooth_window, model_window, alpha=alpha
        )
        df[col] = qualitative.calculate_monotonic_classes(curvature, letter_mode=False)

    df = df.add_suffix(f"_{TimeSeriesType.TREND_CURVATURE.value}")
    return df


def compute_ts_types(
    df: pd.DataFrame,
    ts_types: list[TimeSeriesType],
    smooth_window: int,
    model_window: int,
    alpha: float = 0.05,
    period: int = 7,
) -> pd.DataFrame:
    if ts_types is None or not ts_types:
        return df
    original_df = df.copy()
    df = df.copy()
    if period and len(df) > 2 * period:
        # df = deseasonalize(df, period=period)
        df = high_pass_filter(df, 0.03, list(HEALTH_NAMES_LOOKUP.values()))
    dfs = []
    for ts_type in ts_types:
        if ts_type == TimeSeriesType.RAW:
            raw_df = original_df.add_suffix(f"_{TimeSeriesType.RAW.value}")
            dfs.append(raw_df)

        elif ts_type == TimeSeriesType.SMOOTH:
            smooth_df = compute_smooth(df, smooth_window, times=1)
            dfs.append(smooth_df)

        elif ts_type == TimeSeriesType.SLOPE:
            slope_df = compute_slope(df, smooth_window, model_window, alpha=alpha)
            dfs.append(slope_df)
        elif ts_type == TimeSeriesType.CURVATURE:
            curvature_df = compute_curvature(
                df, smooth_window, model_window, alpha=alpha
            )
            dfs.append(curvature_df)
        elif ts_type == TimeSeriesType.TREND_SLOPE:
            slope_trend_df = compute_trend_slope(
                df, smooth_window, model_window, alpha=alpha
            )
            dfs.append(slope_trend_df)
        elif ts_type == TimeSeriesType.TREND_CURVATURE:
            curvature_df = compute_trend_curvature(
                df, smooth_window, model_window, alpha=alpha
            )
            dfs.append(curvature_df)

    df = pd.concat(dfs, axis=1)
    return df


def create_load_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns in the DataFrame based on a set of predefined formulas using existing columns.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with new columns added based on predefined formulas.
    """
    df = df.copy()
    cov_n1 = str(find_key(DATASET_COL_NAMES, Variable.COVN1.value))
    flow = str(find_key(DATASET_COL_NAMES, Variable.FLOW.value))
    # bod = str(find_key(DATASET_COL_NAMES, Variable.BOD.value))
    cod = str(find_key(DATASET_COL_NAMES, Variable.COD.value))
    nh4 = str(find_key(DATASET_COL_NAMES, Variable.NH4.value))
    tss = str(find_key(DATASET_COL_NAMES, Variable.TSS.value))

    df[Variable.N1_FLOW.value] = df[cov_n1] * df[flow]
    if cod in df.columns:
        df[Variable.COD_FLOW.value] = df[cod] * df[flow]
    # df[Variable.BOD_FLOW.value] = df[bod] * df[flow]
    df[Variable.NH4_FLOW.value] = df[nh4] * df[flow]
    df[Variable.TSS_FLOW.value] = df[tss] * df[flow]
    return df


def interpolate_values(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing values in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to interpolate. The index should be a datetime index

    Returns:
        pd.DataFrame: The input DataFrame with missing values filled in by interpolation.

    """
    df = df.copy()
    value_cols = [col for col in df.columns if "value" in col]
    # interpolate the values to fill the missing dates
    # in between the start and end dates
    for col in value_cols:
        if len(df[col].dropna()) > 3:
            df[col] = df[col].interpolate(method="polynomial", order=3)
        else:
            df[col] = df[col].interpolate(method="nearest")
    for col in value_cols:
        first_valid_index = df[col].first_valid_index()

        if first_valid_index != df.index[0]:
            df.loc[:first_valid_index, col] = df[col][:first_valid_index].fillna(  # type: ignore
                df.loc[first_valid_index, col]  # type: ignore
            )
    for col in value_cols:
        last_valid_index = df[col].last_valid_index()

        if last_valid_index != len(df[col]) - 1:
            df.loc[last_valid_index:, col] = df[col][last_valid_index:].fillna(  # type: ignore
                df.loc[last_valid_index, col]  # type: ignore
            )
    return df


def compute_centreau_health(df: pd.DataFrame, city: City):
    df = df.copy()
    site_health_regions = {
        City.QUEBEC_EAST: "prov_qc_hlthreg_capitale_nationale",
        City.QUEBEC_WEST: "prov_qc_hlthreg_capitale_nationale",
        City.MONTREAL_NORTH: "prov_qc_hlthreg_montreal",
        City.MONTREAL_SOUTH: "prov_qc_hlthreg_montreal",
        City.LAVAL_AUTEUIL: "prov_qc_hlthreg_laval",
        City.LAVAL_FABREVILLE: "prov_qc_hlthreg_laval",
        City.LAVAL_LAPINIERE: "prov_qc_hlthreg_laval",
        City.GATINEAU: "prov_qc_hlthreg_outaouais",
    }
    health_region_id = site_health_regions[city]
    population = extract_poly_population(
        df, health_region_id, sewershed="sw" in health_region_id
    )

    health_cols = [
        col
        for col in df.columns
        if f"{HEALTH_PREFIX}{health_region_id}" in col and "value" in col
    ]
    for col in health_cols:
        var_type = "_".join(col.split("_")[-3:])
        new_name = HEALTH_NAMES_LOOKUP[f"{HEALTH_PREFIX}{var_type}"]
        # turn the values into a rate per 100k people
        if "death" in col or "hosp" in col or "conf" in col or "postest" in col:
            df[new_name] = 1e5 * (df[col] / population).astype(np.float64)

        else:
            df[new_name] = df[col].astype(np.float64)
    return df


def compute_stpaul_health(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # apply the preprocessing for St. Paul
    heatlh_regions_to_add = [
        "hennepin_county_hlthreg",
        "anoka_county_hlthreg",
        "dakota_county_hlthreg",
        "ramsey_county_hlthreg",
        "washington_county_hlthreg",
        "metro_plant_sw",
    ]
    var_pop: dict[str, float] = {}
    polygon_pop: dict[str, float] = {
        health_region_id: extract_poly_population(
            df, health_region_id, sewershed="sw" in health_region_id
        )
        for health_region_id in heatlh_regions_to_add
    }
    health_vars: dict["str", list[pd.Series]] = {}

    for health_region_id, population in polygon_pop.items():
        health_cols = [
            col
            for col in df.columns
            if f"{HEALTH_PREFIX}{health_region_id}" in col and "value" in col
        ]

        for col in health_cols:
            df[col] = df[col].astype(np.float64)
            var_type = "_".join(col.split("_")[-3:])
            if var_type not in var_pop.keys():
                var_pop[var_type] = 0
            var_pop[var_type] += population
            if var_type not in health_vars.keys():
                health_vars[var_type] = []
            # turn the values into a rate per 100k people
            if "death" in col or "hosp" in col or "conf" in col or "postest" in col:
                df[col] = 1e5 * df[col] / population
            health_vars[var_type].append((df[col] * population).astype(np.float64))
    for var_type, series_list in health_vars.items():
        to_sum_df = pd.concat(series_list, axis=1)
        new_name = HEALTH_NAMES_LOOKUP[f"{HEALTH_PREFIX}{var_type}"]
        df[new_name] = (to_sum_df.sum(axis=1) / var_pop[var_type]).astype(np.float64)
        # df[new_name] = (to_sum_df.sum(axis=1)).astype(np.float64)
    df = df.sort_index(axis=1)
    return df


def compute_health_variables(df: pd.DataFrame, city: City) -> pd.DataFrame:
    """Compute health variables for the given DataFrame. Differenrt cities have different health variables, therefore the computations vary slightly.

    Arguments:
        df {pd.DataFrame} -- The input DataFrame from a given city
        city {City} -- The city to compute health variables for.

    Returns:
        pd.DataFrame -- The input DataFrame with health variables added.
    """
    df = df.copy()

    if city == City.STPAUL:
        df = compute_stpaul_health(df)
    elif city in [
        City.QUEBEC_EAST,
        City.QUEBEC_WEST,
        City.MONTREAL_NORTH,
        City.MONTREAL_SOUTH,
        City.LAVAL_AUTEUIL,
        City.LAVAL_FABREVILLE,
        City.LAVAL_LAPINIERE,
        City.GATINEAU,
    ]:
        df = compute_centreau_health(df, city)

    return df


def get_variable_rank(
    variables: list[Variable], ts_types: list[TimeSeriesType]
) -> dict[tuple[Variable, TimeSeriesType], int]:
    """Returns a dictionary where the keys are tuples of `(Variable, TimeSeriesType)` and the values are the rank of each tuple in alphebically sorted list of all possible pairs.

    Args:
        variables: A list of `Variable` objects to consider.
        ts_types: A list of `TimeSeriesType` objects to consider.

    """
    result: dict[tuple[Variable, TimeSeriesType], int] = {}
    names: list[str] = [
        f"{variable.value}_{ts_type.value}"
        for variable, ts_type in itertools.product(variables, ts_types)
    ]
    names.sort()
    for variable in variables:
        for ts_type in ts_types:
            name = f"{variable.value}_{ts_type.value}"
            result[(variable, ts_type)] = names.index(name)
    return result


def to_tensor(
    parquet_path: str | Path,
    city: City,
    start_date: str,
    end_date: str,
    tensor_config: TensorConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Converts a parquet file to a tuple of tensors `(X, y, info)` where `X` is the input tensor, `y` is the target tensor and 'info' is a dictionary containing relevant information about the X,y pair."""

    input_variables = tensor_config.input_variables
    target_variables = tensor_config.target_variables
    input_ts_types = tensor_config.input_ts_types
    target_ts_types = tensor_config.target_ts_types
    n_timesteps_back = tensor_config.n_timesteps_back
    n_timesteps_forward = tensor_config.n_timesteps_forward

    task = tensor_config.task

    if input_variables is None:
        input_variables = []
    if target_variables is None:
        target_variables = []
    if input_ts_types is None:
        input_ts_types = [TimeSeriesType.RAW]
    if target_ts_types is None:
        target_ts_types = [TimeSeriesType.RAW]

    # load the parquet file
    file_df = pd.read_parquet(parquet_path)
    file_df = file_df.rename(
        columns={
            "SiteMeasure_wwtemp_°c_single-to-mean_value": "SiteMeasure_wwtemp_degc_single-to-mean_value"
        }
    )
    # # rename the columns to be more readable
    # if "SiteMeasure_envrnf_mm_single-to-mean_value" not in file_df.columns:
    #     raise ValueError("No rain")
    fake_data = tensor_config.days_before != -1
    noise = tensor_config.artificial_noise

    if fake_data:
        file_df["WWMeasure_covn1_gcml_single-to-mean_value"] = (
            file_df["CPHD-conf_report_value"].shift(-tensor_config.days_before) / 1000
        )
        if tensor_config.days_before:
            file_df = file_df.iloc[: -tensor_config.days_before]
    if noise:
        file_df["noise"] = np.random.rand(len(file_df))
        file_df["noise"] = file_df["noise"].apply(lambda x: 1 + np.log(x))
        file_df["WWMeasure_covn1_gcml_single-to-mean_value"] = (
            file_df["WWMeasure_covn1_gcml_single-to-mean_value"]
            + file_df["WWMeasure_covn1_gcml_single-to-mean_value"]
            * 0.35
            * file_df["noise"]
        )
    if tensor_config.insert_dummy_variable:
        file_df[Variable.DUMMY.value] = 1
        if Variable.DUMMY not in input_variables:
            input_variables.append(Variable.DUMMY)

    # clip the data to the desired time range
    start_clip_date = pd.to_datetime(start_date) - pd.to_timedelta(
        n_timesteps_back - 1, unit="D"
    )
    end_clip_date = pd.to_datetime(end_date) + pd.to_timedelta(
        n_timesteps_forward, unit="D"
    )
    df = (
        file_df.sort_index()[start_clip_date:end_clip_date]
        .asfreq("D")
        .dropna(axis=1, how="all")
    )  # type: ignore
    df = interpolate_values(df)

    if city not in [
        City.EASYVILLE_1,
        City.EASYVILLE_2,
    ]:
        df = create_load_variables(df)
    df = compute_health_variables(df, city)

    df = df.rename(columns=DATASET_COL_NAMES)

    # remove unwanted variables
    input_df = df[[var.value for var in input_variables]].astype(np.float32)
    target_df = df[[var.value for var in target_variables]].astype(np.float32)

    smooth_window = (
        n_timesteps_back if n_timesteps_back % 2 == 1 else n_timesteps_back + 1
    )
    # create the time series types that are asked for
    input_df = compute_ts_types(
        input_df,
        input_ts_types,
        smooth_window=smooth_window,
        model_window=tensor_config.trend_model_window,
    )
    input_df = input_df.sort_index(axis=1)
    input_cols = list(input_df.columns)

    for col in input_cols:
        if "RAW" in col or "SMOOTH" in col:
            input_df[col] = input_df[col].clip(lower=0)
    input_column_lookup = {col: i for i, col in enumerate(input_cols)}
    input_tensor = torch.empty(
        (len(input_df), len(input_column_lookup), n_timesteps_back)
    )
    for i, col in enumerate(input_column_lookup):
        input_tensor[:, i, :] = torch.tensor(
            shift_back(input_df[col].astype(np.float32).values, n_timesteps_back),
            dtype=torch.float32,
        )

    target_df = compute_ts_types(
        target_df,
        target_ts_types,
        smooth_window=smooth_window,
        model_window=tensor_config.trend_model_window,
    )  # type: ignore
    target_df = target_df.sort_index(axis=1)
    for col in target_df.columns:
        if "RAW" in col or "SMOOTH" in col:
            target_df[col] = target_df[col].clip(lower=0)
    target_column_lookup = {col: i for i, col in enumerate(target_df.columns)}
    target_tensor = torch.empty(
        (len(target_df), len(target_df.columns), n_timesteps_forward),
        dtype=torch.float32 if task == Task.REGRESSION else torch.long,
    )
    for i, col in enumerate(target_df.columns):
        target_tensor[:, i, :] = torch.tensor(
            shift_forward(target_df[col].values, n_timesteps_forward)  # type: ignore
        )

    # create a list of info dictionaries for each timestep
    info_dicts: list[dict[str, Any]] = []
    for i in range(n_timesteps_back - 1, len(input_df) - n_timesteps_forward):
        info_dict = {
            "first_input_date": pd.to_datetime(input_df.index[i - n_timesteps_back + 1])
            .date()
            .strftime("%Y-%m-%d")
        }
        info_dict["last_input_date"] = (
            pd.to_datetime(input_df.index[i]).date().strftime("%Y-%m-%d")
        )  # type: ignore
        info_dict["first_target_date"] = (
            pd.to_datetime(target_df.index[i] + pd.to_timedelta(1, unit="D"))
            .date()
            .strftime("%Y-%m-%d")
        )  # type: ignore
        info_dict["last_target_date"] = (
            pd.to_datetime(target_df.index[i + n_timesteps_forward])
            .date()
            .strftime("%Y-%m-%d")
        )  # type: ignore

        info_dict["city"] = city.value
        info_dict["augmented"] = False
        info_dict["x_column_lookup"] = input_column_lookup
        info_dict["y_column_lookup"] = target_column_lookup

        info_dicts.append(info_dict)
    X = input_tensor[n_timesteps_back - 1 : -n_timesteps_forward, ...]
    y = target_tensor[n_timesteps_back - 1 : -n_timesteps_forward, ...]

    # make sure that X and y don't have any nan values
    if torch.isnan(X).any() or torch.isnan(y).any():
        raise ValueError("X or y contain NaN values")
    return X, y, info_dicts

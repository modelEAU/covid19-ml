import numpy as np
import pandas as pd
import torch
from types_ml import StepResult


def extract_true_values_series_from_results_list(
    results: list[StepResult], n_days_back, name
):
    true_values = pd.Series(dtype=float)
    for i, result in enumerate(results):
        date_t = result.date_t
        if i == 0:
            first_date = date_t - pd.Timedelta(days=n_days_back - 1)
            values = result.head_results[name].input
            date_range = pd.date_range(start=first_date, end=date_t, freq="D")
            true_values = pd.Series(values, index=date_range)  # type: ignore
        else:
            true_values[date_t] = result.head_results[name].input[-1].numpy().item()
    return true_values.sort_index()


def extract_true_values_from_results_list_matched(results: list[StepResult], name):
    """matched because it outputs as many values as there are results"""
    true_values = pd.Series(dtype=float)
    for _, result in enumerate(results):
        date_t = result.date_t
        true_values[date_t] = result.head_results[name].input[-1].numpy().item()
    return true_values.sort_index()


def extract_true_values_matrix_from_results_list_matched(
    results: list[StepResult], name
):
    """matched because it outputs as many values as there are results"""
    true_values = []
    for _, result in enumerate(results):
        true_values.append(result.head_results[name].input.numpy().flatten())
    return np.array(true_values)


def extract_target_values_df_from_results_list(
    results: list[StepResult], n_days_forward, name, use_delta
):
    extracted_rows = {}
    for result in results:
        date_t = result.date_t
        extracted_rows[date_t] = result.head_results[name].target.numpy()
        if use_delta:
            last_known_value = result.head_results[name].input[-1].numpy().item()
            extracted_rows[date_t] = extracted_rows[date_t] - last_known_value
    df = pd.DataFrame(extracted_rows).T.sort_index()
    df.columns = [f"t+{i+1}" for i in range(n_days_forward)]
    return pd.DataFrame(extracted_rows).T.sort_index()


def extract_predictions_values_df_from_results_list(
    results: list[StepResult], n_days_forward, name, use_delta
):
    extracted_rows = {}
    for result in results:
        date_t = result.date_t
        values = result.head_results[name].prediction.numpy()

        if use_delta:
            last_known_value = result.head_results[name].input[-1].numpy().item()
            values = values - last_known_value

        extracted_rows[date_t] = values
    df = pd.DataFrame(extracted_rows).T.sort_index()
    df.columns = [f"t+{i+1}" for i in range(n_days_forward)]
    return pd.DataFrame(extracted_rows).T.sort_index()


def extract_prediction_from_results_list_by_horizon(
    results: list[StepResult],
    horizon,
    name,
    full_mat=False,
    monte_carlo=False,
):
    predictions = {}

    for result in results:
        date_t = result.date_t
        date_for_idx = date_t + pd.Timedelta(days=horizon)

        if monte_carlo:
            values = torch.vstack(
                [x[0][name][0][horizon - 1] for x in result.mc_results]
            ).numpy()
            values = values.reshape(-1)
        else:
            values = result.head_results[name].prediction[horizon - 1].numpy().item()
        predictions[date_for_idx] = values
    if full_mat:
        predictions = pd.DataFrame(predictions).sort_index().T
    else:
        if monte_carlo:
            #
            predictions = pd.DataFrame.from_dict(predictions, orient="index")
        else:
            predictions = pd.Series(predictions, dtype=float).sort_index()
    return predictions


def extract_prediction_from_results_list_mc_by_horizon(
    results: list[StepResult], horizon, name
) -> pd.DataFrame:
    predictions_mean = {}
    predictions_std = {}
    for result in results:
        if result.mc_results is None:
            continue
        date_t = result.date_t
        date_for_idx = date_t + pd.Timedelta(days=horizon)

        values = np.array(
            [
                result.mc_results[i][0][name][0][horizon - 1].numpy().item()
                for i in range(len(result.mc_results))
            ]
        )

        value_mean = values.mean()
        value_std = values.std()
        predictions_mean[date_for_idx] = value_mean
        predictions_std[date_for_idx] = value_std
    predictions_mean = pd.Series(predictions_mean, dtype=float).sort_index()
    predictions_mean.name = f"t+{horizon}_mc_mean"
    predictions_std = pd.Series(predictions_std, dtype=float).sort_index()
    predictions_std.name = f"t+{horizon}_mc_std"
    return pd.concat([predictions_mean, predictions_std], axis=1)


def extract_target_series_from_results_list_by_horizon(
    results: list[StepResult], horizon, name
):
    predictions_series = pd.Series(dtype=float)
    for result in results:
        date_t = result.date_t
        date_for_idx = date_t + pd.Timedelta(days=horizon)
        values = result.head_results[name].target[horizon - 1].numpy().item()
        predictions_series[date_for_idx] = values
    return predictions_series.sort_index()

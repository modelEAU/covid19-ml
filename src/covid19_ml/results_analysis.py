import os
import pickle
import sqlite3
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    PearsonCorrCoef,
    Precision,
    R2Score,
    Recall,
)
from torchmetrics.functional import (
    accuracy,
    auroc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    precision,
    r2_score,
    recall,
)

from covid19_ml.model_results import (
    extract_prediction_from_results_list_by_horizon,
    extract_target_series_from_results_list_by_horizon,
    extract_true_values_from_results_list_matched,
)
from covid19_ml.types_ml import (
    City,
    ClassificationMetric,
    RegressionMetric,
    SavedResults,
    Stage,
    StepResult,
    Task,
    TimeSeriesType,
    Variable,
)


@torch.no_grad()
def compute_linear_regression(y_known: torch.Tensor, n_predictions: int):
    x_known = torch.arange(len(y_known)).reshape(-1, 1).to(torch.float32)
    x_pred = (
        torch.arange(
            len(y_known),
            len(y_known) + n_predictions,
        )
        .reshape(-1, 1)
        .to(torch.float32)
    )

    lin_model = LinearRegression().fit(x_known.numpy(), y_known.numpy())
    return (
        torch.tensor(lin_model.predict(x_pred.numpy())).reshape(1, -1).to(torch.float32)
    )


@torch.no_grad()
def compute_lin_regression_array_from_result_list_by_horizon(
    results: list[StepResult], horizon, name
):
    prediction_list = []
    for result in results:
        known_values = result.head_results[name].input.flatten()
        yhat_lin = compute_linear_regression(known_values, horizon).flatten()
        prediction_list.append(yhat_lin[-1].item())
    return np.array(prediction_list)


@torch.no_grad()
def clip_classification_output(output: torch.Tensor):
    output[output > 2] = 2.0
    output[output < 0] = 0.0
    return output


@torch.no_grad()
def ml_lin_ratio(lin_predictions, ml_predictions, targets, metric_func):
    # Then you calculate the norm for lin and the norm for the model
    lin_norm = metric_func(lin_predictions, targets)
    ml_norm, ml_unc = calculate_metric_and_bootstrap_uncertainty(
        metric_func, ml_predictions, targets
    )
    ml_norm = torch.tensor([ml_norm])
    lin_norm = torch.tensor([lin_norm])
    return 1 - ml_norm / lin_norm, ml_unc / lin_norm


@torch.no_grad()
def compute_ratio(
    lin_score: float, ml_score: tuple[float, float]
) -> tuple[float, float]:
    return ml_score[0] / (lin_score + 1e-6), ml_score[1] / (lin_score + 1e-6)


@torch.no_grad()
def compute_naive(
    task: Task,
    city_results: list[StepResult],
    variable: Variable,
    ts_type: TimeSeriesType,
    metric: str,
    horizon: int | str = "all",
) -> float:
    if task != Task.REGRESSION:
        raise NotImplementedError("Only regression task is supported")
    if not city_results:
        return 0.0
    head_name = ",".join([variable.value, ts_type.value])
    n_preds = city_results[0].head_results[head_name].prediction.shape[-1]
    horizons = [horizon] if isinstance(horizon, int) else range(1, n_preds + 1)
    horizon_mean_scores = []  # either one long or n horizons long

    for horizon in horizons:
        # you need:
        # the linear regression prediction array for the horizon (steps x 1)
        lin_predictions = torch.tensor(
            compute_lin_regression_array_from_result_list_by_horizon(
                city_results, horizon, head_name
            )
        )
        # the target values array for the horizon (step x 1)
        targets = torch.tensor(
            extract_target_series_from_results_list_by_horizon(
                city_results, horizon, head_name
            ).values
        )
        len_targets = targets.shape[0]
        len_lin = lin_predictions.shape[0]
        smallest = min(len_lin, len_targets)
        targets = targets[: smallest - 1]
        lin_predictions = lin_predictions[: smallest - 1]
        value = metric_lookup[metric](lin_predictions, targets)
        horizon_mean_scores.append(value)

    horizon_mean_scores = torch.tensor(horizon_mean_scores)

    return horizon_mean_scores.mean().item()


def compile_regression_metrics(
    city: str,
    name: str,
    horizon: int,
    true_tensor: torch.Tensor,
    predicted_tensor: torch.Tensor,
):
    return {
        f"{city}_{name}_horizon={horizon}_{RegressionMetric.MSE.value}": MeanSquaredError(
            squared=True
        )(predicted_tensor, true_tensor),
        f"{city}_{name}_horizon={horizon}_{RegressionMetric.RMSE.value}": MeanSquaredError(
            squared=False
        )(predicted_tensor, true_tensor),
        f"{city}_{name}_horizon={horizon}_{RegressionMetric.MAE.value}": MeanAbsoluteError()(
            predicted_tensor, true_tensor
        ),
        f"{city}_{name}_horizon={horizon}_{RegressionMetric.NSE.value}": metrics.nse_loss(
            predicted_tensor, true_tensor
        ),
        f"{city}_{name}_horizon={horizon}_{RegressionMetric.R2.value}": (
            R2Score()(predicted_tensor, true_tensor) if len(true_tensor) > 1 else 0
        ),
        f"{city}_{name}_horizon={horizon}_{RegressionMetric.PEARSON.value}": (
            PearsonCorrCoef()(predicted_tensor, true_tensor)
            if len(true_tensor) > 1
            else 0
        ),
    }


def compile_classification_metrics(
    city: str,
    name: str,
    horizon: int,
    true_tensor: torch.Tensor,
    predicted_tensor: torch.Tensor,
    num_classes: int,
    class_idx: int,
):
    return {
        f"{city}_{name}_horizon={horizon}_{ClassificationMetric.ACCURACY.value}_class={class_idx}": Accuracy(  # type: ignore
            task="multiclass",
            num_classes=num_classes,
            average=None,
        )(predicted_tensor, true_tensor)[class_idx],
        f"{city}_{name}_horizon={horizon}_{ClassificationMetric.PRECISION.value}_class={class_idx}": Precision(  # type: ignore
            task="multiclass",
            num_classes=num_classes,
            average=None,
        )(predicted_tensor, true_tensor)[class_idx],
        f"{city}_{name}_horizon={horizon}_{ClassificationMetric.RECALL.value}_class={class_idx}": Recall(  # type: ignore
            task="multiclass",
            num_classes=num_classes,
            average=None,
        )(predicted_tensor, true_tensor)[class_idx],
        f"{city}_{name}_horizon={horizon}_{ClassificationMetric.F1SCORE.value}_class={class_idx}": F1Score(  # type: ignore
            task="multiclass",
            num_classes=num_classes,
            average=None,
        )(predicted_tensor, true_tensor)[class_idx],
        f"{city}_{name}_horizon={horizon}_{ClassificationMetric.AUROC.value}_class={class_idx}": AUROC(  # type: ignore
            task="multiclass",
            num_classes=num_classes,
            average=None,
        )(predicted_tensor, true_tensor)[class_idx],
    }


def compile_city_classification_metrics(
    true_tensor: torch.Tensor,
    predicted_tensor: torch.Tensor,
    city_lookup: dict[str, int],
    stage: str,
):
    num_classes = len(city_lookup)
    metrics = {}

    metrics[f"CityClassifier_{ClassificationMetric.ACCURACY.value}_{stage}"] = Accuracy(  # type: ignore
        task="multiclass",
        num_classes=num_classes,
    )(predicted_tensor, true_tensor).item()
    return metrics


def symmetric_log(x):
    return torch.log10(torch.abs(x) + 1)


# same but for the classificaiton metrics
# import the functional for of the metrics

metric_lookup = {
    RegressionMetric.MSE: lambda preds, targets: torch.mean((preds - targets) ** 2),
    RegressionMetric.MSSE: lambda preds, targets: torch.mean(
        ((preds - targets) / (targets + 1e-6)) ** 2
    ),
    RegressionMetric.RMSE: lambda preds, targets: torch.sqrt(
        torch.mean((preds - targets) ** 2)
    ),
    RegressionMetric.RMSSE: lambda preds, targets: torch.sqrt(
        torch.mean(((preds - targets) / (targets + 1e-6)) ** 2)
    ),
    RegressionMetric.MAE: lambda preds, targets: torch.mean(torch.abs(preds - targets)),
    RegressionMetric.MASE: lambda preds, targets: torch.mean(
        torch.abs((preds - targets) / (targets + 1e-6))
    ),
    RegressionMetric.SignAgreement: lambda preds, targets: torch.mean(
        (torch.sign(preds) == torch.sign(targets)).to(torch.float32)
    ),
    RegressionMetric.SSLE: lambda preds, targets: torch.mean(
        symmetric_log(((preds - targets) / (targets + 1e-6)))
    ),
    ClassificationMetric.ACCURACY: accuracy,
    ClassificationMetric.PRECISION: precision,
    ClassificationMetric.RECALL: recall,
    ClassificationMetric.F1SCORE: f1_score,
    ClassificationMetric.AUROC: auroc,
}


def calculate_metric_and_bootstrap_uncertainty(
    metric_fn, preds, targets, num_bootstrap=100
):
    metric_values = []
    n = len(preds)

    for _ in range(num_bootstrap):
        # Bootstrap sampling
        indices = np.random.choice(range(n), n, replace=True)
        sampled_preds = preds[indices]
        sampled_targets = targets[indices]

        # Compute the metric
        metric_value = metric_fn(sampled_preds, sampled_targets)
        metric_values.append(metric_value.item())

    # Calculate mean and standard deviation of the metric values
    mean_value = np.mean(metric_values)
    std_value = np.std(metric_values) / np.sqrt(n)

    return mean_value, std_value


def compute_metric(
    results: list[StepResult],
    variable: Variable,
    time_series_type: TimeSeriesType,
    horizon: int,
    metric: RegressionMetric | ClassificationMetric,
    monte_carlo: bool = False,
) -> float:
    name = ",".join([variable.value, time_series_type.value])
    target = torch.tensor(
        extract_target_series_from_results_list_by_horizon(
            results, horizon, name
        ).values
    )

    predicted = torch.tensor(
        extract_prediction_from_results_list_by_horizon(
            results,
            horizon,
            name,
            full_mat=False,
            monte_carlo=monte_carlo,
        ).values
    )
    if metric == RegressionMetric.SignAgreement:
        last_true_values = extract_true_values_from_results_list_matched(
            results, name
        ).values
        target = target - last_true_values
        predicted = predicted - last_true_values.reshape(-1, 1)

    if monte_carlo:
        # Calculate the metric for each Monte Carlo simulation
        n_examples, max_monte_carlos = predicted.shape
        metric_means = []
        metric_stds = []
        for i in range(n_examples):
            preds_sims = predicted[i]
            n_non_nan = torch.sum(~torch.isnan(preds_sims))
            predicted_value = preds_sims[:n_non_nan]
            # Assume metric_lookup[metric] can handle batched inputs
            target_repeated = target[i].expand(n_non_nan, *target[i].shape)
            metric_value, metric_unc = calculate_metric_and_bootstrap_uncertainty(
                metric_lookup[metric], predicted_value, target_repeated
            )
            metric_means.append(metric_value)
            metric_stds.append(metric_unc)

        metric_means = torch.tensor(metric_means)
        metric_stds = torch.tensor(metric_stds)

        agg_mean, agg_err = calculate_metric_and_bootstrap_uncertainty(
            lambda x, _: torch.mean(x), metric_means, metric_means
        )

        return agg_mean, agg_err

    return metric_lookup[metric](predicted, target).mean().item(), 0  # type: ignore


def convert_color_to_rgb(color: str) -> str:
    string = "rgb("
    for i in range(3):
        string += str(int(color[i] * 255))
        if i != 2:
            string += ","
    string += ")"
    return string


def closest_pair(tensor_a, tensor_b):
    expanded_a = tensor_a.unsqueeze(1)
    expanded_b = tensor_b.unsqueeze(0)
    distances = torch.norm(expanded_a - expanded_b, dim=2)
    min_distance_idx = torch.argmin(distances)
    i, j = divmod(min_distance_idx.item(), distances.size(1))
    return i, j, distances[i, j].item()


def pairwise_distance(matrix_a, matrix_b):
    expanded_a = matrix_a[:, np.newaxis, :]  # shape becomes (a, 1, n)
    expanded_b = matrix_b[np.newaxis, :, :]  # shape becomes (1, b, n)

    # Compute pairwise distances
    differences = expanded_a - expanded_b
    distances = np.linalg.norm(differences, axis=2)  # shape becomes (a, b)

    return distances


def create_distances_dico(city_coords_dico):
    distances_dico = {}
    for city_a, coords_dico_list_a in city_coords_dico.items():
        for city_b, coords_dico_list_b in city_coords_dico.items():
            if city_a == city_b:
                continue
            if (city_b, city_a) in distances_dico:
                continue
            coords_a = np.vstack([dico["coordinates"] for dico in coords_dico_list_a])
            coords_b = np.vstack([dico["coordinates"] for dico in coords_dico_list_b])
            distances = pairwise_distance(coords_a, coords_b)
            distances_dico[(city_a, city_b)] = distances
    return distances_dico


def find_closest_examples_between_cities(results_by_city, city_distances_dico):
    min_distances = {}
    for city_a, city_b in city_distances_dico.keys():
        distances_matrix = city_distances_dico[(city_a, city_b)]
        min_distance = np.min(distances_matrix)
        min_distances[(city_a, city_b)] = min_distance
        # find i, j such that distances[i, j] == min_distance using argmin
        i, j = np.unravel_index(np.argmin(distances_matrix), distances_matrix.shape)
        min_distances[(city_a, city_b)] = {
            city_a: results_by_city[city_a][i],
            city_b: results_by_city[city_b][j],
            "min_distance": min_distance,
        }
    return min_distances


def min_distance_between_cities(dico_of_tensors):
    min_distances = {}
    for city_a, tensor_a in dico_of_tensors.items():
        for city_b, tensor_b in dico_of_tensors.items():
            if city_a == city_b:
                continue
            if (city_b, city_a) in min_distances:
                continue
            min_distances[(city_a, city_b)] = closest_pair(tensor_a, tensor_b)
    return min_distances


class Purpose(Enum):
    FAKE_BASELINE = "FAKE_BASELINE"  # no ww at all
    INTERCITY_MID_PRED = "INTERCITY_MID_PRED"  # no ww at all
    INTERCITY_MID_PRED_WEST = "INTERCITY_MID_PRED_WEST"  # no ww at all
    INTERCITY_MID_PRED_REAL = "INTERCITY_MID_PRED_REAL"  # no ww at all
    FAKE_CLEAN_LAG = "FAKE_CLEAN_LAG"
    FAKE_NOISY_LAG = "FAKE_NOISY_LAG"
    FAKE_WEM_ADV_RATIO = "FAKE_WEM_ADV_RATIO"
    FAKE_WEST_ADV_RATIO = "FAKE_WEST_ADV_RATIO"
    FAKE_WEST = "FAKE_WEST"
    REAL_WW_BASELINE = "REAL_WW_BASELINE"
    REAL_WW_RECIPES = "REAL_WW_RECIPES"
    LOSS_TARGET = "LOSS_TARGET"
    OPTIMIZER = "OPTIMIZER"
    BATCH_SIZE = "BATCH_SIZE"
    DEBUG = "DEBUG"
    ACTIVATION = "ACTIVATION"
    DROPOUT = "DROPOUT"
    MID_PRED_LHS2 = "MID_PRED_LHS2"
    LOG = "LOG"
    MID_PRED_BEST = "MID_PRED_BEST"
    MID_PRED_BEST2 = "MID_PRED_BEST2"
    LHS_STPAUL = "LHS_STPAUL"
    LHS_ENCODERS = "LHS_ENCODERS"
    RECIPES_REALCITIES2 = "RECIPES_REALCITIES2"  # stpaul unseen
    RECIPES_REALCITIES3 = "RECIPES_REALCITIES3"  # gatineau unseen
    RECIPES_REALCITIES4 = "RECIPES_REALCITIES4"  # stpaul unseen no FT
    RECIPES_REALCITIES5 = "RECIPES_REALCITIES5"  # gatineau unseen no FT
    RECIPES_REALCITIES6 = "RECIPES_REALCITIES6"  # no encoders st paul unseen
    RECIPES_REALCITIES7 = "RECIPES_REALCITIES7"  # no encoders gatineau unseen
    BEST_TRANSFER = "BEST_TRANSFER"


def get_file_date(file_path):
    return datetime.fromtimestamp(os.path.getctime(file_path))


def collect_results(results):
    collected_results = {}
    for stage in Stage:
        stage_results = [
            result for result in results.complete_tests if result.stage == stage
        ]
        if not stage_results:
            continue
        collected_results[stage] = {}
        for city in City:
            city_results = [
                result for result in stage_results if result.city == city.value
            ]
            if not city_results:
                continue
            collected_results[stage][city] = {}
            for variable in Variable:
                for ts_type in TimeSeriesType:
                    head_name = ",".join([variable.value, ts_type.value])
                    if head_name in city_results[0].head_results:
                        if variable not in collected_results[stage][city]:
                            collected_results[stage][city][variable] = {}
                        collected_results[stage][city][variable][ts_type] = city_results
    return collected_results


def get_horizon_length(city_results, head_name):
    return city_results[0].head_results[head_name].target.shape[-1]


def compute_value(
    task, city_results, horizon, metric, variable, ts_type, monte_carlo=False
):
    if horizon == "all":
        means, stds = [], []
        for i in range(1, 8):
            value = compute_metric(
                city_results,
                variable,
                ts_type,
                i,
                metric,
                monte_carlo=monte_carlo,
            )
            means.append(value[0])
            stds.append(value[1])
        mean = torch.tensor(means).mean().item()
        std = torch.sqrt((torch.tensor(stds) ** 2).mean()).item() / np.sqrt(7)
        return mean, std
    else:
        return compute_metric(
            city_results,
            variable,
            ts_type,
            horizon,
            metric,
            monte_carlo=monte_carlo,
        )


def create_record(
    file,
    stage,
    results,
    city,
    variable,
    ts_type,
    horizon,
    metric,
    is_wave_only,
    ml_value,
    lin_value,
    file_date,
    purpose,
    model_type,
    train_classification,
    train_prediction,
    use_identity_for_city_heads,
    loss_fn,
    batch_size,
    target_type,
    weighted_loss,
    pooling_type,
    dropout_rate,
    activation,
    leaking_rate,
    optimizer_type,
    learning_rate,
    momentum,
    weight_decay,
    learning_rate_decay,
    n_middle_channels_city,
    n_out_channels_city,
    n_layers_city,
    city_pooling_padding,
    city_pooling_kernel_size,
    city_pooling_stride,
    city_pooling_dilation,
    city_kernel_size,
    city_conv_padding,
    city_conv_dilation,
    city_conv_stride,
    n_middle_channels_middle,
    n_out_channels_middle,
    n_layers_middle,
    middle_pooling_padding,
    middle_pooling_kernel_size,
    middle_pooling_stride,
    middle_pooling_dilation,
    middle_kernel_size,
    middle_conv_padding,
    middle_conv_dilation,
    middle_conv_stride,
    n_middle_channels_prediction,
    n_out_channels_prediction,
    n_layers_prediction,
    prediction_kernel_size,
    prediction_pooling_padding,
    prediction_pooling_kernel_size,
    prediction_pooling_stride,
    prediction_pooling_dilation,
    prediction_conv_padding,
    prediction_conv_dilation,
    prediction_conv_stride,
    log_transform,
    entry_nn_middle,
    entry_nn_prediction,
    classifier_n_layers,
    classifier_n_hidden,
):
    if isinstance(ml_value, tuple):
        value, error = ml_value
    else:
        error = 0.0
    return {
        "File": file,
        "Stage": stage.value,
        "Model": results.model_name,
        "Recipe": results.recipe,
        "City": city.value,
        "Variable": variable.value,
        "Time Series Type": ts_type.value,
        "Horizon": str(horizon),
        "Metric": str(metric).split(".")[-1],
        "Is Wave Only": is_wave_only,
        "Model Value": value,
        "Naive Value": lin_value,
        "Error": error,
        "Experiment Date": file_date,
        "Experiment Purpose": purpose.value,
        "Classifier Regularization Factor": results.classifier_regularization,
        "Backwards shift of case signal (d)": results.fake_ww_shift,
        "Artificial noise": results.artificial_noise,
        "created_on": datetime.now(),
        # hyperparameters
        "model_type": model_type,
        "train_classification": train_classification,
        "train_prediction": train_prediction,
        "use_identity_for_city_heads": use_identity_for_city_heads,
        "loss_fn": loss_fn,
        "batch_size": batch_size,
        "target_type": target_type,
        "weighted_loss": weighted_loss,
        "pooling_type": pooling_type,
        "dropout_rate": dropout_rate,
        "activation": activation,
        "leaking_rate": leaking_rate,
        "optimizer_type": optimizer_type,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "learning_rate_decay": learning_rate_decay,
        "n_middle_channels_city": n_middle_channels_city,
        "n_out_channels_city": n_out_channels_city,
        "n_layers_city": n_layers_city,
        "city_pooling_padding": city_pooling_padding,
        "city_pooling_kernel_size": city_pooling_kernel_size,
        "city_pooling_stride": city_pooling_stride,
        "city_pooling_dilation": city_pooling_dilation,
        "city_kernel_size": city_kernel_size,
        "city_conv_padding": city_conv_padding,
        "city_conv_dilation": city_conv_dilation,
        "city_conv_stride": city_conv_stride,
        "n_middle_channels_middle": n_middle_channels_middle,
        "n_out_channels_middle": n_out_channels_middle,
        "n_layers_middle": n_layers_middle,
        "middle_pooling_padding": middle_pooling_padding,
        "middle_pooling_kernel_size": middle_pooling_kernel_size,
        "middle_pooling_stride": middle_pooling_stride,
        "middle_pooling_dilation": middle_pooling_dilation,
        "middle_kernel_size": middle_kernel_size,
        "middle_conv_padding": middle_conv_padding,
        "middle_conv_dilation": middle_conv_dilation,
        "middle_conv_stride": middle_conv_stride,
        "n_middle_channels_prediction": n_middle_channels_prediction,
        "n_out_channels_prediction": n_out_channels_prediction,
        "n_layers_prediction": n_layers_prediction,
        "prediction_kernel_size": prediction_kernel_size,
        "prediction_pooling_padding": prediction_pooling_padding,
        "prediction_pooling_kernel_size": prediction_pooling_kernel_size,
        "prediction_pooling_stride": prediction_pooling_stride,
        "prediction_pooling_dilation": prediction_pooling_dilation,
        "prediction_conv_padding": prediction_conv_padding,
        "prediction_conv_dilation": prediction_conv_dilation,
        "prediction_conv_stride": prediction_conv_stride,
        "log_transform": log_transform,
        "entry_nn_middle": entry_nn_middle,
        "entry_nn_prediction": entry_nn_prediction,
        "classifier_n_layers": classifier_n_layers,
        "classifier_n_hidden": classifier_n_hidden,
    }


def create_records_for_file(
    purpose, filename, results, file_date, task, collected_results
):
    records = []
    for stage, stage_data in collected_results.items():
        for city, city_data in stage_data.items():
            for variable, variable_data in city_data.items():
                for ts_type, city_results in variable_data.items():
                    head_name = ",".join([variable.value, ts_type.value])
                    print(f"processing {city} {variable} {ts_type}...")
                    horizon_length = get_horizon_length(city_results, head_name)
                    for horizon in list(range(1, horizon_length + 1)) + ["all"]:
                        if "is_wave" in city_results[0].info_item.keys():
                            has_waves = True
                        else:
                            has_waves = False
                        for is_wave_only in [False]:  # [True, False]:
                            # print(f"Is wave only: {is_wave_only}, horizon: {horizon}")
                            if is_wave_only and has_waves:
                                rez = [
                                    result
                                    for result in city_results
                                    if result.info_item["is_wave"]
                                ]
                            elif is_wave_only and not has_waves:
                                with open("peak_lookup", "rb") as f:
                                    PEAK_LOOKUP = pickle.load(f)
                                if city.value in PEAK_LOOKUP.keys():
                                    rez = [
                                        result
                                        for result in city_results
                                        if result.date_t in PEAK_LOOKUP[city.value]
                                    ]
                                else:
                                    raise KeyError(
                                        f"city {city.value} not in the lookup"
                                    )
                            else:
                                rez = city_results
                            for metric in [
                                # RegressionMetric.RMSSE,
                                # RegressionMetric.MASE,
                                RegressionMetric.SignAgreement,
                                # RegressionMetric.SSLE,
                                RegressionMetric.MAE,
                                # RegressionMetric.RMSE,
                            ]:
                                print(
                                    f"Is wave only: {is_wave_only}, horizon: {horizon}, metric: {metric}"
                                )
                                ml_value = compute_value(
                                    task,
                                    rez,
                                    horizon,
                                    metric,
                                    variable,
                                    ts_type,
                                    monte_carlo=True,
                                )
                                lin_value = compute_naive(
                                    task,
                                    rez,
                                    variable,
                                    ts_type,
                                    metric,
                                    horizon,
                                )

                                records.append(
                                    create_record(
                                        file=filename,
                                        stage=stage,
                                        results=results,
                                        city=city,
                                        variable=variable,
                                        ts_type=ts_type,
                                        horizon=horizon,
                                        metric=metric,
                                        is_wave_only=is_wave_only,
                                        ml_value=ml_value,
                                        lin_value=lin_value,
                                        file_date=file_date,
                                        purpose=purpose,
                                        model_type=results.model_type,
                                        train_classification=results.train_classification,
                                        train_prediction=results.train_prediction,
                                        use_identity_for_city_heads=results.use_identity_for_city_heads,
                                        loss_fn=results.loss_fn,
                                        batch_size=results.batch_size,
                                        target_type=results.target_type,
                                        weighted_loss=results.weighted_loss,
                                        pooling_type=results.pooling_type,
                                        dropout_rate=results.dropout_rate,
                                        activation=results.activation,
                                        leaking_rate=results.leaking_rate,
                                        optimizer_type=results.optimizer_type,
                                        learning_rate=results.learning_rate,
                                        momentum=results.momentum,
                                        weight_decay=results.weight_decay,
                                        learning_rate_decay=results.learning_rate_decay,
                                        n_middle_channels_city=results.n_middle_channels_city,
                                        n_out_channels_city=results.n_out_channels_city,
                                        n_layers_city=results.n_layers_city,
                                        city_pooling_padding=results.city_pooling_padding,
                                        city_pooling_kernel_size=results.city_pooling_kernel_size,
                                        city_pooling_stride=results.city_pooling_stride,
                                        city_pooling_dilation=results.city_pooling_dilation,
                                        city_kernel_size=results.city_kernel_size,
                                        city_conv_padding=results.city_conv_padding,
                                        city_conv_dilation=results.city_conv_dilation,
                                        city_conv_stride=results.city_conv_stride,
                                        n_middle_channels_middle=results.n_middle_channels_middle,
                                        n_out_channels_middle=results.n_out_channels_middle,
                                        n_layers_middle=results.n_layers_middle,
                                        middle_pooling_padding=results.middle_pooling_padding,
                                        middle_pooling_kernel_size=results.middle_pooling_kernel_size,
                                        middle_pooling_stride=results.middle_pooling_stride,
                                        middle_pooling_dilation=results.middle_pooling_dilation,
                                        middle_kernel_size=results.middle_kernel_size,
                                        middle_conv_padding=results.middle_conv_padding,
                                        middle_conv_dilation=results.middle_conv_dilation,
                                        middle_conv_stride=results.middle_conv_stride,
                                        n_middle_channels_prediction=results.n_middle_channels_prediction,
                                        n_out_channels_prediction=results.n_out_channels_prediction,
                                        n_layers_prediction=results.n_layers_prediction,
                                        prediction_kernel_size=results.prediction_kernel_size,
                                        prediction_pooling_padding=results.prediction_pooling_padding,
                                        prediction_pooling_kernel_size=results.prediction_pooling_kernel_size,
                                        prediction_pooling_stride=results.prediction_pooling_stride,
                                        prediction_pooling_dilation=results.prediction_pooling_dilation,
                                        prediction_conv_padding=results.prediction_conv_padding,
                                        prediction_conv_dilation=results.prediction_conv_dilation,
                                        prediction_conv_stride=results.prediction_conv_stride,
                                        log_transform=results.log_transform,
                                        entry_nn_middle=results.entry_nn_middle,
                                        entry_nn_prediction=results.entry_nn_prediction,
                                        classifier_n_layers=getattr(
                                            results, "classifier_n_layers", None
                                        ),
                                        classifier_n_hidden=getattr(
                                            results, "classifier_n_hidden", None
                                        ),
                                    )
                                )
    print(f"Created {len(records)} records")
    return records


def process_kde_stats(file, kde_stats):
    records = []
    if kde_stats is None or len(kde_stats) == 0:
        return None
    for epoch, stats_dict in kde_stats.items():
        records.append(
            {
                "File": file,
                "epoch": epoch,
                "stage": stats_dict["stage"],
                "training_mode": stats_dict["training_mode"],
                "pct_5": stats_dict["pct_5"],
                "pct_50": stats_dict["pct_50"],
                "pct_95": stats_dict["pct_95"],
            }
        )
    df = pd.DataFrame(records)
    return df


def process_files(FILES, directory, db_path):
    failed_files = []

    def handle_future(future, purpose, file):
        try:
            future.result()
        except Exception as e:
            print(f"Processing file {file} failed with error: {e}")
            traceback.print_exc()
            failed_files.append((purpose, file))

    with ProcessPoolExecutor() as executor:
        futures = []
        for purpose, file_dico in FILES.items():
            for file in file_dico.values():
                print(f"Submitting file {file} for processing...")
                future = executor.submit(
                    process_file, purpose, file, directory, db_path
                )
                futures.append((future, purpose, file))

        # Wait for all threads to complete and handle exceptions
        for future, purpose, file in futures:
            handle_future(future, purpose, file)

    # Log or handle the failed files as needed
    if failed_files:
        print("The following files failed during processing:")
        for purpose, file in failed_files:
            print(f"Purpose: {purpose}, File: {file}")

    # for purpose, file_dico in FILES.items():
    #     for file in file_dico.values():
    #         print(f"Submitting file {file} for processing...")
    #         process_file(purpose, file, directory, db_path)


def process_file(purpose, file, directory, db_path):
    print(f"processing file {file}...")
    file_path = os.path.join(directory, file)
    results = torch.load(file_path)
    if getattr(results, "recipe", None) is None:
        results.recipe = "None"
    file_date = get_file_date(file_path)
    task = results.task
    collected_results = collect_results(results)
    records = create_records_for_file(
        purpose, file, results, file_date, task, collected_results
    )
    df = pd.DataFrame(records)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("results", conn, if_exists="append", index=False)
    kde_stats = getattr(results, "kde_stats", None)
    kde_table = process_kde_stats(file, kde_stats)
    if kde_table is not None:
        for col in kde_table.columns:
            if kde_table[col].dtype == "object":
                kde_table[col] = kde_table[col].astype(str)
        with sqlite3.connect(db_path) as conn:
            kde_table.to_sql("kde_stats", conn, if_exists="append", index=False)
    return None

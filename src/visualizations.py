import datetime
import itertools
import warnings

import matplotlib.dates as mdates
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import torch
from matplotlib import cm, patches, ticker
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
from model_results import (
    extract_prediction_from_results_list_by_horizon,
    extract_predictions_values_df_from_results_list,
    extract_target_values_df_from_results_list,
    extract_true_values_series_from_results_list,
)
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from torchmetrics import ConfusionMatrix
from types_ml import StepResult, TimeSeriesType, Variable

plt.style.use("seaborn-v0_8-talk")
plt.rcParams["font.family"] = "Arial"

pio.templates.default = "plotly_white"
pl_colors = pc.qualitative.Plotly


def get_plotly_symbol_names(num_items: int) -> list:
    raw_symbols = SymbolValidator().values
    return raw_symbols[2::3][::4][:num_items]


def plot_45deg_per_day(
    target: np.ndarray, pred: np.ndarray, var_name: str, colors
) -> figure.Figure:
    """Adapt this function to plot the 45-degree plot for every horizon, which corresponds to a column in the target and pred arrays."""

    n_horizons = target.shape[1]
    n_cols = (n_horizons + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 8, 16))

    # Flatten the axes for easier indexing
    axes = axes.ravel()

    # Calculate global min and max for consistent scaling
    global_min = min(target.min(), pred.min())
    global_max = max(target.max(), pred.max())

    for horizon in range(n_horizons):
        horizon_target = target[:, horizon]
        horizon_pred = pred[:, horizon]

        ax = axes[horizon]
        ax.set_title(
            f"{var_name} t+{horizon+1}".replace(",SMOOTH", "").title(), fontsize=36
        )
        ax.set_xlabel("True", fontsize=28)
        ax.set_ylabel("Predicted", fontsize=28)
        ax.grid(True)
        ax.plot(
            [global_min, global_max],
            [global_min, global_max],
            color="black",
            linestyle="solid",
            marker=None,
            linewidth=1,
        )
        ax.plot(
            horizon_target,
            horizon_pred,
            marker="o",
            color=next(colors),
            linestyle="None",
        )

        # Set consistent axis limits
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

        # Adjust font sizes for tick labels
        ax.tick_params(axis="both", which="major", labelsize=24)

        # Use fewer tick labels
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Calculate and display R2 in the upper left corner
        r2 = r2_score(horizon_target, horizon_pred)
        ax.text(
            0.05,
            0.95,
            f"$R^2$: {r2:.2f}",
            transform=ax.transAxes,
            fontsize=28,
            verticalalignment="top",
        )

    # If n_horizons is odd, remove the last (unused) subplot
    if n_horizons % 2 != 0:
        fig.delaxes(axes[-1])

    # Adjust the layout with more breathing room
    fig.tight_layout(pad=6.0)

    return fig


def plot_in_out_target(
    var_name: str,
    inputs: np.ndarray,
    outputs: np.ndarray,
    targets: np.ndarray,
    city: str,
    last_input_date: str | pd.Timestamp,
    first_target_date: str | pd.Timestamp,
) -> figure.Figure:
    last_input_date = pd.to_datetime(last_input_date, format="%Y-%m-%d")
    first_target_date = pd.to_datetime(first_target_date, format="%Y-%m-%d")
    # Create date ranges for input, target, and prediction
    input_dates = pd.date_range(
        end=last_input_date,
        periods=inputs.shape[0],
        freq="D",
    )
    target_dates = pd.date_range(
        start=first_target_date,
        periods=targets.shape[0],
        freq="D",
    )
    pred_dates = target_dates.copy()
    # Create dataframes for input, target, and prediction
    input_df = pd.DataFrame({"Input": inputs}, index=input_dates)
    target_df = pd.DataFrame({"Target": targets}, index=target_dates)
    pred_df = pd.DataFrame({"Predictions": outputs}, index=pred_dates)
    concat_df = pd.concat([input_df, target_df, pred_df], axis=1)
    # Create plot using seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis="x", rotation=45, labelsize=16)
    sns.lineplot(data=concat_df, ax=ax)
    ax.set_title(f"{city} - {var_name}", fontsize=24)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(fontsize=16)
    return fig


def plot_confusion_matrices(
    results: list[StepResult], n_days
) -> dict[str, figure.Figure]:
    """Plot the confusion matrix for the predictions of a model."""

    if not results:
        raise ValueError("The list of results is empty.")
    cities = list({result.city for result in results})
    head_names = list(results[0].head_results.keys())
    n_heads = len(head_names)

    # initialize a figure with a row for each head, a column for each day in the prediction window
    plt.tight_layout()
    figures = {}
    for city in cities:
        # subplots in a gris of n_heads x n_days
        fig, axes = plt.subplots(n_heads, n_days, figsize=(n_days * 8, n_heads * 8))
        # if there is only one head, axes is a 1D array, so we need to reshape it
        if n_heads == 1:
            axes = axes.reshape(1, -1)

        city_results = [x for x in results if x.city == city]
        for i, name in enumerate(head_names):
            head_data_list = [result.head_results[name] for result in city_results]
            y_true = torch.vstack([head_data.target for head_data in head_data_list])
            y_pred = torch.vstack(
                [head_data.prediction.argmax(axis=1) for head_data in head_data_list]  # type: ignore
            )
            n_classes = head_data_list[0].prediction.shape[1]
            for j in range(n_days):
                # get the true and predicted values for the day j
                y_true_day = y_true[:, j]
                y_pred_day = y_pred[:, j]
                # compute the confusion matrix
                cm = ConfusionMatrix(task="multiclass", num_classes=n_classes)
                # plot the confusion matrix
                sns.heatmap(
                    cm(y_pred_day, y_true_day).numpy(),  # type: ignore
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=axes[i, j],
                    cbar=False,
                    square=True,
                    annot_kws={"size": 20},
                )
                # set the labels
                axes[i, j].set_xlabel("Predicted", fontsize=20)
                axes[i, j].set_ylabel("True", fontsize=20)
                axes[i, j].set_title(f"{name} Day t=t+{j+1}", fontsize=20)
                axes[i, j].tick_params(axis="both", which="major", labelsize=20)
        fig_name = city.replace("_", " ").title()
        figures[fig_name] = fig

    return figures


def plot_city_classifier_cm(
    predicted_city_labels: torch.Tensor,
    true_city_labels: torch.Tensor,
    city_lookup: dict[str, int],
) -> figure.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cm = ConfusionMatrix(task="multiclass", num_classes=len(city_lookup))
    # plot the confusion matrix
    sns.heatmap(
        cm(predicted_city_labels, true_city_labels).numpy(),  # type: ignore
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        square=True,
        annot_kws={"size": 20},
    )
    # set the labels
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("True", fontsize=20)
    ax.set_title("City Classification", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    return fig


def plot_city_classifier_distribution_tsne(
    city_outputs: np.ndarray,
    true_city_labels: np.ndarray,
    city_lookup: dict[str, int],
) -> figure.Figure:
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(city_outputs)) - 1
    )
    city_outputs = city_outputs.reshape(city_outputs.shape[0], -1)

    tsne_results = tsne.fit_transform(city_outputs)
    tsne_df = pd.DataFrame(tsne_results, columns=["x", "y"])
    tsne_df["city"] = true_city_labels
    city_map = {v: k for k, v in city_lookup.items()}
    tsne_df["city"] = tsne_df["city"].map(city_map)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cities = tsne_df["city"].values
    x = tsne_df["x"].values
    y = tsne_df["y"].values
    sns.scatterplot(
        x=x,
        y=y,
        hue=cities,
        palette="tab10",
        ax=ax,
        legend="full",
        alpha=0.7,
    )
    ax.set_title("TSNE analysis of City Encoder output features", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)
    # set axis labels size
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.legend(
        title="City of origin",
        fontsize=16,
        title_fontsize=16,
        loc="upper right",
    )
    fig.tight_layout()
    return fig


def plot_city_classifier_distribution_tsne2(
    city_outputs: np.ndarray,
    true_city_labels: np.ndarray,
    city_lookup: dict[str, int],
) -> figure.Figure:
    tsne = TSNE(n_components=2, perplexity=min(30, len(city_outputs) - 1))
    city_outputs = city_outputs.reshape(city_outputs.shape[0], -1)

    tsne_results = tsne.fit_transform(city_outputs)
    tsne_df = pd.DataFrame(tsne_results, columns=["x", "y"])
    tsne_df["city"] = true_city_labels
    city_map = {v: k for k, v in city_lookup.items()}
    tsne_df["city"] = tsne_df["city"].map(city_map)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Define a list of markers and ensure it's at least as long as the number of unique cities
    markers = ["o", "D", "X", "P", "H", "*", "s", "p", "^"]
    unique_cities = tsne_df["city"].unique()
    if len(unique_cities) > len(markers):
        print(
            "Warning: Not enough unique markers for each city, markers will be reused"
        )
    # colors = sns.color_palette("tab10", len(unique_cities))
    for i, city in enumerate(unique_cities):
        city_data = tsne_df[tsne_df["city"] == city]
        # Cycle through markers if there are more cities than markers
        marker = markers[i % len(markers)]
        city_name = " ".join(city.split("_")).title()
        for j in range(len(city_data)):
            # Change alpha based on the index, ensuring it's visible and capped at 1
            alpha = 0.5
            ax.scatter(
                city_data.iloc[j]["x"],
                city_data.iloc[j]["y"],
                marker=marker,
                edgecolors="black",
                color=pl_colors[i],
                alpha=alpha,
                label=city_name if j == 0 else "",
            )
    ax.set_title("TSNE analysis of City Encoders' output features", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=14)
    # set axis labels size
    ax.set_xlabel("Component 1 (-)", fontsize=16)
    ax.set_ylabel("Component 2 (-)", fontsize=16)
    ax.legend(
        title="City of origin",
        fontsize=14,
        title_fontsize=14,
        loc="upper right",
    )
    fig.tight_layout()
    return fig


def plot_time_series_with_monte_carlo_bounds_plotly(
    mc_comparison_df: pd.DataFrame,
    name: str,
    city: str,
    max_horizon: int,
    use_log: bool,
) -> go.Figure:
    fig = go.Figure()
    # plot the true values
    fig.add_trace(
        go.Scatter(
            x=mc_comparison_df.index,
            y=mc_comparison_df["true"],
            mode="markers",
            name="Targets",
            marker=dict(
                color="DarkSlateGrey",
                size=4,
                symbol="diamond",
                line=dict(width=0, color="DarkSlateGrey"),
            ),
        )
    )
    color_cycle = get_color_cycle(
        num_colors=2 * max_horizon, palette="twilight_shifted"
    )
    colors1 = [next(color_cycle) for _ in range(max_horizon)]
    colors = [convert_color_to_rgb(color) for color in colors1]
    # turn rgb string to rgba string
    transparent_colors = []
    for color in colors:
        triplet_str = color.split(")")[0][4:]
        transparent_colors.append(f"rgba({triplet_str}, 0.4)")

    # plot the mean predictions for each horizon
    for horizon in range(1, max_horizon + 1):
        if horizon not in [1, 4, 7]:
            continue
        upper_bound = (
            1.96 * mc_comparison_df[f"t+{horizon}_mc_std"]
            + mc_comparison_df[f"t+{horizon}_mc_mean"]
        )
        lower_bound = (
            mc_comparison_df[f"t+{horizon}_mc_mean"]
            - 1.96 * mc_comparison_df[f"t+{horizon}_mc_std"]
        )
        mean_mc = mc_comparison_df[f"t+{horizon}_mc_mean"]

        if use_log:
            upper_bound += 1
            lower_bound += 1
            mean_mc += 1

        # Mean prediction
        fig.add_trace(
            go.Scatter(
                x=mc_comparison_df.index,
                y=mean_mc,
                mode="lines",
                name=f"t+{horizon} prediction",
                line=dict(color=colors[horizon - 1]),
            )
        )

        # Upper boundary (start of shaded area)
        fig.add_trace(
            go.Scatter(
                x=mc_comparison_df.index,
                y=upper_bound,
                mode="lines",
                name=f"t+{horizon} Upper Bound",
                line=dict(width=0),
                showlegend=False,
            )
        )

        # Lower boundary (end of shaded area)
        fig.add_trace(
            go.Scatter(
                x=mc_comparison_df.index,
                y=lower_bound,
                mode="lines",
                name=f"t+{horizon} bounds",
                fill="tonexty",  # This fills the area between this trace and the one before it
                fillcolor=transparent_colors[horizon - 1],  # Adjust opacity as needed
                line=dict(width=0),
                showlegend=True,
            )
        )
    fig.update_layout(
        title=f"{city} - {name}".split(",")[0].replace("_", " ").title(),
        xaxis_title="Date",
        yaxis_title=name.split(",")[0].title(),
        xaxis=dict(
            gridcolor="lightgrey",
            showgrid=True,
            # tickmode="auto",  # Adjust based on your data density
            nticks=18,
            tickformat="%Y-%m-%d",
            tickangle=-55,
        ),
        yaxis=dict(
            gridcolor="lightgrey",
            showgrid=True,
        ),
        # make the y axis log scale
        yaxis_type="log" if use_log else "linear",
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
            font=dict(size=11),
        )
    )
    fig.write_image(
        "/Users/jeandavidt/Developer/jeandavidt/cov19ml/plotly_mc_figs/mc_"
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%s")
        + ".png",
        scale=2,
    )
    return fig


def plot_montecarlo_dropout(
    predicted_variable: Variable,
    predicted_ts_type: TimeSeriesType,
    var_inputs: np.ndarray,
    var_targets: np.ndarray,
    var_mc_preds: np.ndarray,
    city: str,
    last_input_date: str | pd.Timestamp,
    first_target_date: str | pd.Timestamp,
) -> figure.Figure:
    # Compute mean and std of the predictions
    preds_mean = np.mean(var_mc_preds, axis=0)
    preds_std = np.std(var_mc_preds, axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.set_style("whitegrid")

    name = f"{predicted_variable.value},{predicted_ts_type.value}"

    var_input = var_inputs.flatten()
    var_target = var_targets.flatten()
    var_pred = preds_mean.flatten()
    var_std = preds_std.flatten()

    # Create date ranges for input, target, and prediction
    input_dates = pd.date_range(
        end=last_input_date,
        periods=len(var_input),
        freq="D",
    )
    target_dates = pd.date_range(
        start=first_target_date,
        periods=len(var_target),
        freq="D",
    )
    # store time series in a data frame
    input_df = pd.DataFrame({"Input": var_input}, index=input_dates)
    target_df = pd.DataFrame({"Target": var_target}, index=target_dates)
    pred_df = pd.DataFrame({"Predictions": var_pred}, index=target_dates)
    concat_df = pd.concat([input_df, target_df, pred_df], axis=1)

    # Create plot using seaborn
    ax.tick_params(axis="both", which="major", labelsize=16)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    sns.lineplot(data=concat_df, ax=ax)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Value", fontsize=16)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Adjust x-axis tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        date_fmt = DateFormatter("%y/%m/%d")
        # Set date format for x-axis ticks
        ax.xaxis.set_major_formatter(date_fmt)
    # add light major and minor grid lines
    ax.grid(which="major", color="#666666", linestyle="-", alpha=0.2)

    ax.fill_between(
        target_dates,
        var_pred - var_std,
        var_pred + var_std,
        alpha=0.5,
        label="uncertainty",
    )
    ax.set_title(f"MC dropout for {city} {name}, t0={last_input_date}")
    ax.legend(fontsize=13)
    return fig


def plot_time_series(
    true: np.ndarray,
    predicted: np.ndarray,
    dates: np.ndarray,
    name: str,
    city: str,
    horizon: int,
    ax: plt.Axes,
) -> plt.Axes:
    markers = {
        "baseline": "o",
        1: "<",
        2: "v",
        3: "s",
        4: "P",
        5: "D",
        6: "X",
        7: "<",
        8: "1",
        9: "2",
        10: "3",
        11: "4",
        12: "8",
        13: "h",
        14: "H",
    }

    colors = {
        "baseline": "black",
        1: "red",
        2: "orange",
        3: "green",
        4: "blue",
        5: "purple",
        6: "brown",
        7: "pink",
        8: "gray",
        9: "olive",
        10: "cyan",
        11: "magenta",
        12: "yellow",
        13: "darkgreen",
        14: "darkblue",
    }

    if horizon == 1:
        sns.lineplot(
            x=dates,
            y=true,
            ax=ax,
            label="True",
            marker=markers["baseline"],
            color=colors["baseline"],
        )
    sns.lineplot(
        x=dates,
        y=predicted,
        ax=ax,
        label=f"Predicted t={horizon}",
        marker=markers[horizon],
        color=colors[horizon],
        linestyle="solid",
    )

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_title(f"{city} - {name}", fontsize=24, pad=20, loc="center")
    ax.set_xlabel("Date", labelpad=10)
    ax.set_ylabel(name.split(",")[0])
    ax.legend(fontsize=16, loc="upper left")
    # adjust x and y axis label sizes
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Adjust x-axis tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        date_fmt = DateFormatter("%y/%m/%d")
        # Set date format for x-axis ticks
        ax.xaxis.set_major_formatter(date_fmt)

    # make the y axis log scale
    ax.set_yscale("log")
    # make the grid make sense for log scale
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.5)
    # add light major and minor grid lines
    ax.grid(which="major", color="#666666", linestyle="-", alpha=0.2)

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    return ax


def plot_prediction_runs(
    step_results: list[StepResult],
    city: str,
    name: str,
    n_days_back: int,
    n_days_forward: int,
    colors,
    only_delta: bool = True,
) -> figure.Figure:
    city_results = [x for x in step_results if x.city == city]
    city_results = sorted(city_results, key=lambda x: x.date_t)

    t_dates = [step_result.date_t for step_result in city_results]
    # true begin at t0-13 and end at tf
    min_date_true = t_dates[0] - pd.Timedelta(days=n_days_back - 1)
    max_date_true = t_dates[-1] + pd.Timedelta(days=n_days_forward)

    true_dates_str = pd.date_range(
        start=min_date_true, end=max_date_true, freq="D"
    ).strftime("%Y-%m-%d")

    train_date_colors = {date: next(colors["train"]) for date in true_dates_str}
    validation_date_colors = {
        date: next(colors["validation"]) for date in true_dates_str
    }
    test_date_colors = {date: next(colors["test"]) for date in true_dates_str}

    fig, ax = plt.subplots()
    ax.set_title(f"{city} {name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True)
    if len(city_results) > 1000 * 7:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    # ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.tick_params(axis="x", rotation=45)
    # plot true values

    true_values = pd.Series(dtype=float)
    true_stage = pd.Series(dtype=str)
    for city_result in city_results:
        date_t = city_result.date_t
        stage = city_result.stage.value
        true_values[date_t] = city_result.head_results[name].input[-1].item()
        true_stage[date_t] = stage

    true_values = true_values.sort_index()
    true_stage = true_stage.sort_index()

    for i, (date, value) in enumerate(true_values.items()):
        stage = true_stage[date]  # type: ignore
        if stage == "train":
            date_colors = train_date_colors
        elif stage == "validation":
            date_colors = validation_date_colors
        else:
            date_colors = test_date_colors
        ax.plot(
            date,
            value - value,
            color=date_colors[date.strftime("%Y-%m-%d")],  # type: ignore
            marker="o",
            linestyle="solid",
            label="True values" if i == 0 else "_nolegend_",
        )
    show_label = True
    for city_result in city_results:
        date_t = city_result.date_t
        predictions = city_result.head_results[name].prediction  # - true_values[date_t]
        current_value = city_result.head_results[name].input[-1].item()
        date_range = pd.date_range(
            start=date_t + pd.Timedelta(days=1),
            end=date_t + pd.Timedelta(days=n_days_forward),
            freq="D",
        )
        prediction_series = pd.Series(predictions, index=date_range).sort_index()  # type: ignore
        if city_result.stage.value == "train":
            date_colors = train_date_colors
        elif city_result.stage.value == "validation":
            date_colors = validation_date_colors
        else:
            date_colors = test_date_colors
        ax.plot(
            prediction_series.index,
            (
                prediction_series.values - current_value
                if only_delta
                else prediction_series.values
            ),
            color=date_colors[date_t.strftime("%Y-%m-%d")],
            linestyle="solid",
            label="Predictions" if show_label else "_nolegend_",
        )
        targets = city_result.head_results[name].target  # - true_values[date_t]
        target_series = pd.Series(targets, index=date_range)  # type: ignore
        target_series = target_series.sort_index()
        ax.plot(
            target_series.index,
            (
                target_series.values - current_value
                if only_delta
                else target_series.values
            ),
            color=date_colors[date_t.strftime("%Y-%m-%d")],
            linestyle="dashed",
            label="Targets" if show_label else "_nolegend_",
        )
        show_label = False
    # make the y axis log scale
    # ax.set_yscale("log")
    # make the grid make sense for log scale
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_prediction_runs_plotly(
    step_results: list[StepResult],
    city: str,
    name: str,
    n_days_back: int,
    n_days_forward: int,
    colors,
    use_log: bool = False,
    only_delta: bool = True,
):
    city_results = [x for x in step_results if x.city == city]
    city_results = sorted(city_results, key=lambda x: x.date_t)

    t_dates = [step_result.date_t for step_result in city_results]
    min_date_true = t_dates[0] - pd.Timedelta(days=n_days_back - 1)
    max_date_true = t_dates[-1] + pd.Timedelta(days=n_days_forward)
    true_dates_str = pd.date_range(
        start=min_date_true, end=max_date_true, freq="D"
    ).strftime("%Y-%m-%d")

    # Define colors for each date based on the stage
    train_date_colors = {date: next(colors["train"]) for date in true_dates_str}
    validation_date_colors = {
        date: next(colors["validation"]) for date in true_dates_str
    }
    test_date_colors = {date: next(colors["test"]) for date in true_dates_str}

    # Initialize Plotly figure
    fig = go.Figure()

    # Set figure layout (title, labels, and grid)
    fig.update_layout(
        title=f"{city} {name}".split(",")[0].replace("_", " ").title(),
        xaxis_title="Date",
        yaxis_title="Change from last known value" if only_delta else "Value",
        xaxis=dict(
            gridcolor="lightgrey",
            showgrid=True,
            # tickmode="auto",  # Adjust based on your data density
            nticks=18,
            tickformat="%Y-%m-%d" if len(city_results) <= 1000 * 7 else "%Y-%m",
            tickangle=-55,
        ),
        yaxis=dict(
            gridcolor="lightgrey",
            showgrid=True,
        ),
    )

    seen_legend_names = set()
    # Plot true values, predictions, and targets for each city_result

    for i, city_result in enumerate(city_results):
        if i % 6 != 0:
            continue
        date_t = city_result.date_t.strftime("%Y-%m-%d")
        stage = city_result.stage.value
        date_colors = (
            train_date_colors
            if stage == "train"
            else validation_date_colors
            if stage == "validation"
            else test_date_colors
        )
        # transform values of colors from 4-way tuple tu rgba string.
        # must multiply by 255 to convert from 0-1 to 0-255
        date_colors = {
            k: f"rgba({v[0] * 255},{v[1] * 255},{v[2] * 255},{v[3]})"
            for k, v in date_colors.items()
        }

        # True values
        true_value = city_result.head_results[name].input[-1].item()
        past_values = city_result.head_results[name].input.flatten()
        regression_model = LinearRegression()
        past_dates = pd.date_range(end=date_t, periods=len(past_values), freq="D")
        regression_model.fit(
            past_dates.to_julian_date().values.reshape(-1, 1), past_values
        )
        future_dates = pd.date_range(start=date_t, periods=n_days_forward, freq="D")
        predictions_regression = regression_model.predict(
            future_dates.to_julian_date().values.reshape(-1, 1)
        )

        # Plot linear regression predictions as a dotted line
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=(
                    predictions_regression - true_value
                    if only_delta
                    else predictions_regression
                ),
                mode="lines",
                name="LR Predictions",
                line=dict(color=date_colors[date_t], dash="dot"),
                showlegend=(
                    True if "LR Predictions" not in seen_legend_names else False
                ),
            )
        )
        seen_legend_names.add("LR Predictions")
        # Predictions
        predictions = city_result.head_results[name].prediction
        prediction_dates = pd.date_range(start=date_t, periods=n_days_forward, freq="D")
        trace_name = "CNN Predictions"
        show_legend = trace_name not in seen_legend_names
        if show_legend:
            seen_legend_names.add(trace_name)
        fig.add_trace(
            go.Scatter(
                x=prediction_dates,
                y=predictions - true_value if only_delta else predictions,
                mode="lines",
                name=trace_name,
                line=dict(color=date_colors[date_t]),
                showlegend=show_legend,
            )
        )

        # Targets
        targets = city_result.head_results[name].target
        if use_log:
            targets += 1
        trace_name = "Targets"
        show_legend = trace_name not in seen_legend_names
        if show_legend:
            seen_legend_names.add(trace_name)
        fig.add_trace(
            go.Scatter(
                x=prediction_dates,
                y=targets - true_value if only_delta else targets,
                mode="markers",
                name=trace_name,
                marker=dict(
                    size=4,
                    symbol="circle-open",
                    color=date_colors[date_t],
                    line=dict(width=1, color=date_colors[date_t]),
                ),
                showlegend=show_legend,
            )
        )

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
            font=dict(size=11),
        )
    )

    # make the y axis log scale
    fig.update_yaxes(type="log" if use_log else "linear")
    fig.write_image(
        "/Users/jeandavidt/Developer/jeandavidt/cov19ml/plotly_mc_figs/delta_"
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%s")
        + ".png",
        scale=2,
    )
    return fig


def plot_all_prediction_time_series(
    step_results: list[StepResult],
    n_days_back: int,
    n_days_forward: int,
    stage: str,
    only_delta=True,
) -> dict[str, figure.Figure]:
    train_colors = get_color_cycle(num_colors=n_days_forward, palette="viridis")
    validation_colors = get_color_cycle(num_colors=n_days_forward, palette="magma")
    test_colors = get_color_cycle(num_colors=n_days_forward, palette="twilight_shifted")
    colors = {
        "train": train_colors,
        "validation": validation_colors,
        "test": test_colors,
    }
    figures = {}
    cities = list(set({x.city for x in step_results}))
    head_names = list(step_results[0].head_results.keys())
    # plot predictions
    for city in cities:
        for name in head_names:
            if stage == "test":
                # figures[f"{city}_{name}_log"] = plot_prediction_runs_plotly(
                #     step_results,
                #     city,
                #     name,
                #     n_days_back,
                #     n_days_forward,
                #     task,
                #     colors,
                #     True,
                # )
                figures[f"{city}_{name}"] = plot_prediction_runs_plotly(
                    step_results,
                    city,
                    name,
                    n_days_back,
                    n_days_forward,
                    colors,
                    False,
                    only_delta=only_delta,
                )
            else:
                figures[f"{city}_{name}"] = plot_prediction_runs(
                    step_results,
                    city,
                    name,
                    n_days_back,
                    n_days_forward,
                    colors,
                    only_delta=only_delta,
                )
    return figures


def plot_all_prediction_45_degrees(
    step_results: list[StepResult],
    n_days_forward: int,
    stage: str,
    use_delta: bool,
) -> dict[str, figure.Figure]:
    train_colors = get_color_cycle(num_colors=n_days_forward, palette="viridis")
    validation_colors = get_color_cycle(num_colors=n_days_forward, palette="magma")
    test_colors = get_color_cycle(num_colors=n_days_forward, palette="twilight_shifted")
    colors = {
        "train": train_colors,
        "validation": validation_colors,
        "test": test_colors,
    }
    figures = {}
    cities = list(set({x.city for x in step_results}))
    head_names = list(step_results[0].head_results.keys())
    # plot predictions
    for city in cities:
        for name in head_names:
            results = [x for x in step_results if x.city == city]
            plot_name = f"45 degree plot - {city}_{name}"
            targets = extract_target_values_df_from_results_list(
                results, n_days_forward, name, use_delta
            )
            predictions = extract_predictions_values_df_from_results_list(
                results, n_days_forward, name, use_delta
            )
            figures[plot_name] = plot_45deg_per_day(
                targets.to_numpy(), predictions.to_numpy(), name, colors[stage]
            )
    return figures


def convert_color_to_rgb(color: str) -> str:
    string = "rgb("
    for i in range(3):
        string += str(int(color[i] * 255))
        if i != 2:
            string += ","
    string += ")"
    return string


def get_color_cycle(num_colors: int, palette: str = "viridis") -> itertools.cycle:
    cmap = cm.get_cmap(palette)
    colors_list = [cmap(value) for value in np.linspace(0, 0.8, num_colors)]
    colors = itertools.cycle(colors_list)
    return colors


def plot_head_results_by_horizon(
    comparison_df: pd.DataFrame, name: str, city: str, n_days_ahead: int
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for horizon in range(1, n_days_ahead + 1):
        ax = plot_time_series(
            true=comparison_df.true.to_numpy(),
            predicted=comparison_df[f"t+{horizon}"].to_numpy(),
            dates=comparison_df.index.to_numpy(),
            name=name,
            city=city,
            horizon=horizon,
            ax=ax,
        )
    return fig


TRIANGULAR_COLORS = {
    "D": "red",
    "E": "red",
    "A": "red",
    "C": "green",
    "F": "green",
    "B": "green",
    "G": "blue",
}
TRIANGULAR_ALPHAS = {
    "D": 0.8,
    "E": 0.5,
    "A": 0.2,
    "C": 0.2,
    "F": 0.5,
    "B": 0.8,
    "G": 0.5,
}
MONOTONIC_COLORS = {
    "U": "red",
    "D": "green",
    "G": "blue",
}
MONOTONIC_ALPHAS = {
    "U": 0.8,
    "D": 0.8,
    "G": 0.8,
}


def plot_classes(axes, ax_index, series, colors, alphas, height, y0):
    series = series.dropna()
    current_letter = series.iloc[0]
    start_index = series.index[0]
    for i, letter in series.items():
        if letter != current_letter:
            width = i - start_index
            rect = patches.Rectangle(
                (start_index, y0),
                width,
                height,
                color=colors[current_letter],
                alpha=alphas[current_letter],
            )
            axes[ax_index].add_patch(rect)
            text_x = start_index + width / 2
            text_y = y0 + height / 2
            axes[ax_index].text(
                text_x, text_y, current_letter, ha="center", va="center"
            )

            start_index = i
            current_letter = letter

    width = series.index[-1] - series.index[-2]
    rect = patches.Rectangle(
        (start_index, y0),
        width,
        height,
        color=colors[current_letter],
        alpha=alphas[current_letter],
    )
    axes[ax_index].add_patch(rect)
    text_x = start_index + width / 2
    text_y = y0 + height / 2
    axes[ax_index].text(text_x, text_y, current_letter, ha="center", va="center")
    return axes


def plot_trend_classes(slope, curvature, monotonic_classes, triangular_classes):
    # make a figure with 3 subplots on top of each other
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    sns.lineplot(data=slope, ax=ax, color="black", linestyle="solid", label="slope")
    ax.set_ylabel("slope")

    axb = ax.twinx()
    sns.lineplot(
        data=curvature, ax=axb, color="grey", linestyle="solid", label="curvature"
    )

    # set y limits for ax3 from 0 to 1

    ax2.set_ylim(0, 1)
    ax2 = plot_classes(
        [ax2],
        0,
        triangular_classes,
        TRIANGULAR_COLORS,
        TRIANGULAR_ALPHAS,
        height=1,
        y0=0,
    )

    ax3.set_ylim(0, 1)
    ax3 = plot_classes(
        [ax3], 0, monotonic_classes, MONOTONIC_COLORS, MONOTONIC_ALPHAS, height=1, y0=0
    )

    ax.legend([], frameon=False)
    ax2.legend([], frameon=False)
    ax3.legend([], frameon=False)
    fig.legend(loc="upper left")
    return fig


def plot_classification(results: list[StepResult], n_days) -> dict[str, figure.Figure]:
    if not results:
        raise ValueError("The list of results is empty.")
    first_test_data = results[0]
    head_names = list(first_test_data.head_results.keys())
    n_days_back = first_test_data.head_results[head_names[0]].input.shape[0]
    # initialize a figure with a row for each head, a column for each day in the prediction window
    plt.tight_layout()
    figures = {}
    cities = list({result.city for result in results})
    for city in cities:
        city_results = [result for result in results if result.city == city]
        for i, head_name in enumerate(head_names):
            # subplots in a grid of 1 col and  n_days + 1 rows
            fig, axes = plt.subplots(n_days, 1, figsize=(24, 20))

            true_values = extract_true_values_series_from_results_list(
                city_results, n_days_back=n_days_back, name=head_name
            )

            for j in range(1, n_days):
                # get the true and predicted values for the day j
                predictions = extract_prediction_from_results_list_by_horizon(
                    city_results, horizon=j, name=head_name, task="CLASSIFICATION"
                )

                # plot the true values as rectangles with the color corresponding to the class
                sns.lineplot(
                    data=predictions,
                    ax=axes[j],
                    color="blue",
                    linestyle="solid",
                    label=f"prediction t+{j+1}",
                )
                sns.lineplot(
                    data=true_values,
                    ax=axes[j],
                    color="red",
                    linestyle="solid",
                    label="target",
                )
                axes[j].set_ylabel("class")
                # set the title of the subplot to the name of the head
                axes[j].set_title(f"t+{j+1}")
            fig_name = f"{city}_{head_name}"
            figures[fig_name] = fig
    return figures


def plot_distance_kde_between_cities(distances_dico):
    fig, _ = plt.subplots(figsize=(10, 8))
    max_distance = max(
        [distances.max() for distances in distances_dico.values()]
    ).item()

    for i, ((city_a, city_b), distances) in enumerate(distances_dico.items()):
        distances = distances.flatten()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sns.kdeplot(
                distances,
                label=f"{city_a}-{city_b}",
                color=pl_colors[i % len(pl_colors)],
            )
    # set x axis limits
    plt.xlim(-max_distance, max_distance)
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title("Distance between cities")
    return fig


def plot_closest_examples_between_cities(examples_by_city, closest_pairs):
    # Ok! So, for each pait in closest pairs, I want a subplot.
    # For each subplot, I plot all the examples for the two cities
    # That means extracting the inputs and the targets of each example and align them with the date of the example
    # Then, I plot the two time series on the same plot
    # Then, I take the examples in closest_pairs and plot them on the same plot
    # how many heads?

    n_cols = len(examples_by_city[list(examples_by_city.keys())[0]][0].head_results)
    n_rows = len(closest_pairs)
    colors = ["blue", "red"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 12, n_rows * 10))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    current_row = 0
    for city_a, city_b in closest_pairs.keys():
        for i, city in enumerate([city_a, city_b]):
            inputs = {}
            examples = examples_by_city[city]
            closest_items = closest_pairs[(city_a, city_b)]
            for col_no, head_name in enumerate(examples[0].head_results.keys()):
                for example in examples:
                    date = example.date_t
                    inputs[date] = (
                        example.head_results[head_name].input[-1].numpy().item()
                    )
                inputs = pd.Series(inputs).sort_index()
                sns.lineplot(
                    data=inputs,
                    ax=axes[current_row, col_no],
                    color=colors[i],
                    linestyle="solid",
                    label=f"{city}",
                )
                axes[current_row, col_no].set_ylabel(head_name)
                axes[current_row, col_no].set_title(f"{city_a}, {city_b} {head_name}")

                city_a_closest = closest_items[city_a]
                city_b_closest = closest_items[city_b]
                date_closest_a = city_a_closest.date_t
                date_closest_b = city_b_closest.date_t
                value_closest_a = (
                    city_a_closest.head_results[head_name].input[-1].numpy().item()
                )
                value_closest_b = city_b_closest.head_results[head_name].input[-1]
                sns.scatterplot(
                    x=[date_closest_a, date_closest_b],
                    y=[value_closest_a, value_closest_b],
                    ax=axes[current_row, col_no],
                    color="black",
                    marker="x",
                    s=200,
                    label="closest",
                )
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())

        current_row += 1
    return fig

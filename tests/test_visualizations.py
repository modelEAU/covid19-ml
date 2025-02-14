import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append("src")
from types_ml import HeadTensors, Stage, StepResult
from visualizations import (
    get_color_cycle,
    plot_45deg_per_day,  # noqa: E402
    plot_all_prediction_45_degrees,
    plot_city_classifier_distribution_tsne,
)


def test_plot_city_classifier_distribution_tsne():
    # Create some sample data
    n_samples = 100
    n_features = 10
    prediction_tensors = np.random.rand(n_samples, n_features)
    true_city_labels = np.random.randint(0, 5, n_samples)
    city_lookup = {
        "New York": 0,
        "Los Angeles": 1,
        "Chicago": 2,
        "Houston": 3,
        "Phoenix": 4,
    }

    # Call the function with the sample data
    fig = plot_city_classifier_distribution_tsne(
        prediction_tensors, true_city_labels, city_lookup
    )

    # Check that the figure has the expected type
    assert isinstance(fig, plt.Figure)

    # Check that the figure contains a scatter plot
    assert len(fig.axes) == 1
    assert isinstance(fig.axes[0], plt.Axes)

    # Check that the scatter plot has the expected data
    scatter_data = fig.axes[0].collections[0].get_offsets()
    assert scatter_data.shape == (n_samples, 2)


def test_plit_45deg_per_day():
    test_true = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
        ]
    )
    test_pred = np.array(
        [
            [0, 1, 2, 2, 2, 2, 2],
            [0, 1, 2, 2, 2, 2, 2],
            [0, 1, 3, 2, 2, 2, 2],
            [0, 1, 2, 4, 2, 2, 2],
            [0, 1, 6, 2, 2, 2, 2],
            [0, 1, 2, 2, 7, 2, 2],
            [0, 1, 2, 2, 5, 2, 2],
            [0, 1, 2, 2, 2, 2, 2],
            [0, 1, 2, 1, 8, 2, 2],
            [0, 1, 2, 2, 0, 2, 2],
        ]
    )
    test_colors = get_color_cycle(num_colors=7, palette="twilight_shifted")
    fig = plot_45deg_per_day(test_true, test_pred, "test var", test_colors)
    fig.savefig("/tmp/test.png")
    assert isinstance(fig, plt.Figure)  # type: ignore

    assert len(fig.axes) == 7
    assert isinstance(fig.axes[0], plt.Axes)


def test_plot_all_prediction_45_degrees():
    step_result = StepResult(
        city="New York",
        task="REGRESSION",
        date_t=pd.Timestamp("2020-01-01"),
        x_item=torch.tensor([1, 2, 3]),
        y_item=torch.tensor([1, 2, 3]),
        city_predicted_label=torch.tensor([1]),
        city_output=torch.tensor([1, 2, 3]),
        info_item={},
        head_results={
            "example_var": HeadTensors(
                input=torch.tensor([1, 2, 3]),
                target=torch.tensor([4, 5, 6]),
                prediction=torch.tensor([3, 5, 9]),
            ),
        },
        stage=Stage.TEST,
        mc_results=None,
    )
    results = [step_result]
    figures = plot_all_prediction_45_degrees(results, 3, "test", use_delta=True)
    for name, fig in figures.items():
        assert isinstance(fig, plt.Figure)
        fig.savefig(f"/tmp/test_{name}.png")

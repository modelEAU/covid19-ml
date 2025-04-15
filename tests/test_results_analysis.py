import sys

import torch

sys.path.append("src")
from covid19_ml.results_analysis import (
    closest_pair,
    min_distance_between_cities,
)


def test_closest_pair():
    city_a = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [7.0, 5.0, 6.0],
        ]
    )
    city_b = torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0], [1.0, 1.0, 3.0]])

    expected_closest_pair = (0, 1, 0.0)
    assert closest_pair(city_a, city_b) == expected_closest_pair


def test_min_distances_between_cities():
    city_a = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [7.0, 5.0, 6.0],
        ]
    )
    city_b = torch.tensor(
        [[4.0, 5.0, 6.0], [1.0, 1.0, 3.0], [1.0, 6.0, 3.0]]  # this guy
    )
    city_c = torch.tensor([[5.0, 6.0, 2.0], [1.0, 1.0, 3.0]])  # this guy
    dico = {"city_a": city_a, "city_b": city_b, "city_c": city_c}

    expected = {
        ("city_a", "city_b"): (0, 1, 1.0),
        ("city_a", "city_c"): (0, 1, 1.0),
        ("city_b", "city_c"): (1, 1, 0.0),
    }

    min_distances = min_distance_between_cities(dico)
    assert min_distances == expected

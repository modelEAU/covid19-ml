import collections
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE, RandomOverSampler

from covid19_ml.datasets import CityDataSet


def stack_all_datsets(
    ds_list: list[CityDataSet],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    # stack dataset X, y and infos
    X = torch.cat([ds.X for ds in ds_list])
    y = torch.cat([ds.y for ds in ds_list])
    # get convert the list of info dicts of each dataset to a single list of dicts
    infos = []
    for ds in ds_list:
        infos.extend(ds.info)
    return X, y, infos


def resample_with_sampler(
    X: torch.Tensor, y: torch.Tensor, labels: list[str], sampler
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    # Flatten the data and convert to numpy
    X_flat = X.reshape(X.shape[0], -1).numpy()
    y_flat = y.reshape(y.shape[0], -1).numpy()

    # Combine X and y for SMOTE
    Xy_combined = np.hstack((X_flat, y_flat))

    Xy_resampled, labels_resampled = sampler.fit_resample(Xy_combined, labels)
    # Split and reshape the resampled data back to original format
    X_resampled_flat = Xy_resampled[:, : X_flat.shape[1]]
    y_resampled_flat = Xy_resampled[:, X_flat.shape[1] :]

    X_resampled = X_resampled_flat.reshape(-1, X.shape[1], X.shape[2])
    y_resampled = y_resampled_flat.reshape(-1, y.shape[1], y.shape[2])

    # back to torch tensors
    X_resampled = torch.from_numpy(X_resampled)
    y_resampled = torch.from_numpy(y_resampled)

    return X_resampled, y_resampled, labels_resampled


def get_indices_of_cities_from_infos(city_labels, infos):
    cities = list(set([info["city"] for info in infos]))
    city_indices = {}
    for city in cities:
        indices = [i for i, city_label in enumerate(city_labels) if city_label == city]
        city_indices[city] = indices
    return city_indices


def generate_new_infos(infos, new_city_labels, original_city_indices, first_info):
    new_infos = []
    original_length = len(infos)
    first_first_input_date = pd.to_datetime(first_info["first_input_date"])
    first_last_input_date = pd.to_datetime(first_info["last_input_date"])
    first_first_target_date = pd.to_datetime(first_info["first_target_date"])
    first_last_target_date = pd.to_datetime(first_info["last_target_date"])
    for i, new_label_city in enumerate(new_city_labels[original_length:]):
        sample_index = np.random.choice(original_city_indices[new_label_city], 1)[0]
        info = deepcopy(infos[sample_index])
        info["augmented"] = True
        first_input_date = (
            (first_first_input_date - pd.DateOffset(days=i)).date().strftime("%Y-%m-%d")
        )
        last_input_date = (
            (first_last_input_date - pd.DateOffset(days=i)).date().strftime("%Y-%m-%d")
        )

        first_target_date = (
            (first_first_target_date - pd.DateOffset(days=i))
            .date()
            .strftime("%Y-%m-%d")
        )
        last_target_date = (
            (first_last_target_date - pd.DateOffset(days=i)).date().strftime("%Y-%m-%d")
        )
        info["first_input_date"] = first_input_date
        info["last_input_date"] = last_input_date
        info["first_target_date"] = first_target_date
        info["last_target_date"] = last_target_date
        new_infos.append(info)
    return new_infos


def augment_datasets(dataset_list, task):
    if task == "CLASSIFICATION":
        new_datasets = []
        for ds in dataset_list:
            X = ds.X
            y = ds.y
            infos = ds.info
            class_counter = collections.Counter()
            class_labels = []
            for i in range(y.shape[0]):
                unique_values, counts = y[i].unique(return_counts=True)
                # Get the index of the maximum count
                idx = counts.argmax()
                # Fetch the most common integer
                most_common = unique_values[idx].item()
                class_counter[most_common] += 1
                class_labels.append(most_common)
            # print("Class counts before augmentation: ", class_counter)
            sampler = RandomOverSampler(random_state=42, sampling_strategy="auto")
            X_resampled, y_resampled, new_city_labels = resample_with_sampler(
                X, y, class_labels, sampler
            )
            # convert to float32 or long, depending on feature
            y_resampled = y_resampled.long()
            X_resampled = X_resampled.to(torch.float32)

            original_class_indices = {
                0: [i for i, label in enumerate(class_labels) if label == 0],
                1: [i for i, label in enumerate(class_labels) if label == 1],
                2: [i for i, label in enumerate(class_labels) if label == 2],
            }
            first_info = infos[0]
            new_infos = generate_new_infos(
                infos, new_city_labels, original_class_indices, first_info
            )

            total_infos = infos + new_infos
            new_dataset = deepcopy(ds)
            new_dataset.X = X_resampled
            new_dataset.y = y_resampled
            new_dataset.info = total_infos
            new_dataset.sort_by_date()

            new_datasets.append(new_dataset)
            new_class_counter = collections.Counter()
            for i in range(y_resampled.shape[0]):
                unique_values, counts = y_resampled[i].unique(return_counts=True)
                # Get the index of the maximum count
                idx = counts.argmax()
                # Fetch the most common integer
                most_common = unique_values[idx].item()
                new_class_counter[most_common] += 1
            # print("Class counts after augmentation: ", new_class_counter)
        class_balanced_datasets = new_datasets
    else:
        class_balanced_datasets = dataset_list

    X, y, infos = stack_all_datsets(class_balanced_datasets)
    # get the city labels
    city_labels = [info["city"] for info in infos]
    # count number of samples per city

    # print("Count before augmentation: ", counter)

    # print(task)
    if task == "CLASSIFICATION":
        sampler = RandomOverSampler(random_state=42, sampling_strategy="auto")
    else:
        sampler = SMOTE(random_state=42, sampling_strategy="auto")

    X_resampled, y_resampled, new_city_labels = resample_with_sampler(
        X, y, city_labels, sampler
    )
    # convert to float32 or long, depending on feature

    if task == "CLASSIFICATION":
        y_resampled = y_resampled.long()
    else:
        y_resampled = y_resampled.to(torch.float32)
    X_resampled = X_resampled.to(torch.float32)

    original_city_indices = get_indices_of_cities_from_infos(city_labels, infos)

    first_info = infos[0]
    new_infos = generate_new_infos(
        infos, new_city_labels, original_city_indices, first_info
    )

    total_infos = infos + new_infos

    resampled_city_indices = get_indices_of_cities_from_infos(
        new_city_labels, total_infos
    )

    new_ds_list = [deepcopy(ds) for ds in dataset_list]
    for ds in new_ds_list:
        ds.X = X_resampled[resampled_city_indices[ds.city.value]]
        ds.y = y_resampled[resampled_city_indices[ds.city.value]]
        ds.info = [total_infos[i] for i in resampled_city_indices[ds.city.value]]
        ds.sort_by_date()
    return new_ds_list

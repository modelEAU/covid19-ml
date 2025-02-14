import os
from typing import Callable, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import CityDataSet
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Subset, default_collate
from types_ml import Task


def get_batch_item_info(info, index):
    item_info = {}
    for key, value in info.items():
        if isinstance(value, dict):
            item_info[key] = get_batch_item_info(value, index)
        else:
            item_info[key] = value[index]
    return item_info


def recombine_batch_item_info(info_list):
    combined_info = {}
    for i, item_info in enumerate(info_list):
        if i == 0:
            for key, value in item_info.items():
                if isinstance(value, dict):
                    combined_info[key] = value
                elif isinstance(value, str):
                    combined_info[key] = [value]
                elif isinstance(value, torch.Tensor):
                    combined_info[key] = value
            continue
        for key, value in item_info.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        if len(combined_info[key][k].shape) == 0:
                            combined_info[key][k] = combined_info[key][k].unsqueeze(0)
                        combined_info[key][k] = torch.cat(
                            [combined_info[key][k], v.unsqueeze(0)], dim=0
                        )
                    else:
                        combined_info[key][k].append(v)
            elif isinstance(value, str):
                combined_info[key].append(value)
            elif isinstance(value, torch.Tensor):
                if len(combined_info[key].shape) == 0:
                    combined_info[key] = combined_info[key].unsqueeze(0)
                combined_info[key] = torch.cat(
                    [combined_info[key], value.unsqueeze(0)], dim=0
                )
            else:
                raise ValueError(f"Unexpected value type {type(value)}")
    return combined_info


def get_n_workers() -> int:
    if slurm_nodes := os.environ.get("SLURM_JOB_NUM_NODES"):
        return int(slurm_nodes) * 4
    return 0


class CityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        city_dataset: CityDataSet,
        test_split_date: str,
        batch_size: int,
        n_splits: int,
        validation_split: int,
        task: Task,
        city_balancer: Optional[Callable] = None,
        log_transform: bool = False,
    ):
        super().__init__()
        self.dataset = city_dataset
        self.city = city_dataset.city.value
        self.tensor_config = city_dataset.tensor_config
        self.batch_size = batch_size
        self.test_split_date = test_split_date
        self.n_splits = n_splits
        self.task = task
        self.num_workers = get_n_workers()
        self.city_balancer = city_balancer
        self.x_norm: Optional[dict[str, tuple[float, float]]] = {}
        self.y_norm: Optional[dict[str, tuple[float, float]]] = {}
        self.kf = KFold(n_splits=n_splits)
        self.current_split = -1
        self.validation_split = validation_split
        self.log_transform = log_transform

    def set_normalization_stats(self, training_subset, task: Task) -> None:
        inputs_lookup = training_subset[0][2]["x_column_lookup"]
        targets_lookup = training_subset[0][2]["y_column_lookup"]

        inputs = torch.stack([item[0] for item in iter(training_subset)])
        targets = torch.stack([item[1] for item in iter(training_subset)])
        x_norm = {}
        y_norm: Optional[dict] = {}
        for var_name, index in inputs_lookup.items():
            if self.log_transform and ("RAW" in var_name or "SMOOTH" in var_name):
                inputs[:, index, :] = torch.log1p(inputs[:, index, :])
            x_norm[var_name] = (
                inputs[:, index, :].mean().item(),
                inputs[:, index, :].std().item(),
            )
        if task == Task.REGRESSION:
            for var_name, index in targets_lookup.items():
                if self.log_transform and ("RAW" in var_name or "SMOOTH" in var_name):
                    targets[:, index, :] = torch.log1p(targets[:, index, :])
                y_norm[var_name] = (  # type: ignore
                    targets[:, index, :].mean().item(),
                    targets[:, index, :].std().item(),
                )
        else:
            y_norm = None
        self.x_norm = x_norm
        self.y_norm = y_norm
        return

    def split_data(self, split: int):
        # 1. Splitting into train/val and test
        test_start_idx = self.dataset.date_index(self.test_split_date)
        if test_start_idx + self.tensor_config.n_timesteps_back >= len(self.dataset):
            raise ValueError(
                f"Test split date {self.test_split_date} is too late for dataset"
            )

        test_indices = []
        for i, info in enumerate(self.dataset.info):
            if pd.to_datetime(info["first_input_date"]) >= pd.to_datetime(
                self.test_split_date
            ):
                test_indices.append(i)
        train_val_indices = []
        for i, info in enumerate(self.dataset.info):
            if pd.to_datetime(info["last_input_date"]) < pd.to_datetime(
                self.test_split_date
            ):
                train_val_indices.append(i)

        # 3. K-fold cross-validation
        for i, (train_idx, val_idx) in enumerate(self.kf.split(train_val_indices)):
            if i != split:
                continue
            first_val_date = pd.to_datetime(
                self.dataset[val_idx[0]][2]["first_input_date"]
            )
            last_val_date = pd.to_datetime(
                self.dataset[val_idx[-1]][2]["last_input_date"]
            )

            train_idx = [
                idx
                for idx in train_idx
                if (
                    pd.to_datetime(self.dataset[idx][2]["last_input_date"])
                    < first_val_date
                )
                or (
                    pd.to_datetime(self.dataset[idx][2]["first_input_date"])
                    > last_val_date
                )
            ]
            return train_idx, val_idx, test_indices
        raise ValueError(f"Split {split} not found")

    def setup(self, stage: str):
        if stage == "fit":
            train_idx, val_idx, test_indices = self.split_data(self.validation_split)
            self.train_dataset = Subset(
                self.dataset,
                train_idx,
            )
            self.val_dataset = Subset(self.dataset, val_idx)  # type: ignore

            self.set_normalization_stats(self.train_dataset, self.task)

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        elif stage == "test":
            train_idx, val_idx, test_indices = self.split_data(self.validation_split)
            self.test_dataset = Subset(
                self.dataset,
                test_indices,
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class CombinedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datamodules: list[CityDataModule],
        batch_size: int,
        city_balancer: Optional[Callable] = None,
    ):
        super().__init__()
        self.datamodules = datamodules
        self.batch_size = batch_size
        self.num_workers = get_n_workers()
        self.city_balancer = city_balancer

        self.x_norm: dict[str, Optional[dict[str, tuple[float, float]]]] = {}
        self.y_norm: dict[str, Optional[dict[str, tuple[float, float]]]] = {}

    def prepare_data(self):
        for datamodule in self.datamodules:
            datamodule.prepare_data()

    def setup(self, stage=None):
        for datamodule in self.datamodules:
            datamodule.setup(stage=stage)
            datamodule_city = datamodule.city
            self.x_norm[datamodule_city] = datamodule.x_norm
            self.y_norm[datamodule_city] = datamodule.y_norm

        if stage == "fit":
            # Combine all datasets
            train_datasets = [dm.train_dataset for dm in self.datamodules]
            val_datasets = [dm.val_dataset for dm in self.datamodules]
            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)
        elif stage == "test":
            # Combine all datasets
            test_datasets = [dm.test_dataset for dm in self.datamodules]
            self.test_dataset = ConcatDataset(test_datasets)

    def balance_cities(self, batch):
        if self.city_balancer is None:
            augmented_batch = batch
        else:
            augmented_batch = self.city_balancer(batch)
        return default_collate(augmented_batch)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.balance_cities,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.balance_cities,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class ClassBalancedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datamodule: CombinedDataModule,
        class_balancer: Optional[Callable] = None,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.batch_size = self.datamodule.batch_size
        self.num_workers = self.datamodule.num_workers
        self.class_balancer = class_balancer
        self.x_norm = self.datamodule.x_norm
        self.y_norm = self.datamodule.y_norm

    def prepare_data(self):
        self.datamodule.prepare_data()

    def setup(self, stage=None):
        datamodule = self.datamodule
        datamodule.setup(stage=stage)

        if stage == "fit":
            self.train_dataset = datamodule.train_dataset
            self.val_dataset = datamodule.val_dataset
        elif stage == "test":
            self.test_dataset = datamodule.test_dataset

    def balance_classes(self, batch):
        if self.class_balancer is None:
            augmented_batch = batch
        else:
            augmented_batch = self.class_balancer(batch)
        return default_collate(augmented_batch)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.balance_classes,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.balance_classes,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

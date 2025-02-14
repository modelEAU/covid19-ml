from pathlib import Path

import pandas as pd
from to_tensor import to_tensor
from torch.utils.data import Dataset
from types_ml import City, TensorConfig


class CityDataSet(Dataset):
    def __init__(
        self,
        path: Path | str,
        city: City,
        start_date: str,
        end_date: str,
        tensor_config: TensorConfig,
    ):
        self.city = city
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.X, self.y, self.info = to_tensor(
            path,
            city=city,
            start_date=start_date,
            end_date=end_date,
            tensor_config=tensor_config,
        )
        self.tensor_config = tensor_config

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        target = self.y[idx]
        info = self.info[idx]
        return sample, target, info

    def sort_by_date(self):
        """Sorts the dataset by date"""
        indices = sorted(
            range(len(self.info)),
            key=lambda k: pd.to_datetime(self.info[k]["first_input_date"]),
        )
        self.X = self.X[indices]
        self.y = self.y[indices]
        self.info = [self.info[i] for i in indices]

    def date_index(self, date: str) -> int:
        # search for index where info["first_input_date"] == date
        for i, info in enumerate(self.info):
            if pd.to_datetime(info["last_input_date"]) == pd.to_datetime(date):
                return i
        raise IndexError(f"Date {date} not found in dataset")

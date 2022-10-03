"""
os is used to read the mushroom file.
pandas handles dataframe operations.
sklearn is used to load the iris dataset.
"""
import pandas as pd
from sklearn.datasets import load_iris, load_digits


class DataLoader:
    """
    Offers the ability to change the dataset, between Iris and Mushrooms.
    """

    def __init__(self, dataset: str = "Iris"):
        self.data: pd.DataFrame
        self.features: list[str]
        self.target_column: list[str]
        self.target_names: list[str]
        self.iris = load_iris(as_frame=True)  # type: ignore
        self.digits = load_digits(as_frame=True)  # type: ignore
        self._dataset_map = {
            "Iris": {
                "data": self.iris["frame"],  # type: ignore
                "features": self.iris["feature_names"],  # type: ignore
                "target_column": "target",
                "target_names": self.iris["target_names"],  # type: ignore
            },
            "Digits": {
                "data": self.digits["frame"],  # type: ignore
                "features": self.digits["feature_names"],  # type: ignore
                "target_column": "target",
                "target_names": self.digits["target_names"],  # type: ignore
            },
        }
        self.load(dataset)

    def load(self, dataset: str):
        """
        Loads the dataset with assigned feature and target names.
        """
        self.data = self._dataset_map[dataset]["data"]
        self.features = self._dataset_map[dataset]["features"]
        self.target_column = self._dataset_map[dataset]["target_column"]
        self.target_names = self._dataset_map[dataset]["target_names"]

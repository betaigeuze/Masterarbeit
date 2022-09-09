"""
os is used to read the mushroom file.
pandas handles dataframe operations.
sklearn is used to load the iris dataset.
"""
import os
import pandas as pd
from sklearn.datasets import load_iris


class DataLoader:
    """
    Offers the ability to change the dataset, between Iris and Mushrooms.
    """

    def __init__(self, dataset: str = "Iris"):
        self.data = None
        self.features = None
        self.target = None
        self.target_names = None
        self._dirname = os.path.dirname(__file__)
        self._filename_mushrooms = os.path.join(self._dirname, "data/mushrooms.csv")
        self._dataset_map = {
            "Iris": {
                "data": load_iris(as_frame=True)["frame"],
                "features": [
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ],
                "target": "target",
                "target_names": ["setosa", "versicolor", "virginica"],
            },
            "Mushrooms": {
                "data": pd.read_csv(self._filename_mushrooms, sep=";"),
                "features": [
                    "cap-diameter",
                    "cap-shape",
                    "cap-surface",
                    "cap-color",
                    "does-bruise-or-bleed",
                    "gill-attachment",
                    "gill-spacing",
                    "gill-color",
                    "stem-height",
                    "stem-width",
                    "stem-root",
                    "stem-surface",
                    "stem-color",
                    "veil-type",
                    "veil-color",
                    "has-ring",
                    "ring-type",
                    "spore-print-color",
                    "habitat",
                    "season",
                ],
                "target": "class",
                "target_names": ["p", "e"],
            },
        }
        self.load(dataset)

    def load(self, dataset: str):
        """
        Loads the dataset with assigned feature and target names.
        """
        self.data = self._dataset_map[dataset]["data"]
        self.features = self._dataset_map[dataset]["features"]
        self.target = self._dataset_map[dataset]["target"]
        self.target_names = self._dataset_map[dataset]["target_names"]

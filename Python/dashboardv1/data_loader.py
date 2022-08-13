import pandas as pd
from sklearn.datasets import load_iris


class DataLoader:
    def __init__(self, dataset: str = "iris"):
        self.data = None
        self.features = None
        self.target = None
        self.target_names = None
        self._dataset_map = {
            "iris": {
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
        }
        self.load(dataset)

    def load(self, dataset: str):
        self.data = self._dataset_map[dataset]["data"]
        self.features = self._dataset_map[dataset]["features"]
        self.target = self._dataset_map[dataset]["target"]
        self.target_names = self._dataset_map[dataset]["target_names"]

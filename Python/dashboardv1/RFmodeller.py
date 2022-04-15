from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


class RFmodeller:
    def __init__(self, data: pd.DataFrame, feature_list: list = None):
        self.data = data
        self.features = feature_list
        (
            self.model,
            self.train_X,
            self.val_X,
            self.train_y,
            self.val_y,
        ) = self.train_model()

    def train_model(self):
        self.data = self.data.dropna(axis=0)
        y = self.data.Price
        X = self.data[self.features]
        train_X, val_X, train_y, val_y = train_test_split(
            X, y, random_state=0, test_size=0.25
        )
        forest_model = RandomForestRegressor(
            n_estimators=100, random_state=1, n_jobs=-1
        )
        forest_model.fit(train_X, train_y)
        return forest_model, train_X, val_X, train_y, val_y

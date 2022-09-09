"""
numpy is used for some calculations.
pandas handles dataframe operations.
sklearn calculates some performance metrics and the the decision tree
class is just used as a type hint, as some trees are passed as arguments.
RFmodeller is used to create the random forest model.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from random_forest_modeller import RFmodeller


class DataframeOperator:
    """Handling everything related to preparing the {tree_df} dataframe for visualization."""

    def __init__(self, rfm: RFmodeller, features: list[str]):
        self.rfm = rfm
        self.features = features
        self.tree_df = self.get_tree_df_from_model(rfm, features)
        self.tree_df = self.add_cluster_information_to_tree_df(rfm, features)
        self.tree_df = self.add_ranks_to_tree_df(self.tree_df)
        self.tree_df = self.add_grid_coordinates_to_tree_df(self.tree_df)

    # Inspect RF trees and retrieve number of leaves and depth for each tree
    # This could be altered to more interesting metrics in the future
    def get_tree_df_from_model(
        self, rfm: RFmodeller, features: list[str]
    ) -> pd.DataFrame:
        """
        Constructs the tree_df dataframe from the random forest model.
        This dataframe contains information about each tree in the random forest.
        The method iterates over each estimator to retrieve metrics about them.
        """
        tree_df = pd.DataFrame(columns=(["n_leaves", "depth"] + features))
        for est in rfm.model.estimators_:
            new_row = {"n_leaves": est.get_n_leaves(), "depth": est.get_depth()}

            # List of tuples with variable and importance
            feature_importances = [
                (feature, round(importance, 2))
                for feature, importance in zip(features, list(est.feature_importances_))
            ]
            # Add feature importance per feature to the new row
            new_row.update(dict(feature_importances))
            self.add_classification_report_metrics_to_row(rfm, est, new_row)
            tree_df = pd.concat(
                [tree_df, pd.DataFrame(new_row, index=[0])], ignore_index=True
            )

        return tree_df

    def add_classification_report_metrics_to_row(
        self, rfm: RFmodeller, est: DecisionTreeClassifier, new_row: dict
    ) -> None:
        """
        Assembles classification report metrics about the given estimator.
        """
        y_predicted = est.predict(rfm.X_train)
        labels = np.unique(rfm.y_train)
        classific_report = classification_report(
            rfm.y_train,
            y_predicted,
            output_dict=True,
            labels=labels,
            target_names=rfm.target_names,
            digits=4,
        )
        # Add each feature's classification report dictionary values to the new row
        for metric, value in classific_report.items():
            if isinstance(value, dict):
                for label, value in value.items():
                    new_row[f"{metric}_{label}"] = value
            else:
                new_row[f"{metric}"] = value

    def add_cluster_information_to_tree_df(
        self, rfm: RFmodeller, features: list[str]
    ) -> pd.DataFrame:
        """
        Adds cluster information to the tree_df dataframe.
        """
        tree_df = self.get_tree_df_from_model(rfm, features)
        tree_df = pd.concat([tree_df, rfm.cluster_df], axis=1)
        tree_df["cluster"] = tree_df["cluster"].apply(
            lambda x: "Noise" if x == -1 else x
        )
        tree_df = pd.concat([tree_df, rfm.tsne_df], axis=1)
        tree_df = pd.concat([tree_df, rfm.silhouette_scores_df], axis=1)
        return tree_df

    def add_ranks_to_tree_df(self, tree_df: pd.DataFrame) -> pd.DataFrame:
        """Arbitrary measure to rank the trees by their rank of classifying a specific class.
        EXSPERIMENTAL"""
        tree_df["weighted_avg_f1_rank"] = tree_df["weighted avg_f1-score"].rank(
            ascending=False, method="first"
        )
        tree_df["accuracy_rank"] = tree_df["accuracy"].rank(
            ascending=False, method="first"
        )
        return tree_df

    def add_grid_coordinates_to_tree_df(self, tree_df: pd.DataFrame) -> pd.DataFrame:
        """Assigns each tree a grid coordinate based on their tree id.
        The coordinate has no deeper meaning and serves only to visualize the trees in a grid."""
        tree_df["grid_x"] = tree_df["tree"].apply(lambda x: str(x)[-1:])
        # assign grid y to the 10^1 position of the tree number
        tree_df["grid_y"] = tree_df["tree"].apply(
            lambda x: int(str(x)[:1] if x > 9 else 0)
        )
        return tree_df

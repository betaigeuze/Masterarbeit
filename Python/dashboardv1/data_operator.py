import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from RFmodeller import RFmodeller


class DataOperator:
    """Handling everything related to preparing the dataframe for the visualization."""

    def __init__(self, rfm: RFmodeller, features: list[str]):
        self.rfm = rfm
        self.features = features
        self.tree_df = self.get_tree_df_from_model(rfm, features)
        self.tree_df = self.add_cluster_information_to_tree_df(rfm, features)
        self.tree_df = self.add_ranks_to_tree_df(self.tree_df)

    # Inspect RF trees and retrieve number of leaves and depth for each tree
    # This could be altered to more interesting metrics in the future
    def get_tree_df_from_model(self, rfm, features) -> pd.DataFrame:
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

    def add_classification_report_metrics_to_row(self, rfm, est, new_row):
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

    def add_cluster_information_to_tree_df(self, rfm, features) -> pd.DataFrame:
        tree_df = self.get_tree_df_from_model(rfm, features)
        tree_df = pd.concat([tree_df, rfm.cluster_df], axis=1)
        tree_df = pd.concat([tree_df, rfm.tsne_df], axis=1)
        tree_df = pd.concat([tree_df, rfm.silhouette_scores_df], axis=1)
        return tree_df

    def add_ranks_to_tree_df(self, tree_df):
        tree_df["weighted_avg_f1_rank"] = tree_df["weighted avg_f1-score"].rank(
            ascending=False, method="first"
        )
        tree_df["accuracy_rank"] = tree_df["accuracy"].rank(
            ascending=False, method="first"
        )
        return tree_df

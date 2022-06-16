from cProfile import label
from dashboard_controller import DashboardController
from RFmodeller import RFmodeller
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import multiprocessing as mp
from sklearn.metrics import classification_report


def main():
    # Define dataset
    iris = load_iris()
    data = pd.DataFrame(
        {
            "sepal length": iris.data[:, 0],
            "sepal width": iris.data[:, 1],
            "petal length": iris.data[:, 2],
            "petal width": iris.data[:, 3],
            "species": iris.target,
        }
    )
    features = ["sepal length", "sepal width", "petal length", "petal width"]
    # Create RF model
    rfm = RFmodeller(data, features, ["species"], iris.target_names)
    # Create dashboard controller
    dc = DashboardController(data, features)
    dc.create_base_dashboard()
    # Create tree dataframe
    tree_df = get_tree_df_from_model(rfm, features)
    tree_df = assemble_tree_df(rfm, features)

    # create_scatter() returns the filtered altair selection interval object
    # in addition to the chart itself
    scatter_filter_interval, scatter_chart = dc.create_scatter(tree_df)
    bar_chart = dc.create_feature_importance_barchart(tree_df, scatter_filter_interval)
    # This is just a test!
    heatmap = dc.create_heatmap(tree_df)
    dc.display_charts(scatter_chart, bar_chart, heatmap)


# Inspect RF trees and retrieve number of leaves and depth for each tree
# This could be altered to more interesting metrics in the future
def get_tree_df_from_model(rfm, features) -> pd.DataFrame:
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

        y_predicted = est.predict(rfm.X_test)

        labels = np.unique(rfm.y_test)

        classific_report = classification_report(
            rfm.y_test,
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

        tree_df = pd.concat(
            [tree_df, pd.DataFrame(new_row, index=[0])], ignore_index=True
        )

    return tree_df


# Add the cluster information to the tree dataframey
def assemble_tree_df(rfm, features) -> pd.DataFrame:
    tree_df = get_tree_df_from_model(rfm, features)
    tree_df = pd.concat([tree_df, rfm.cluster_df], axis=1)
    return tree_df


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

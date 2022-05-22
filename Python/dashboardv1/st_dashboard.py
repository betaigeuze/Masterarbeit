from dashboard_controller import DashboardController
from RFmodeller import RFmodeller
import pandas as pd
import os
import multiprocessing as mp


def main():
    # GET A RELIABLE PATH
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    reliable_path = os.path.join(__location__, "melb_data.csv")
    melbourne_data = pd.read_csv(reliable_path)
    melbourne_features = [
        "Rooms",
        "Bathroom",
        "Landsize",
        "Propertycount",
        "BuildingArea",
        "YearBuilt",
        "Lattitude",
        "Longtitude",
    ]
    rfm = RFmodeller(melbourne_data, melbourne_features)
    dc = DashboardController(melbourne_data, melbourne_features)
    dc.create_base_dashboard()
    tree_df = get_tree_df_from_model(rfm, melbourne_features)

    # create scatter returns the filtered altair selection interval object
    # in addition to the chart itself
    scatter_filter_interval, scatter_chart = dc.create_scatter(tree_df)
    bar_chart = dc.create_feature_importance_barchart(tree_df, scatter_filter_interval)
    dc.display_charts(scatter_chart, bar_chart)


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
        # Add feature importance to the new row
        new_row.update(dict(feature_importances))

        # replaced append with concat
        tree_df = pd.concat(
            [tree_df, pd.DataFrame(new_row, index=[0])], ignore_index=True
        )

    return tree_df


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

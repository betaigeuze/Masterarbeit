from dashboard_controller import DashboardController
from RFmodeller import RFmodeller
import pandas as pd
import os


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
    # TODO:
    # Fix selection interval to work on other charts
    # Find a way that is scalable for many charts on the dashboard
    # (reference this: https://altair-viz.github.io/altair-tutorial/notebooks/06-Selections.html)
    tree_df = get_tree_df_from_model(rfm)

    # create scatter returns the filtered altair selection interval object
    scatter_filter_interval = dc.create_scatter(tree_df)
    dc.feature_importance_barchart(
        rfm.model.feature_importances_, scatter_filter_interval
    )

    # deprecated:
    # dc.create_RF_overview()


# Inspect RF trees and retrieve number of leaves and depth for each tree
# This could be altered to more interesting metrics in the future
def get_tree_df_from_model(rfm):
    tree_df = pd.DataFrame()
    for est in rfm.model.estimators_:
        new_row = {"n_leaves": est.get_n_leaves(), "depth": est.get_depth()}
        # TODO:
        # regarding pandas warning => replace append with concat
        tree_df = tree_df.append(new_row, ignore_index=True)
    return tree_df


if __name__ == "__main__":
    main()

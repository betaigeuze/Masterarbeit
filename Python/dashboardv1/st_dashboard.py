from dashboard_controller import DashboardController
from RFmodeller import RFmodeller
import pandas as pd
import numpy as np


def main():
    melbourne_data = pd.read_csv("melb_data.csv")
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
    # Find a way to connect the indice values of the RF overview with the actual
    # estimators of the model.
    # Also:
    # Think of some interesting sorting criteria for the trees
    # Then:
    # Add interactive selection of the trees (this does not make sense, if neither of the
    # 2 points above are solved.)
    dc.create_RF_overview()
    dc.feature_importance_barchart(rfm.model.feature_importances_)


if __name__ == "__main__":
    main()

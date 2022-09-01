from dashboard_controller import DashboardController
from RFmodeller import RFmodeller
import multiprocessing as mp
from dataframe_operator import DataframeOperator
from data_loader import DataLoader
import streamlit as st


def main():
    dc = base_loader()
    if dc.app_mode == "Expert":
        dc.create_expert_page()
    elif dc.app_mode == "Tutorial":
        dc.create_tutorial_page()
    elif dc.app_mode == "Standard":
        dc.create_standard_page(show_df=False)


def base_loader():
    # Load dataset
    if "dataset" in st.session_state:
        dl = DataLoader(st.session_state["dataset"])
    else:
        dl = DataLoader()
    # Create RF model
    rfm = RFmodeller(dl.data, dl.features, dl.target, dl.target_names, n_estimators=100)
    # Create tree dataframe
    df_operator = DataframeOperator(rfm, dl.features)
    # Create dashboard controller
    dc = DashboardController(dl.data, dl.features, df_operator.tree_df)
    return dc


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

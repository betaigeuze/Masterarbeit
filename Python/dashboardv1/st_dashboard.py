from email.mime import base
from dashboard_controller import DashboardController
from RFmodeller import RFmodeller
import multiprocessing as mp
from dataframe_operator import DataframeOperator
from data_loader import DataLoader
import streamlit as st


def main():
    # Handling page selection here for now
    # Might want to offload this to dashboard_controller
    # Changing the order of this, will change which page is displayed first
    app_mode = st.sidebar.radio("Select a page to display", ["Dashboard", "Tutorial"])
    if app_mode == "Dashboard":
        display_expert_dashboard()
    elif app_mode == "Tutorial":
        display_tutorial()


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


def display_tutorial():
    dc = base_loader()
    dc.create_tutorial_page()


def display_expert_dashboard():
    dc = base_loader()
    dc.create_expert_page(show_df=False)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

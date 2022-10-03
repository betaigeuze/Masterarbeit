from dashboard_controller import DashboardController
from random_forest_modeller import RFmodeller
import multiprocessing as mp
from dataframe_operator import DataframeOperator
from data_loader import DataLoader
from dashboard_page_creator import DashboardPageCreator
import streamlit as st


def main():
    # Pylance pull request regarding altair change
    # https://github.com/microsoft/pylance-release/issues/3210
    dc = base_loader()
    dpc = DashboardPageCreator(dc)
    if dc.app_mode == "Tutorial":
        dpc.create_tutorial_page()
    elif dc.app_mode == "Dashboard":
        dpc.create_dashboard_page(show_df=False)


def base_loader() -> DashboardController:
    st.set_page_config(layout="wide")
    # Load dataset
    if "data_choice" in st.session_state:
        dl = DataLoader(st.session_state["data_choice"])
    else:
        dl = DataLoader()

    # Create RF model
    rfm = RFmodeller(dl.data, dl.features, dl.target_column, dl.target_names)

    # Create tree dataframe
    df_operator = DataframeOperator(rfm, dl.features)
    # Create dashboard controller
    dc = DashboardController(dl.data, dl.features, df_operator)

    return dc


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

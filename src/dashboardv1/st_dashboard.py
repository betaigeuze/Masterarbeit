from dashboard_controller import DashboardController
from random_forest_modeller import RFmodeller
import multiprocessing as mp
from dataframe_operator import DataframeOperator
from data_loader import DataLoader
from dashboard_page_creator import DashboardPageCreator
from os.path import exists
from pathlib import Path
import streamlit as st
import pickle


def main():
    # Pylance pull request regarding altair change
    # https://github.com/microsoft/pylance-release/issues/3210
    dc = base_loader()
    dpc = DashboardPageCreator(dc)
    if dc.app_mode == "Expert":
        dpc.create_expert_page()
    elif dc.app_mode == "Tutorial":
        dpc.create_tutorial_page()
    elif dc.app_mode == "Standard":
        dpc.create_standard_page(show_df=False)


def base_loader():
    # Load dataset
    if "dataset" in st.session_state:
        dl = DataLoader(st.session_state["dataset"])
    else:
        dl = DataLoader()

    pickle_path = Path.cwd().joinpath("src", "dashboardv1", "pickle", "rfm.pickle")

    # Check for existing pickle
    if not exists(pickle_path):
        # Create RF model
        rfm = RFmodeller(
            dl.data, dl.features, dl.target, dl.target_names, n_estimators=100
        )
        # Serialize
        with open(pickle_path, "wb") as outfile:
            pickle.dump(rfm, outfile)

    # Deserialization
    with open(pickle_path, "rb") as infile:
        rfm_unpickled = pickle.load(infile)

    # Create tree dataframe
    df_operator = DataframeOperator(rfm_unpickled, dl.features)
    # Create dashboard controller
    dc = DashboardController(dl.data, dl.features, df_operator.tree_df)
    return dc


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

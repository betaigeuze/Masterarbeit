from dashboard_controller import DashboardController
from RFmodeller import RFmodeller
import multiprocessing as mp
from dataframe_operator import DataframeOperator
from data_loader import DataLoader
import streamlit as st


def main():
    # Load dataset
    if "dataset" in st.session_state:
        dl = DataLoader(st.session_state["dataset"])
    else:
        dl = DataLoader()
    # Create RF model
    rfm = RFmodeller(dl.data, dl.features, dl.target, dl.target_names, n_estimators=100)
    # Create dashboard controller
    dc = DashboardController(dl.data, dl.features)
    # Create tree dataframe
    df_operator = DataframeOperator(rfm, dl.features)
    tree_df = df_operator.tree_df
    scatter_chart = dc.basic_scatter(tree_df)
    tsne_chart = dc.create_tsne_scatter(tree_df)
    bar_chart = dc.create_feature_importance_barchart(tree_df)
    cluster_comparison_chart = dc.create_cluster_comparison_bar_plt(tree_df)
    # cluster_comparison_chart2 = dc.create_cluster_comparison_bar_plt_dropdown(tree_df)
    # rank_scatter = dc.create_basic_rank_scatter(tree_df)

    dc.create_base_dashboard(tree_df, show_df=False)
    dc.display_charts(scatter_chart, cluster_comparison_chart, tsne_chart, bar_chart)
    # dc.test_modal()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

from pyexpat import features
import streamlit as st
import pandas as pd
import altair as alt
import itertools


class DashboardController:
    def __init__(self, dataset: pd.DataFrame, features: list):
        self.dashboard_sidebar = st.sidebar.empty()
        self.dashboard_sidebar.title("Sidebar")
        self.dashboard_sidebar.markdown("# Sidebar")
        self.dashboard = st.container()
        st.header("RaFoView")
        self.dataset = dataset
        self.features = features

    def create_base_dashboard(self):
        st.subheader("Input Data")

    def create_feature_importance_barchart(
        self, tree_df: pd.DataFrame, filter_interval: alt.selection_interval
    ):
        st.write(tree_df)
        chart = (
            alt.Chart(tree_df)
            .transform_fold(self.features, as_=["feature", "importance"])
            .mark_bar(opacity=0.3)
            .encode(
                x="mean(importance):Q",
                y=alt.Y("feature:N", stack=None, sort="-x"),
            )
            .transform_filter(filter_interval)
        )
        return chart

    """Scatterplot displaying all estimators of the RF model
    x-Axis: number of leaves
    y-Axis: depth of the tree"""

    def create_scatter(self, tree_df: pd.DataFrame):
        # (reference this: https://altair-viz.github.io/altair-tutorial/notebooks/06-Selections.html)
        # (reference this: https://altair-viz.github.io/user_guide/selection_intervals.html)
        # to understand what the interval variable is doing
        interval = alt.selection_interval()
        chart = (
            alt.Chart(tree_df)
            .mark_point()
            .encode(
                x=alt.X("n_leaves", scale=alt.Scale(zero=False)),
                y=alt.Y("depth", scale=alt.Scale(zero=False)),
            )
            .add_selection(interval)
        )

        return interval, chart

    """Pass any number of altair charts to this function and they will be displayed.
    The order in the provided list is the order in which the charts will be displayed"""

    def display_charts(self, *charts):
        # This logic is necessary in order to make the selection interval work
        # Meaning: "concatenating" the two charts with "&" is the only way this will work
        if len(charts) == 1:
            st.altair_chart(charts[0], use_container_width=True)
        else:
            st.altair_chart(alt.vconcat(charts), use_container_width=True)

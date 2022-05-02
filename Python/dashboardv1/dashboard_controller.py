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
        # st.write(self.dataset.head())

    def feature_importance_barchart(
        self,
        feature_importances_: pd.DataFrame,
        scatter_interval: alt.selection_interval,
    ):
        importances = list(feature_importances_)
        # List of tuples with variable and importance
        feature_importance = [
            (feature, round(importance, 2))
            for feature, importance in zip(self.features, importances)
        ]
        # Sort the feature importances by most important first
        # feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        feature_importance_df = pd.DataFrame(
            sorted(feature_importance, key=lambda x: x[1], reverse=True)
        )
        chart = (
            alt.Chart(feature_importance_df)
            .mark_bar(opacity=0.3)
            .encode(x="1:Q", y=alt.Y("0:N", stack=None, sort="-x"))
        ).transform_filter(scatter_interval)
        st.altair_chart(chart, use_container_width=True)

    """Scatterplot displaying all estimators of the RF model
    x-Axis: number of leaves
    y-Axis: depth of the tree"""

    def create_scatter(self, tree_df: pd.DataFrame):
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
        st.altair_chart(chart, use_container_width=True)
        return interval

    # Deprecated and replaced by create_scatter
    def create_RF_overview(self):
        a = range(10)
        b = range(10)
        product = itertools.product(a, b)
        product_df = pd.DataFrame(
            [x for x in product], columns=["latitude", "longitude"]
        )
        chart = (
            alt.Chart(product_df)
            .mark_circle(size=60)
            .encode(
                x="latitude:N",
                y="longitude:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)

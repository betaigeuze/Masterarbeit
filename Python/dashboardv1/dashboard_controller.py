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

        self.dataset = dataset
        self.features = features

    def create_base_dashboard(self):
        st.write(self.dataset.head())

    def feature_importance_barchart(self, feature_importances_: pd.DataFrame):
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
        )
        st.altair_chart(chart, use_container_width=True)

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
            .encode(x="latitude:N", y="longitude:N",)
        )
        st.altair_chart(chart, use_container_width=True)

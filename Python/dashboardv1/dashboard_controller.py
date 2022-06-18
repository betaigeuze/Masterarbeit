import streamlit as st
import pandas as pd
import altair as alt


class DashboardController:
    def __init__(self, dataset: pd.DataFrame, features: list[str]):
        self.dashboard_sidebar = st.sidebar.empty()
        self.dashboard_sidebar.title("Sidebar")
        self.dashboard_sidebar.markdown("# Sidebar")
        self.dashboard = st.container()
        self.dashboard.header("RaFoView")
        self.dataset = dataset
        self.features = features
        self.filter_interval = alt.selection_interval()
        self.scale_color = alt.Scale(scheme="redblue")

    def create_base_dashboard(self, tree_df: pd.DataFrame):
        self.dashboard.subheader("Tree Data with Feature Importances")
        self.dashboard.write(tree_df)

    def create_feature_importance_barchart(self, tree_df: pd.DataFrame) -> alt.Chart:
        chart = (
            alt.Chart(tree_df)
            .transform_fold(self.features, as_=["feature", "importance"])
            .mark_bar(opacity=0.3)
            .encode(
                x=alt.X("mean(importance):Q"),
                y=alt.Y("feature:N", stack=None, sort="-x"),
            )
            .transform_filter(self.filter_interval)
        )
        return chart

    """Scatterplot displaying all estimators of the RF model
    x-Axis: number of leaves
    y-Axis: depth of the tree"""

    def basic_scatter(self, tree_df: pd.DataFrame) -> alt.Chart:
        # (reference this: https://altair-viz.github.io/altair-tutorial/notebooks/06-Selections.html)
        # (reference this: https://altair-viz.github.io/user_guide/selection_intervals.html)
        # to understand what the interval variable is doing
        chart = (
            alt.Chart(tree_df)
            .mark_point()
            .encode(
                x=alt.X("virginica_f1-score:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("versicolor_f1-score:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("cluster:N"),
            )
            .add_selection(self.filter_interval)
        )

        return chart

    def create_tsne_scatter(self, tree_df: pd.DataFrame) -> alt.Chart:
        chart = (
            alt.Chart(tree_df)
            .mark_point()
            .encode(
                x=alt.X("Component 1:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("Component 2:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("cluster:N", scale=self.scale_color),
            )
            .add_selection(self.filter_interval)
        )
        return chart

    def create_silhouette_plot(self, tree_df: pd.DataFrame) -> alt.Chart:
        chart = (
            alt.Chart(tree_df)
            .mark_area()
            .encode(
                x=alt.X("tree:N", sort="-y", axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y("Silhouette Score:Q"),
                color=alt.Color("cluster:N", scale=self.scale_color),
            )
            .properties(width=600, height=300)
        )
        return chart

    """Pass any number of altair charts to this function and they will be displayed.
    The order in the provided list is the order in which the charts will be displayed
    on the page. Concatenated charts will be able to be affected by the selection in the
    scatterplot."""

    def display_charts(self, *charts: list[alt.Chart]):
        if len(charts) == 1:
            self.dashboard.altair_chart(charts[0], use_container_width=True)
        else:
            self.dashboard.altair_chart(alt.vconcat(*charts), use_container_width=True)

from cProfile import label
from mimetypes import init
from numpy import spacing
import streamlit as st
import pandas as pd
import altair as alt

# TODO:
# color scheme
# more in depth analysis of clusters
# more in depth analysis of trees?


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

    def basic_scatter(self, tree_df: pd.DataFrame) -> alt.Chart:
        """
        Scatterplot displaying all estimators of the RF model
        x-Axis: number of leaves
        y-Axis: depth of the tree
        """
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
        tsne_chart = (
            alt.Chart(tree_df)
            .mark_point()
            .encode(
                x=alt.X("Component 1:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("Component 2:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("cluster:N", scale=self.scale_color),
                tooltip="cluster:N",
            )
            .add_selection(self.filter_interval)
        )
        silhoutte_chart = self.create_silhouette_plot(tree_df)
        return alt.hconcat(tsne_chart, silhoutte_chart)

    def create_silhouette_plot(self, tree_df: pd.DataFrame) -> alt.Chart:
        # This is not optimal, but apparently there is no way in altair (and not even in)
        # Vega-Lite to sort by 2 attributes at the same time...
        # Let's just hope, we dont need any sorting after this point
        # Note the sort=None, because altair would otherwise overwrite the pandas sort
        tree_df.sort_values(
            by=["cluster", "Silhouette Score"], ascending=False, inplace=True
        )
        chart = (
            alt.Chart(tree_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "tree:N",
                    sort=None,
                    axis=alt.Axis(labels=False, ticks=False),
                ),
                y=alt.Y("Silhouette Score:Q"),
                color=alt.Color("cluster:N", scale=self.scale_color),
                tooltip="Silhouette Score:Q",
            )
            .facet(
                column=alt.Row("cluster:N", sort="descending", title="Cluster"),
                spacing=0.4,
            )
            .resolve_scale(x="independent")
        )
        # Not pretty at all, but autosizing does not work with faceted charts (yet)
        # see: https://github.com/vega/vega-lite/pull/6672
        chart.spec.width = 20

        return chart

    def create_cluster_comparison_bar_plt(self, tree_df: pd.DataFrame) -> alt.Chart:
        # Likely the prefered way to do this
        # However, combining it with the dropdown from below would be really cool
        # It seems I can only do one of each in one chart:
        # Either dropdown OR aggragation and repeat
        chart = (
            alt.Chart(tree_df)
            .mark_bar()
            .encode(
                x=alt.X("cluster:N", sort="-y"),
                y=alt.Y(alt.repeat("column"), type="quantitative"),
                color=alt.Color("cluster:N", scale=self.scale_color),
            )
            .transform_aggregate(
                average_virginica_f1_score="average(virginica_f1-score)",
                average_versicolor_f1_score="average(versicolor_f1-score)",
                average_setosa_f1_score="average(setosa_f1-score)",
                groupby=["cluster"],
            )
            .repeat(
                column=[
                    "average_virginica_f1_score",
                    "average_versicolor_f1_score",
                    "average_setosa_f1_score",
                ]
            )
        )
        return chart

    def create_cluster_comparison_bar_plt_2(self, tree_df: pd.DataFrame) -> alt.Chart:
        # Example for selection based encodings:
        # https://github.com/altair-viz/altair/issues/1617
        # Problem here:
        # Can't get sorting to work
        # Also, not really sure what kind of aggregation is used for calculating the cluster scores
        # Trying to specify the method made me reach the library limits

        columns = [
            "virginica_f1-score",
            "versicolor_f1-score",
            "setosa_f1-score",
        ]
        select_box = alt.binding_select(options=columns, name="column")
        sel = alt.selection_single(
            fields=["column"],
            bind=select_box,
            init={"column": "virginica_f1-score"},
        )
        chart = (
            alt.Chart(tree_df)
            .transform_fold(columns, as_=["column", "value"])
            .transform_filter(sel)
            .mark_bar()
            .encode(
                x=alt.X("cluster:N", sort="-y"),
                y=alt.Y("value:Q"),
                color=alt.Color("cluster:N", scale=self.scale_color),
            )
            .add_selection(sel)
        )
        return chart

    def display_charts(self, *charts: list[alt.Chart]):
        """
        Pass any number of altair charts to this function and they will be displayed.
        The order in the provided list is the order in which the charts will be displayed
        on the page. Concatenated charts will be able to be affected by the selection in the
        scatterplot.
        """
        if len(charts) == 1:
            self.dashboard.altair_chart(charts[0], use_container_width=True)
        else:
            self.dashboard.altair_chart(alt.vconcat(*charts), use_container_width=True)

        # Problems might arise from this:
        # Selection over multiple charts requires me to concatenate them - or so I think
        # However, if I concatenate the charts, interactive selections like the dropdown
        # will appear at the bottom of the page instead of next to the relevant chart

"""Streamlit is used to display the dashboard in the browser.
Pandas handles all of the dataframes in the background.
Altair is responsible for the charts."""
import streamlit as st
import pandas as pd
import altair as alt
from typing import Union
from dataframe_operator import DataframeOperator


class DashboardController:
    """Creates all of the visualizations"""

    def __init__(
        self, dataset: pd.DataFrame, features: list[str], dfo: DataframeOperator
    ):
        self.app_mode = None  # Defined in create_sidebar()
        self.show_explanations = None  # Defined in create_sidebar()
        self.dashboard_sidebar = self.create_sidebar()
        self.dashboard = st.container()
        self.dashboard.header("RaFoView")
        self.dataset = dataset
        self.features = features
        self.tree_df = dfo.tree_df
        self.rfm = dfo.rfm
        self.dfo = dfo
        self.brush = alt.selection_interval()
        self.range_ = [
            "#8e0152",
            "#c51b7d",
            "#de77ae",
            "#f1b6da",
            "#fde0ef",
            "#f7f7f7",
            "#e6f5d0",
            "#b8e186",
            "#7fbc41",
            "#4d9221",
            "#276419",
        ]
        self.scale_color = alt.Scale(range=self.range_)
        self.color = alt.condition(
            self.brush,
            alt.Color(
                "Silhouette Score:Q",
                scale=self.scale_color,
                legend=alt.Legend(
                    orient="right",
                    # legendX=210,
                    # legendY=-40,
                    direction="vertical",
                    titleAnchor="start",
                    title="Silhouette Score",
                ),
            ),
            alt.value("lightblue"),
        )

    def create_sidebar(self) -> st.sidebar:  # type: ignore
        """
        Creates the sidebar of the dashboard.
        """

        def _change_data_key(key: str):
            st.session_state["dataset"] = key

        sidebar = st.sidebar
        sidebar.title("Sidebar")
        # Page selection
        self.app_mode = sidebar.radio(
            "Select a page to display", ["Expert", "Standard", "Tutorial"]
        )
        # Example selection
        self.data_form = sidebar.form("Data Selection", clear_on_submit=True)
        self.data_form.markdown("## Example Use Cases")
        data_choice = self.data_form.selectbox(
            "Choose a dataset:", ["Iris", "Mushrooms"]
        )
        self.data_form.form_submit_button(
            "Run",
            help="On 'run' the selected dataset will be loaded into the dashboard",
            on_click=_change_data_key(data_choice),
        )
        # Explanation toggle
        sidebar.markdown("## Toggle Explanations")
        self.show_explanations = sidebar.checkbox(
            label="Show explanations", value=True, key="show_explanations"
        )
        sidebar.markdown("## Algorithm Parameters")
        sidebar.slider(
            label="Select a value for the DBSCAN parameter 'min samples':",
            min_value=2,
            max_value=5,
            step=1,
            value=3,
            key="min_samples",
        )
        sidebar.slider(
            label="Select a value for the DBSCAN parameter 'epsilon':",
            min_value=0.1,
            max_value=0.99,
            step=0.01,
            value=0.3,
            key="eps",
        )
        sidebar.slider(
            label="Select a value for the t-SNE parameter 'learning rate':",
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            value=100.0,
            key="learning_rate",
        )
        sidebar.slider(
            label="Select a value for the t-SNE parameter 'perplexity':",
            min_value=2,
            max_value=100,
            step=1,
            value=5,
            key="perplexity",
        )
        sidebar.slider(
            label="Select a value for the t-SNE parameter 'early exaggeration':",
            min_value=2.0,
            max_value=50.0,
            step=1.0,
            value=4.0,
            key="early_exaggeration",
        )

        return sidebar

    def create_base_dashboard(self, show_df: bool = False):
        """
        Creates the base dashboard object.
        """
        self.dashboard.subheader("Investigating the Random Forest")
        if show_df:
            self.dashboard.write(self.tree_df)

    def create_feature_importance_barchart(
        self, selection: bool = True, flip: bool = False
    ) -> alt.Chart:
        """
        Create a barchart of the feature importances of the random forest.
        selection allows to concatenate the chart with others and interact with
        their selections.
        flip, if True, swaps the x and y axis.
        """
        chart = (
            alt.Chart(self.tree_df)
            .transform_fold(self.features, as_=["feature", "importance"])
            .mark_bar(fill="#4E1E1E")
            .encode(
                x=alt.X("mean(importance):Q"),
                y=alt.Y("feature:N", stack=None, sort="-x"),
            )
        )
        if selection:
            chart = chart.transform_filter(self.brush)
        if flip:
            chart.encoding.x = alt.X("feature:N", stack=None, sort="-x")
            chart.encoding.y = alt.Y("mean(importance):Q")
        return chart

    def basic_scatter(
        self, color: Union[dict, alt.SchemaBase, list], selection: bool = True
    ) -> alt.Chart:
        """
        Scatterplot displaying all estimators of the RF model
        selection allows to concatenate the chart with others and interact with
        their selections.
        """
        chart = (
            alt.Chart(self.tree_df)
            .mark_circle(stroke="#4E1E1E", strokeWidth=1)
            .encode(
                x=alt.X(
                    "grid_x:N",
                    scale=alt.Scale(zero=False),
                    title=None,
                    axis=alt.Axis(labels=False, ticks=False),
                ),
                y=alt.Y(
                    "grid_y:N",
                    scale=alt.Scale(zero=False),
                    title=None,
                    axis=alt.Axis(labels=False, ticks=False),
                ),
                color=color,
                tooltip=[
                    alt.Tooltip("tree:N", title="Tree"),
                ],
            )
        )
        if selection:
            chart = chart.add_selection(self.brush)
        return chart

    def create_tsne_scatter(self, importance: bool = False) -> alt.Chart:
        """
        Scatterplot displaying the t-SNE embedding of the RF model
        importance, if True, displays the feature importance bar chart instead
        of the silhouette score plot.
        The returned plot is a horizontal concatenation of the two plots.
        """
        tsne_chart = (
            alt.Chart(self.tree_df)
            .mark_circle(stroke="#4E1E1E", strokeWidth=1)
            .encode(
                x=alt.X("Component 1:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("Component 2:Q", scale=alt.Scale(zero=False)),
                color=self.color,
                tooltip=[
                    alt.Tooltip("cluster:N", title="Cluster"),
                ],
            )
            .add_selection(self.brush)
        )
        if importance:
            return alt.hconcat(
                tsne_chart,
                self.create_feature_importance_barchart(selection=True, flip=True),
            )  # type: ignore
        else:
            return alt.hconcat(tsne_chart, self.create_silhouette_plot())  # type: ignore

    def create_silhouette_plot(self) -> alt.Chart:
        """
        Silhouette plot displaying the silhouette score of the RF model
        The plot is sorted by cluster and the silhouette score.
        """
        # This is not optimal, but apparently there is no way in altair (and not even in)
        # Vega-Lite to sort by 2 attributes at the same time...
        # Let's just hope, we dont need any sorting after this point
        # Note the sort=None, because altair would otherwise overwrite the pandas sort
        self.tree_df.sort_values(
            by=["cluster", "Silhouette Score"], ascending=False, inplace=True
        )
        chart = (
            alt.Chart(self.tree_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "tree:N",
                    sort=None,
                    axis=alt.Axis(labels=False, ticks=False, title=None),
                ),
                y=alt.Y("Silhouette Score:Q"),
                color=self.color,
                tooltip=[
                    alt.Tooltip("Silhouette Score:Q", title="Silhouette Score"),
                ],
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

    def create_cluster_comparison_bar_repeat(self) -> alt.Chart:
        """
        Bar plot displaying the cluster comparison of the RF model
        This is one of the 2 ways of comparing the clusters of the RF model.
        The other method is create_cluster_comparison_bar_dropdown()
        """
        chart = (
            alt.Chart(self.tree_df)
            .mark_bar()
            .encode(
                x=alt.X("cluster:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(alt.repeat("column"), type="quantitative"),
                color=alt.Color("mean_silhouette_score:Q", scale=self.scale_color),
                tooltip=[alt.Tooltip("count_tree:Q", title="Number of Trees")],
            )
            .transform_aggregate(
                mean_virginica_f1_score="mean(virginica_f1-score)",
                mean_versicolor_f1_score="mean(versicolor_f1-score)",
                mean_setosa_f1_score="mean(setosa_f1-score)",
                mean_silhouette_score="mean(Silhouette Score)",
                count_tree="count(tree)",
                groupby=["cluster"],
            )
            .repeat(
                column=[
                    "mean_virginica_f1_score",
                    "mean_versicolor_f1_score",
                    "mean_setosa_f1_score",
                ]
            )
        )
        return chart

    def create_cluster_comparison_bar_dropdown(self) -> alt.Chart:
        """
        Bar plot displaying the cluster comparison of the RF model
        This is one of the 2 ways of comparing the clusters of the RF model.
        The other method is create_cluster_comparison_bar_repeat()
        """
        # Reference:
        # https://github.com/altair-viz/altair/issues/1617

        columns = [
            "mean_virginica_f1_score",
            "mean_versicolor_f1_score",
            "mean_setosa_f1_score",
        ]
        select_box = alt.binding_select(options=columns, name="column")
        sel = alt.selection_single(
            fields=["column"],
            bind=select_box,
            init={"column": "mean_virginica_f1_score"},
        )
        chart = (
            alt.Chart(self.tree_df)
            .transform_aggregate(
                mean_virginica_f1_score="mean(virginica_f1-score)",
                mean_versicolor_f1_score="mean(versicolor_f1-score)",
                mean_setosa_f1_score="mean(setosa_f1-score)",
                mean_silhouette_score="mean(Silhouette Score)",
                groupby=["cluster"],
            )
            .transform_fold(columns, as_=["column", "value"])
            .mark_bar(fill="#4E1E1E")
            .encode(
                x=alt.X("cluster:N", sort="-y"),
                y=alt.Y("value:Q"),
                tooltip=[
                    alt.Tooltip("value:Q", title="Value"),
                ],
                # color=alt.Color("mean_silhouette_score:Q", scale=self.scale_color),
            )
            .transform_filter(sel)
            .add_selection(sel)
        )
        return chart

    def display_charts(self, charts: list[alt.Chart]):
        """
        Pass any number of altair charts to this function and they will be displayed.
        The order in the provided list is the order in which the charts will be displayed
        on the page. The passed charts will be concatenated and will
        be affected by selections in plots.
        """
        if len(charts) == 1:
            self.dashboard.altair_chart(charts[0], use_container_width=True)
        else:
            self.dashboard.altair_chart(alt.vconcat(*charts), use_container_width=True)  # type: ignore

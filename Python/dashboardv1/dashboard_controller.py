import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path


class DashboardController:
    """Creates all of the visualizations"""

    # TODO: Think of a way to add explanation/tutorial part
    # Requirements:
    # - Explanation and accompanying pictures
    # - Space in the dashboard
    # - A way to toggle the explanation on and off

    def __init__(self, dataset: pd.DataFrame, features: list[str]):
        self.dashboard_sidebar = self.create_sidebar()
        self.dashboard = st.container()
        self.dashboard.header("RaFoView")
        self.dataset = dataset
        self.features = features
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
                    orient="none",
                    legendX=210,
                    legendY=-40,
                    direction="vertical",
                    titleAnchor="middle",
                    title="Silhouette Score",
                ),
            ),
            alt.value("lightblue"),
        )

    def create_sidebar(self):
        def _change_data_key(key: str):
            st.session_state["dataset"] = key

        self.dashboard_sidebar = st.sidebar
        self.dashboard_sidebar.title("Sidebar")
        self.data_form = self.dashboard_sidebar.form(
            "Data Selection", clear_on_submit=True
        )
        self.data_form.markdown("## Dataset selection")
        # TODO: Make this dropdown dependent on the dataloader dataset map for robustness
        data_choice = self.data_form.selectbox(
            "Choose a dataset:", ["Iris", "Mushrooms", "My Dataset"]
        )
        self.data_form.form_submit_button(
            "Run",
            help="On 'run' the selected dataset will be loaded into the dashboard",
            on_click=_change_data_key(data_choice),
        )

    def create_base_dashboard(self, tree_df: pd.DataFrame, show_df: bool = False):
        self.dashboard.subheader("Tree Data with Feature Importances")
        if show_df:
            self.dashboard.write(tree_df)

    def create_feature_importance_barchart(self, tree_df: pd.DataFrame) -> alt.Chart:
        chart = (
            alt.Chart(tree_df)
            .transform_fold(self.features, as_=["feature", "importance"])
            .mark_bar(fill="#4E1E1E")
            .encode(
                x=alt.X("mean(importance):Q"),
                y=alt.Y("feature:N", stack=None, sort="-x"),
            )
            .transform_filter(self.brush)
        )
        return chart

    def basic_scatter(self, tree_df: pd.DataFrame) -> alt.Chart:
        """
        Scatterplot displaying all estimators of the RF model
        """
        chart = (
            alt.Chart(tree_df)
            .mark_circle(stroke="#4E1E1E", strokeWidth=1)
            .encode(
                x=alt.X(
                    "grid_x:N",
                    scale=alt.Scale(zero=False),
                    title=None,
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y("grid_y:N", scale=alt.Scale(zero=False), title=None),
                color=self.color,
            )
            .add_selection(self.brush)
        )
        return chart

    def create_tsne_scatter(self, tree_df: pd.DataFrame) -> alt.Chart:
        tsne_chart = (
            alt.Chart(tree_df)
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
                    axis=alt.Axis(labels=False, ticks=False, title=None),
                ),
                y=alt.Y("Silhouette Score:Q"),
                color=self.color,
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

    def create_cluster_zoom_in(self, tree_df: pd.DataFrame) -> alt.Chart:
        # Example for selection based encodings:
        # https://github.com/altair-viz/altair/issues/1617
        # Problem here:
        # Can't get sorting to work
        # Also, not really sure what kind of aggregation is used for calculating the cluster scores
        # Trying to specify the method made me reach the library limits
        # For now: Leave this here as reference

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
        on the page. The passed charts will be concatenated. These concatenated charts will
        be affected by selections in plots.
        """
        if len(charts) == 1:
            self.dashboard.altair_chart(charts[0], use_container_width=True)
        else:
            self.dashboard.altair_chart(alt.vconcat(*charts), use_container_width=True)

        # Problems might arise from this:
        # Selection over multiple charts requires me to concatenate them - or so I think
        # However, if I concatenate the charts, interactive selections like the dropdown
        # will appear at the bottom of the page instead of next to the relevant chart

    def create_tutorial_page(self):
        """
        Creates a tutorial page with some explanations of a random forest.
        """

        def read_md(file_name: str) -> str:
            return Path.cwd().joinpath("Python", "dashboardv1", file_name).read_text()

        tutorial_markdown = read_md("tutorial.md")
        self.dashboard.markdown(tutorial_markdown, unsafe_allow_html=True)

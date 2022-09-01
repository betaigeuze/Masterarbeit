from xmlrpc.client import Boolean
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

    def __init__(
        self, dataset: pd.DataFrame, features: list[str], tree_df: pd.DataFrame
    ):
        self.app_mode = None  # Defined in create_sidebar()
        self.show_explanations = None  # Defined in create_sidebar()
        self.dashboard_sidebar = self.create_sidebar()
        self.dashboard = st.container()
        self.dashboard.header("RaFoView")
        self.dataset = dataset
        self.features = features
        self.tree_df = tree_df
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
                    orient="left",
                    # legendX=210,
                    # legendY=-40,
                    direction="vertical",
                    titleAnchor="start",
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
        # Page selection
        self.app_mode = self.dashboard_sidebar.radio(
            "Select a page to display", ["Standard", "Expert", "Tutorial"]
        )
        # Example selection
        self.data_form = self.dashboard_sidebar.form(
            "Data Selection", clear_on_submit=True
        )
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
        self.dashboard_sidebar.markdown("## Toggle Explanations")
        self.show_explanations = self.dashboard_sidebar.checkbox(
            label="Show explanations", value=True, key="show_explanations"
        )

    def create_base_dashboard(self, show_df: bool = False):
        self.dashboard.subheader("Investigating the Random Forest")
        if show_df:
            self.dashboard.write(self.tree_df)

    def create_feature_importance_barchart(self, selection: bool = True) -> alt.Chart:
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
        return chart

    def basic_scatter(self, color: alt.Color, selection: bool = True) -> alt.Chart:
        """
        Scatterplot displaying all estimators of the RF model
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

    def create_tsne_scatter(self) -> alt.Chart:
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
        silhoutte_chart = self.create_silhouette_plot()
        return alt.hconcat(tsne_chart, silhoutte_chart)

    def create_silhouette_plot(self) -> alt.Chart:
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

    def create_cluster_comparison_bar_plt(self) -> alt.Chart:
        # Likely the prefered way to do this
        # However, combining it with the dropdown from below would be really cool
        # It seems I can only do one of each in one chart:
        # Either dropdown OR aggragation and repeat
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

    def create_cluster_zoom_in(self) -> alt.Chart:
        # Example for selection based encodings:
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
            .mark_bar()
            .encode(
                x=alt.X("cluster:N", sort="-y"),
                y=alt.Y("value:Q"),
                color=alt.Color("mean_silhouette_score:Q", scale=self.scale_color),
            )
            .transform_filter(sel)
            .add_selection(sel)
        )
        return chart

    def display_charts(self, charts: list[alt.Chart]):
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

    def create_tutorial_page(self):
        """
        Create a tutorial page with some examples of how to use the dashboard.
        """
        layout = [
            {"content": "markdown", "file": "tutorial1.md"},
            {"content": "image", "file": "flowers.png"},
            {"content": "image", "file": "flower_measures.png"},
            {"content": "markdown", "file": "tutorial2.md"},
            {"content": "image", "file": "splitting.png"},
            {"content": "markdown", "file": "tutorial3.md"},
            {"content": "image", "file": "decision_tree.png"},
            {"content": "markdown", "file": "tutorial4.md"},
            {"content": "image", "file": "bagging.png"},
            {"content": "markdown", "file": "tutorial5.md"},
        ]
        self.create_page(layout)

    def create_standard_page(self, show_df: bool = False):
        """
        Create the expert dashboard according to the layout dictionary.
        """
        self.create_base_dashboard(show_df=show_df)
        layout = [
            {
                "content": "chart",
                "chart_element": self.basic_scatter(
                    color=alt.value("#4E1E1E"),
                    selection=False,
                ),
            },
            {
                "content": "chart",
                "chart_element": self.create_feature_importance_barchart(
                    selection=False
                ),
            },
            {
                "content": "chart",
                "chart_element": self.create_cluster_zoom_in(),
            },
            {"content": "chart", "chart_element": self.create_tsne_scatter()},
        ]
        if self.show_explanations:
            layout.insert(1, {"content": "markdown", "file": "explanation1.md"})
            layout.insert(3, {"content": "markdown", "file": "explanation2.md"})
        self.create_page(layout)

    def create_expert_page(self, show_df: bool = False):
        """
        Create the expert dashboard according to the layout dictionary.
        """
        self.create_base_dashboard(show_df=show_df)
        layout = [
            {"content": "chart", "chart_element": self.basic_scatter(self.color)},
            {
                "content": "chart",
                "chart_element": self.create_cluster_comparison_bar_plt(),
            },
            {"content": "chart", "chart_element": self.create_tsne_scatter()},
            {
                "content": "chart",
                "chart_element": self.create_feature_importance_barchart(),
            },
        ]
        if self.show_explanations:
            layout.insert(1, {"content": "markdown", "file": "explanation1.md"})

        self.create_page(layout)

    def create_page(self, layout: list[dict]):
        """
        Creates a page with according to the passed layout.
        """

        def read_md(file_name: str) -> str:
            return (
                Path.cwd()
                .joinpath("Python", "dashboardv1", "text", file_name)
                .read_text()
            )

        def read_image(file_name: str) -> str:
            return (
                Path.cwd()
                .joinpath("Python", "dashboardv1", "images", file_name)
                .read_bytes()
            )

        # In order to have concatenated charts between markdown elements,
        # we need to check if the previous element of the loop was a chart.
        # If it was and the current element is not, we can just display it.
        # Otherwise the list of charts grows and gets displayed either with
        # the first non-chart element or at the end of the loop.
        charts = []
        for item in layout:
            if item["content"] == "markdown":
                if charts:
                    self.display_charts(charts)
                    charts = []
                self.dashboard.markdown(read_md(item["file"]))
            elif item["content"] == "image":
                if charts:
                    self.display_charts(charts)
                    charts = []
                self.dashboard.image(read_image(item["file"]))
            elif item["content"] == "chart":
                charts.append(item["chart_element"])
        if charts:
            self.display_charts(charts)

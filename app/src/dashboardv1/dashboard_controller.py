"""Streamlit is used to display the dashboard in the browser.
Pandas handles all of the dataframes in the background.
Altair is responsible for the charts."""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Union
from dataframe_operator import DataframeOperator


class DashboardController:
    """Creates all of the visualizations"""

    def __init__(
        self,
        dataset: pd.DataFrame,
        features: list[str],
        dfo: DataframeOperator,
    ):
        self.app_mode = None  # Defined in create_sidebar()
        # self.show_explanations = None  # Defined in create_sidebar()
        self.rfm = dfo.rfm
        self.tree_df = dfo.tree_df
        self.dfo = dfo
        self.dashboard_sidebar = self.create_sidebar()
        self.dashboard_container = st.container()
        self.dashboard_container.header("RaFoView")
        self.dataset = dataset
        self.features = features
        self.feature_names_plus_importance = [
            feature + "_importance" for feature in self.features
        ]
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
        self.title_format_dict = {
            "anchor": "start",
            "fontSize": 22,
            "font": "serif",
            "subtitleFontSize": 18,
            "subtitleFont": "serif",
        }

    def create_sidebar(self) -> st.sidebar:  # type: ignore
        """
        Creates the sidebar of the dashboard.
        """
        sidebar = st.sidebar
        sidebar.title("Sidebar")

        # Page selection
        self.app_mode = sidebar.radio(
            "Select a page to display", ["Tutorial", "Dashboard"]
        )

        # Explanation toggle
        # self.show_explanations = sidebar.checkbox(
        #     label="Show explanations", value=True, key="show_explanations"
        # )

        # Example selection
        self.data_form = sidebar.form("Data Selection", clear_on_submit=False)
        self.data_form.markdown("## Example Use Cases")
        self.data_form.selectbox(
            label="Choose an example use case:",
            options=["Iris", "Digits"],
            key="data_choice",
        )
        self.data_form.form_submit_button(
            "Run",
            help="On 'run' the selected dataset will be loaded into the dashboard",
        )

        if self.app_mode == "Dashboard":
            # Algorithm parameter form
            algorithm_parameters_form = sidebar.form(
                "algorithm_parameters", clear_on_submit=False
            )
            algorithm_parameters_form.markdown("## Algorithm Parameters")

            algorithm_parameters_form.markdown(
                """
                <p class="sidebar-font">
                Keep in mind that changing these values, will cause the dashboard to reload. Depending on your settings, this might take a while.
                """,
                unsafe_allow_html=True,
            )
            algorithm_parameters_form.markdown("### Random Forest:")
            algorithm_parameters_form.slider(
                label="Select a value for the number of trees in the Random Forest (Changing this parameter will take a significant amount of time to calculate!):",
                min_value=20,
                max_value=200,
                step=1,
                key="n_estimators",
                help="The number of trees in the forest can boost the overall performance of a random forest. However, it can be interesting to set this to a lower\
                value to see how this affects the clustering and the performance of the tree as a whole.\n\
                There are more parameters for a Random Forest, than the number of trees, but we will keep it at this for now.",
            )
            algorithm_parameters_form.markdown("### DBSCAN:")
            algorithm_parameters_form.metric(
                label="Silhouette Score",
                value=np.round(self.rfm.silhouette_score, decimals=2),
                help="The Silhouette Score, as computed here, is an average of all Silhouette Scores\
                    of the individual datapoints, averaged together. Note, that the points classified as noise are excluded from this calculation.",
            )

            algorithm_parameters_form.slider(
                label="Select a value for the DBSCAN parameter 'min samples':",
                min_value=2,
                max_value=10,
                step=1,
                key="min_samples",
                help="The min_samples parameter is used to determine the minimum number of trees in it's neighborhood for it to be considered as a core point.\
                    Reducing this will naturally yield more clusters, while increasing it will yield less clusters.",
            )
            algorithm_parameters_form.slider(
                label="Select a value for the DBSCAN parameter 'epsilon':",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                key="eps",
                help="The eps parameter determines the maximum distance between two samples to be considered as in the neighborhood of each other.\
                    In turn, this implies that cluster sizes are even across the dataset which is not necessarily the case.",
            )
            algorithm_parameters_form.markdown("### t-SNE:")
            algorithm_parameters_form.slider(
                label="Select a value for the t-SNE parameter 'learning rate':",
                min_value=1.0,
                max_value=500.0,
                step=1.0,
                key="learning_rate",
                help="The learning rate is used to determine the step size in the gradient descent. Increasing or decreasing this parameter can yield\
                    wildly different results, so I recommend to pick a value that looks interesting and stick with it, when tuning the other 2 t-SNE parameters.",
            )
            algorithm_parameters_form.slider(
                label="Select a value for the t-SNE parameter 'perplexity':",
                min_value=5,
                max_value=50,
                step=1,
                key="perplexity",
                help="The perplexity will also change the t-SNE results significantly. Usually, larger values are used for large datasets.",
            )
            algorithm_parameters_form.slider(
                label="Select a value for the t-SNE parameter 'early exaggeration':",
                min_value=2.0,
                max_value=50.0,
                step=1.0,
                key="early_exaggeration",
                help="The early exaggeration parameter is used to increase the distance between clusters.",
            )
            algorithm_parameters_form.form_submit_button(
                "Run",
                help="On 'run' the selected dataset will be loaded into the dashboard",
            )

        return sidebar

    def show_df(self, show_df: bool = False):
        """
        For dev purposes, the dataframe can be shown on the dashboard.
        """
        if show_df:
            self.dashboard_container.write(self.tree_df)

    def create_feature_importance_barchart(
        self,
        title: str,
        subtitle: str,
        top_k: int = None,  # type: ignore
        selection: bool = True,
        flip: bool = False,
    ) -> alt.Chart:
        """
        Create a barchart of the feature importances of the random forest.
        selection allows to concatenate the chart with others and interact with
        their selections.
        flip, if True, swaps the x and y axis.
        """
        chart = (
            alt.Chart(self.tree_df)
            .transform_fold(
                self.feature_names_plus_importance, as_=["feature", "importance"]
            )
            .mark_bar(fill="#7B3514")
            .encode(
                x=alt.X("mean_importance:Q"),
                y=alt.Y(
                    "feature:N",
                    sort=alt.EncodingSortField(
                        field="mean_importance", op="mean", order="descending"
                    ),
                ),
            )
            .transform_aggregate(
                mean_importance="mean(importance)",
                groupby=["feature"],
            )
        )
        if self.check_data_choice() == "Digits":
            text = chart.mark_text(
                align="left",
                baseline="middle",
                dx=3,  # Nudges text to right so it doesn't appear on top of the bar
            ).encode(text="mean_importance:Q")
            chart = chart + text
        if selection:
            chart = chart.transform_filter(self.brush)
        if flip:
            chart.encoding.x = alt.X(
                "feature:N",
                sort=alt.EncodingSortField(
                    field="mean_importance", op="mean", order="descending"
                ),
            )
            chart.encoding.y = alt.Y("mean_importance:Q")
        if top_k:
            chart = chart.transform_window(
                rank="rank(mean_importance)",
                sort=[alt.SortField("mean_importance", order="descending")],
                frame=[None, None],
            ).transform_filter((alt.datum.rank < top_k))
        chart = self.add_title(chart, title, subtitle)
        return chart

    def basic_scatter(
        self,
        title: str,
        subtitle: str,
        color: Union[dict, alt.SchemaBase, list],
        selection: bool = True,
    ) -> alt.Chart:
        """
        Scatterplot displaying all estimators of the RF model
        selection allows to concatenate the chart with others and interact with
        their selections.
        """
        chart = (
            alt.Chart(self.tree_df)
            .mark_circle(stroke="#7B3514", strokeWidth=1)
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
        chart = self.add_title(chart, title, subtitle)
        return chart

    def create_tsne_scatter(
        self, title: str, subtitle: str, importance: bool = False
    ) -> alt.Chart:
        """
        Scatterplot displaying the t-SNE embedding of the RF model
        importance, if True, displays the feature importance bar chart instead
        of the silhouette score plot.
        The returned plot is a horizontal concatenation of the two plots.
        """
        tsne_chart = (
            alt.Chart(self.tree_df)
            .mark_circle(stroke="#7B3514", strokeWidth=1)
            .encode(
                x=alt.X("Component 1:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("Component 2:Q", scale=alt.Scale(zero=False)),
                color=self.color,
                tooltip=[
                    "cluster",
                    "tree",
                ],
            )
            .add_selection(self.brush)
        )
        if importance:
            if self.check_data_choice() == "Iris":
                chart = alt.hconcat(
                    tsne_chart,
                    self.create_feature_importance_barchart(
                        title="", subtitle="", selection=True, flip=False  # type: ignore
                    ),
                )  # type: ignore
                return self.add_title(chart, title, subtitle)  # type: ignore
            else:
                chart = alt.hconcat(
                    tsne_chart,
                    self.create_feature_importance_barchart(
                        title="", subtitle="", selection=True, flip=False  # type: ignore
                    ),
                )  # type: ignore
                return self.add_title(chart, title, subtitle)  # type: ignore
        else:
            chart = alt.hconcat(tsne_chart, self.create_silhouette_plot())
            return self.add_title(chart, title, subtitle)  # type: ignore

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
                column=alt.Row(
                    "cluster:N",
                    sort="descending",
                    title="Cluster",
                    header=alt.Header(orient="bottom"),
                ),
                spacing=0.4,
            )
            .transform_filter(alt.datum.cluster != "Noise")
            .resolve_scale(x="independent")
        )
        # Not pretty at all, but autosizing does not work with faceted charts (yet)
        # see: https://github.com/vega/vega-lite/pull/6672
        chart.spec.width = 20

        return chart

    def create_cluster_comparison_bar_repeat(
        self, title: str, subtitle: str
    ) -> alt.Chart:
        """
        Bar plot displaying the cluster comparison of the RF model
        This is one of the 2 ways of comparing the clusters of the RF model.
        The other method is create_cluster_comparison_bar_dropdown()
        """
        if self.check_data_choice() == "Iris":
            chart = (
                alt.Chart(self.tree_df)
                .mark_bar()
                .encode(
                    x=alt.X("cluster:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(alt.repeat("column"), type="quantitative"),
                    color=alt.Color(
                        "mean_silhouette_score:Q",
                        scale=self.scale_color,
                        legend=alt.Legend(orient="left", title="Mean Silhouette Score"),
                    ),
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
            )
            average_line = (
                alt.Chart(self.tree_df)
                .transform_aggregate(
                    mean_virginica_f1_score="mean(virginica_f1-score)",
                    mean_versicolor_f1_score="mean(versicolor_f1-score)",
                    mean_setosa_f1_score="mean(setosa_f1-score)",
                    mean_silhouette_score="mean(Silhouette Score)",
                    count_tree="count(tree)",
                )
                .mark_rule(size=3, strokeDash=[6, 2])
                .encode(
                    y=alt.Y(alt.repeat("column"), type="quantitative"),
                    color=alt.value("#3CA6D0"),
                    tooltip=alt.Y(alt.repeat("column"), type="quantitative"),
                )
            )
            chart = (
                (chart + average_line)
                .repeat(
                    column=[
                        "mean_virginica_f1_score",
                        "mean_versicolor_f1_score",
                        "mean_setosa_f1_score",
                    ]
                )
                .resolve_scale(y="shared")
            )
        else:
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
                    mean_0_f1_score="mean(0_f1-score)",
                    mean_1_f1_score="mean(1_f1-score)",
                    mean_2_f1_score="mean(2_f1-score)",
                    mean_3_f1_score="mean(3_f1-score)",
                    mean_4_f1_score="mean(4_f1-score)",
                    mean_5_f1_score="mean(5_f1-score)",
                    mean_6_f1_score="mean(6_f1-score)",
                    mean_7_f1_score="mean(7_f1-score)",
                    mean_8_f1_score="mean(8_f1-score)",
                    mean_9_f1_score="mean(9_f1-score)",
                    mean_silhouette_score="mean(Silhouette Score)",
                    count_tree="count(tree)",
                    groupby=["cluster"],
                )
            )
            average_line = (
                alt.Chart(self.tree_df)
                .transform_aggregate(
                    mean_0_f1_score="mean(0_f1-score)",
                    mean_1_f1_score="mean(1_f1-score)",
                    mean_2_f1_score="mean(2_f1-score)",
                    mean_3_f1_score="mean(3_f1-score)",
                    mean_4_f1_score="mean(4_f1-score)",
                    mean_5_f1_score="mean(5_f1-score)",
                    mean_6_f1_score="mean(6_f1-score)",
                    mean_7_f1_score="mean(7_f1-score)",
                    mean_8_f1_score="mean(8_f1-score)",
                    mean_9_f1_score="mean(9_f1-score)",
                )
                .mark_rule(size=3, strokeDash=[6, 2])
                .encode(
                    y=alt.Y(alt.repeat("column"), type="quantitative"),
                    color=alt.value("#3CA6D0"),
                    tooltip=alt.Y(alt.repeat("column"), type="quantitative"),
                )
            )
            average_line.encoding.tooltip.title = "Forest Average F1-Score"
            chart = (
                (chart + average_line)
                .repeat(
                    column=[
                        "mean_0_f1_score",
                        "mean_1_f1_score",
                        "mean_2_f1_score",
                        "mean_3_f1_score",
                        "mean_4_f1_score",
                        "mean_5_f1_score",
                        "mean_6_f1_score",
                        "mean_7_f1_score",
                        "mean_8_f1_score",
                        "mean_9_f1_score",
                    ]
                )
                .resolve_scale(y="shared")
            )
        chart = self.add_title(chart, title, subtitle)
        return chart

    def create_class_comparison_bar_easy(self, title: str, subtitle: str) -> alt.Chart:
        """
        Bar plot displaying the cluster comparison of the RF model
        """
        # Reference:
        # https://github.com/altair-viz/altair/issues/1617
        if self.check_data_choice() == "Iris":
            columns = [
                "mean_virginica_f1_score",
                "mean_versicolor_f1_score",
                "mean_setosa_f1_score",
            ]
            chart = (
                alt.Chart(self.tree_df)
                .transform_aggregate(
                    mean_virginica_f1_score="mean(virginica_f1-score)",
                    mean_versicolor_f1_score="mean(versicolor_f1-score)",
                    mean_setosa_f1_score="mean(setosa_f1-score)",
                )
                .transform_fold(columns, as_=["column", "value"])
                .mark_bar(fill="#7B3514")
                .encode(
                    x=alt.Y("value:Q"),
                    y=alt.X("column:N", sort="-x"),
                    tooltip=[
                        alt.Tooltip("value:Q", title="Value"),
                    ],
                )
            )
        else:
            columns = [
                "mean_0_f1_score",
                "mean_1_f1_score",
                "mean_2_f1_score",
                "mean_3_f1_score",
                "mean_4_f1_score",
                "mean_5_f1_score",
                "mean_6_f1_score",
                "mean_7_f1_score",
                "mean_8_f1_score",
                "mean_9_f1_score",
            ]
            chart = (
                alt.Chart(self.tree_df)
                .transform_aggregate(
                    mean_0_f1_score="mean(0_f1-score)",
                    mean_1_f1_score="mean(1_f1-score)",
                    mean_2_f1_score="mean(2_f1-score)",
                    mean_3_f1_score="mean(3_f1-score)",
                    mean_4_f1_score="mean(4_f1-score)",
                    mean_5_f1_score="mean(5_f1-score)",
                    mean_6_f1_score="mean(6_f1-score)",
                    mean_7_f1_score="mean(7_f1-score)",
                    mean_8_f1_score="mean(8_f1-score)",
                    mean_9_f1_score="mean(9_f1-score)",
                )
                .transform_fold(columns, as_=["column", "value"])
                .mark_bar(fill="#7B3514")
                .encode(
                    x=alt.X("value:Q"),
                    y=alt.Y("column:N", sort="-x"),
                    tooltip=[
                        alt.Tooltip("value:Q", title="Value"),
                    ],
                )
            )
        chart = self.add_title(chart, title, subtitle)
        return chart

    def create_cluster_comparison_bar_dropdown(self) -> alt.Chart:
        """
        Bar plot displaying the cluster comparison of the RF model
        This is one of the 2 ways of comparing the clusters of the RF model.
        The other method is create_cluster_comparison_bar_repeat()
        """
        # Reference:
        # https://github.com/altair-viz/altair/issues/1617

        if self.check_data_choice() == "Iris":
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
                .mark_bar(fill="#7B3514")
                .encode(
                    x=alt.X("cluster:N", sort="-y"),
                    y=alt.Y("value:Q"),
                    tooltip=[
                        alt.Tooltip("value:Q", title="Value"),
                    ],
                )
                .transform_filter(sel)
                .add_selection(sel)
            )
            average_line = (
                alt.Chart(self.tree_df)
                .transform_aggregate(
                    mean_virginica_f1_score="mean(virginica_f1-score)",
                    mean_versicolor_f1_score="mean(versicolor_f1-score)",
                    mean_setosa_f1_score="mean(setosa_f1-score)",
                )
                .transform_fold(columns, as_=["column", "value"])
                .mark_rule(size=3, strokeDash=[6, 2])
                .encode(
                    y=alt.Y("value:Q"),
                    color=alt.value("#3CA6D0"),
                    tooltip=["value:Q"],
                )
                .transform_filter(sel)
            )
        else:
            columns = [
                "mean_0_f1_score",
                "mean_1_f1_score",
                "mean_2_f1_score",
                "mean_3_f1_score",
                "mean_4_f1_score",
                "mean_5_f1_score",
                "mean_6_f1_score",
                "mean_7_f1_score",
                "mean_8_f1_score",
                "mean_9_f1_score",
            ]
            select_box = alt.binding_select(options=columns, name="column")
            sel = alt.selection_single(
                fields=["column"],
                bind=select_box,
                init={"column": columns[0]},
            )
            chart = (
                alt.Chart(self.tree_df)
                .transform_aggregate(
                    mean_0_f1_score="mean(0_f1-score)",
                    mean_1_f1_score="mean(1_f1-score)",
                    mean_2_f1_score="mean(2_f1-score)",
                    mean_3_f1_score="mean(3_f1-score)",
                    mean_4_f1_score="mean(4_f1-score)",
                    mean_5_f1_score="mean(5_f1-score)",
                    mean_6_f1_score="mean(6_f1-score)",
                    mean_7_f1_score="mean(7_f1-score)",
                    mean_8_f1_score="mean(8_f1-score)",
                    mean_9_f1_score="mean(9_f1-score)",
                    mean_silhouette_score="mean(Silhouette Score)",
                    groupby=["cluster"],
                )
                .transform_fold(columns, as_=["column", "value"])
                .mark_bar(fill="#7B3514")
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
            average_line = (
                alt.Chart(self.tree_df)
                .transform_aggregate(
                    mean_0_f1_score="mean(0_f1-score)",
                    mean_1_f1_score="mean(1_f1-score)",
                    mean_2_f1_score="mean(2_f1-score)",
                    mean_3_f1_score="mean(3_f1-score)",
                    mean_4_f1_score="mean(4_f1-score)",
                    mean_5_f1_score="mean(5_f1-score)",
                    mean_6_f1_score="mean(6_f1-score)",
                    mean_7_f1_score="mean(7_f1-score)",
                    mean_8_f1_score="mean(8_f1-score)",
                    mean_9_f1_score="mean(9_f1-score)",
                )
                .transform_fold(columns, as_=["column", "value"])
                .mark_rule()
                .encode(
                    y=alt.Y("value:Q"),
                    color=alt.value("#22607C"),
                    tooltip=["value:Q"],
                )
                .transform_filter(sel)
            )
        return alt.layer(chart, average_line)  # type: ignore

    def create_similarity_matrix(self, title: str, subtitle: str) -> alt.Chart:
        # Turn the similarity matrix into a dataframe that we can display with altair
        distance_matrix = self.rfm.distance_matrix
        x, y = np.meshgrid(
            range(0, distance_matrix.shape[0]), range(0, distance_matrix.shape[0])
        )
        source = pd.DataFrame(
            {
                "tree_x": x.ravel(),
                "tree_y": y.ravel(),
                "distance_value": distance_matrix.ravel(),
            }
        )
        source_with_xcluster_info = source.join(
            self.rfm.cluster_df.set_index("tree"),
            on="tree_x",
        )
        source_with_cluster_info = source_with_xcluster_info.join(
            self.rfm.cluster_df.set_index("tree"), on="tree_y", rsuffix="_y"
        )
        source_with_cluster_info.rename({"cluster": "cluster_x"}, axis=1, inplace=True)
        # Build the chart
        chart = (
            alt.Chart(source_with_cluster_info)
            .mark_rect()
            .encode(
                x=alt.X(
                    "tree_x:N",
                    sort=alt.EncodingSortField(
                        field="cluster_x", op="min", order="ascending"
                    ),
                    axis=None,
                ),
                y=alt.Y(
                    "tree_y:N",
                    sort=alt.EncodingSortField(
                        field="cluster_y", op="min", order="ascending"
                    ),
                    axis=None,
                ),
                color=alt.Color(
                    "distance_value:Q",
                    scale=alt.Scale(scheme="greys", reverse=True),
                ),
                tooltip=[
                    "tree_x",
                    "tree_y",
                    "distance_value:Q",
                ],
            )
            .properties(width=800, height=800)
        )
        return self.add_title(chart, title, subtitle)

    def check_data_choice(self):
        if "data_choice" in st.session_state:
            data_choice = st.session_state.data_choice
        else:
            data_choice = "Iris"
        return data_choice

    def add_title(self, chart: alt.Chart, title: str, subtitle: str) -> alt.Chart:
        """
        Add title and subtitle to chart
        """
        title_dict = self.title_format_dict.copy()
        title_dict.update({"text": [title], "subtitle": [subtitle]})
        chart = chart.properties(title=title_dict)
        return chart

    def display_charts(self, charts: list[alt.Chart]):
        """
        Pass any number of altair charts to this function and they will be displayed.
        The order in the provided list is the order in which the charts will be displayed
        on the page. The passed charts will be concatenated and will
        be affected by selections in plots.
        """
        if len(charts) == 1:
            self.dashboard_container.altair_chart(charts[0], use_container_width=False)
        else:
            self.dashboard_container.altair_chart(alt.vconcat(*charts), use_container_width=False)  # type: ignore

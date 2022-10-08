"""Path enables the reading of files containing the markdown for the dashboard."""
from pathlib import Path
from dashboard_controller import DashboardController


class DashboardPageCreator:
    """
    Handles the creation of the different dashboard pages.
    Methods include a way to format pages by using the layout dictionary.
    The layout dictionary is built differently for each page and should allow for a more
    or less flexible way of creating pages.

    The layout dictionary is a list of dictionaries. Each dictionary contains the following
    keys:
    - content: either "markdown", "image" or "chart"
    Depending on the content, the dictionary should contain the following keys:
    - file: the name of the file containing the markdown or image
    - chart_element: the chart element to be displayed, given by a dashboard controller method

    The layout dictionary is passed to the create_page method, which will then create the page.

    The Dashboard page checks if the explanations should be shown and
    which data set is chosen. This is done by checking with the dashboard controller.
    """

    def __init__(self, dashboard_controller: DashboardController = None):  # type: ignore
        self.dashboard_controller = dashboard_controller

    def create_tutorial_page(self):
        """
        Create a tutorial page with graphics of how a random forest works.
        """
        if self.dashboard_controller.check_data_choice() == "Iris":
            layout = [
                {"content": "markdown", "file": "welcome.md"},
                {"content": "markdown", "file": "iris_tutorial1.md"},
                {"content": "image", "file": "flowers.png"},
                {"content": "image", "file": "flower_measures.png"},
                {"content": "markdown", "file": "iris_tutorial2.md"},
                {"content": "image", "file": "splitting.png"},
                {"content": "markdown", "file": "iris_tutorial3.md"},
                {"content": "image", "file": "decision_tree.png"},
                {"content": "markdown", "file": "iris_tutorial4.md"},
                {"content": "image", "file": "bagging.png"},
                {"content": "markdown", "file": "iris_tutorial5.md"},
            ]
        else:
            layout = [
                {"content": "markdown", "file": "welcome.md"},
                {"content": "markdown", "file": "digits_tutorial1.md"},
                {"content": "image", "file": "digit.png"},
                {"content": "markdown", "file": "digits_tutorial2.md"},
                {"content": "image", "file": "splitting.png"},
                {"content": "markdown", "file": "digits_tutorial3.md"},
                {"content": "image", "file": "decision_tree.png"},
                {"content": "markdown", "file": "digits_tutorial4.md"},
                {"content": "image", "file": "bagging.png"},
                {"content": "markdown", "file": "digits_tutorial5.md"},
            ]
        self.create_page(layout)

    def create_dashboard_page(self, show_df: bool = False):
        """
        Create the dashboard according to the layout dictionary.
        Explanations will be toggled on by default.
        """
        self.dashboard_controller.show_df(show_df=show_df)
        if self.dashboard_controller.check_data_choice() == "Iris":
            layout = [
                {"content": "markdown", "file": "welcome.md"},
                {"content": "markdown", "file": "iris_header.md"},
                {"content": "markdown", "file": "explanation1.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_feature_importance_barchart(
                        title="Feature importances",
                        subtitle="Comparing the feature importances of the Random Forest.",
                        selection=False,
                    ),
                },
                {"content": "markdown", "file": "iris_explanation2.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_class_comparison_bar_easy(
                        title="Class performance comparison",
                        subtitle="Figure 2: A comparison of the different F1-Scores per class.",
                    ),
                },
                {"content": "markdown", "file": "performance_explanation.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_similarity_matrix(
                        title="Pairwise Distance Matrix",
                        subtitle="Distance matrix, using the graph edit distance, as the distance metric.",
                    ),
                },
                {"content": "markdown", "file": "distance_matrix_explanation.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_cluster_comparison_bar_repeat(
                        title="Cluster Performance Comparison",
                        subtitle="Comparing the performance of clusters across the different classes. The Random Forest performance is indicated by the blue line.",
                    ),
                },
                {"content": "markdown", "file": "iris_explanation3.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_tsne_scatter(
                        title="T-SNE Scatter Plot",
                        subtitle="A T-SNE embedding of the Random Forest based on the distance matrix.",
                    ),
                },
                {"content": "markdown", "file": "explanation4.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_tsne_scatter(
                        title="T-SNE Scatter Plot with Importance Bar Chart",
                        subtitle="The same T-SNE embedding, as shown above, interacting with the feature importance bar chart, that was shown earlier.",
                        importance=True,
                    ),
                },
                {"content": "markdown", "file": "explanation5.md"},
            ]
        else:
            layout = [
                {"content": "markdown", "file": "welcome.md"},
                {"content": "markdown", "file": "digits_header.md"},
                {"content": "markdown", "file": "explanation1.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_feature_importance_barchart(
                        title="Feature importances",
                        subtitle="Comparing the feature importances of the Random Forest.",
                        top_k=10,
                        selection=False,
                        flip=True,
                    ),
                },
                {"content": "markdown", "file": "digits_explanation_topk.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_feature_importance_barchart(
                        title="Feature importances",
                        subtitle="Comparing the feature importances of the Random Forest.",
                        selection=False,
                        flip=True,
                    ),
                },
                {"content": "markdown", "file": "digits_explanation2.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_class_comparison_bar_easy(
                        title="Class performance comparison",
                        subtitle="A comparison of the different F1-Scores per class.",
                    ),
                },
                {"content": "markdown", "file": "performance_explanation.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_similarity_matrix(
                        title="Pairwise Distance Matrix",
                        subtitle="Distance matrix, using the graph edit distance, as the distance metric.",
                    ),
                },
                {"content": "markdown", "file": "distance_matrix_explanation.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_cluster_comparison_bar_repeat(
                        title="Cluster Performance Comparison",
                        subtitle="Comparing the performance of clusters across the different classes. The Random Forest performance is indicated by the blue line.",
                    ),
                },
                {"content": "markdown", "file": "digits_explanation3.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_tsne_scatter(
                        title="T-SNE Scatter Plot",
                        subtitle="A T-SNE embedding of the Random Forest based on the distance matrix.",
                    ),
                },
                {"content": "markdown", "file": "explanation4.md"},
                {
                    "content": "chart",
                    "chart_element": self.dashboard_controller.create_tsne_scatter(
                        title="T-SNE Scatter Plot with Importance Bar Chart",
                        subtitle="The same T-SNE embedding, as shown above, interacting with the feature importance bar chart, that was shown earlier.",
                        importance=True,
                    ),
                },
                {"content": "markdown", "file": "explanation5.md"},
                {"content": "markdown", "file": "explanation6.md"},
            ]
        self.create_page(layout)

    def create_page(self, layout: list[dict]):
        """
        Creates a page with according to the passed layout.
        """

        def setup_font_sizes():
            self.dashboard_controller.dashboard_container.markdown(
                """
                <style>
                .text-font {
                    font-size:20px !important;
                }
                .sidebar-font {
                    font-size:18px !important;
                }

                .stRadio label {
                font-size: 20px;
                }
                .stRadio div {
                font-size: 18px;
                }

                .stSelectbox label {
                font-size: 18px;
                }

                .stSlider label {
                    font-size: 18px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

        def read_md(file_name: str) -> str:
            return (
                Path.cwd()
                .joinpath("app", "src", "dashboardv1", "text", file_name)
                .read_text(encoding="utf-8")
            )

        def read_image(file_name: str) -> bytes:
            return (
                Path.cwd()
                .joinpath("app", "src", "dashboardv1", "images", file_name)
                .read_bytes()
            )

        # In order to have concatenated charts between markdown elements,
        # we need to check if the previous element of the loop was a chart.
        # If it was and the current element is not, we can just display it.
        # Otherwise the list of charts grows and gets displayed either with
        # the first non-chart element or at the end of the loop.
        setup_font_sizes()
        charts = []
        for item in layout:
            if (
                item["content"]
                == "markdown"
                # and self.dashboard_controller.show_explanations
            ):
                if charts:
                    self.dashboard_controller.display_charts(charts)
                    charts = []
                self.dashboard_controller.dashboard_container.markdown(
                    read_md(item["file"]), unsafe_allow_html=True
                )
            elif item["content"] == "image":
                if charts:
                    self.dashboard_controller.display_charts(charts)
                    charts = []
                self.dashboard_controller.dashboard_container.image(read_image(item["file"]))  # type: ignore
            elif item["content"] == "chart":
                charts.append(item["chart_element"])
        if charts:
            self.dashboard_controller.display_charts(charts)

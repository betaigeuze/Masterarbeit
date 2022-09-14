"""Path enables the reading of files containing the markdown for the dashboard."""
from pathlib import Path
from typing import Callable
import altair as alt
from dashboard_controller import DashboardController


class DashboardPageCreator:
    """
    Handles the creation of the different dashboard pages.
    Methods include a way to format pages by using the layout dictionary.
    """

    def __init__(self, dashboard_controller: DashboardController):
        self.dashboard_controller = dashboard_controller

    def create_tutorial_page(self):
        """
        Create a tutorial page with graphics of how a random forest works.
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
        Create the standard dashboard according to the layout dictionary.
        Explanations will be toggled on by default.
        """
        self.dashboard_controller.create_base_dashboard(show_df=show_df)
        layout = [
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.basic_scatter(
                    color=alt.value("#4E1E1E"),  # type: ignore
                    selection=False,
                ),
            },
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.create_feature_importance_barchart(
                    selection=False
                ),
            },
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.create_cluster_comparison_bar_dropdown(),
            },
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.create_tsne_scatter(),
            },
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.create_tsne_scatter(
                    importance=True
                ),
            },
        ]
        if self.dashboard_controller.show_explanations:
            layout.insert(1, {"content": "markdown", "file": "explanation1.md"})
            layout.insert(3, {"content": "markdown", "file": "explanation2.md"})
            layout.insert(5, {"content": "markdown", "file": "explanation3.md"})
            layout.insert(7, {"content": "markdown", "file": "explanation4.md"})
        self.create_page(layout)

    def create_expert_page(self, show_df: bool = False):
        """
        Create the expert dashboard according to the layout dictionary.
        """
        self.dashboard_controller.create_base_dashboard(show_df=show_df)
        layout = [
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.basic_scatter(
                    self.dashboard_controller.color
                ),
            },
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.create_cluster_comparison_bar_repeat(),
            },
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.create_tsne_scatter(),
            },
            {
                "content": "streamlit",
                "streamlit_element": self.dashboard_controller.create_eps_slider,
            },
            {
                "content": "streamlit",
                "streamlit_element": self.dashboard_controller.create_learning_rate_slider,
            },
            {
                "content": "streamlit",
                "streamlit_element": self.dashboard_controller.create_min_samples_slider,
            },
            {
                "content": "chart",
                "chart_element": self.dashboard_controller.create_feature_importance_barchart(),
            },
        ]
        if self.dashboard_controller.show_explanations:
            layout.insert(1, {"content": "markdown", "file": "explanation1.md"})

        self.create_page(layout)

    def create_page(self, layout: list[dict]):
        """
        Creates a page with according to the passed layout.
        """

        def read_md(file_name: str) -> str:
            return (
                Path.cwd()
                .joinpath("src", "dashboardv1", "text", file_name)
                .read_text(encoding="utf-8")
            )

        def read_image(file_name: str) -> bytes:
            return (
                Path.cwd()
                .joinpath("src", "dashboardv1", "images", file_name)
                .read_bytes()
            )

        def execute(func: Callable):
            func()

        # In order to have concatenated charts between markdown elements,
        # we need to check if the previous element of the loop was a chart.
        # If it was and the current element is not, we can just display it.
        # Otherwise the list of charts grows and gets displayed either with
        # the first non-chart element or at the end of the loop.
        charts = []
        for item in layout:
            if item["content"] == "markdown":
                if charts:
                    self.dashboard_controller.display_charts(charts)
                    charts = []
                self.dashboard_controller.dashboard.markdown(read_md(item["file"]))
            elif item["content"] == "image":
                if charts:
                    self.dashboard_controller.display_charts(charts)
                    charts = []
                self.dashboard_controller.dashboard.image(read_image(item["file"]))  # type: ignore
            elif item["content"] == "streamlit":
                if charts:
                    self.dashboard_controller.display_charts(charts)
                    charts = []
                execute(item["streamlit_element"])
            elif item["content"] == "chart":
                charts.append(item["chart_element"])
        if charts:
            self.dashboard_controller.display_charts(charts)

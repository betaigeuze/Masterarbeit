import altair as alt
from pathlib import Path
from dashboard_controller import DashboardController


class DashboardPageCreator:
    def __init__(self, dc: DashboardController):
        self.dc = dc

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
        self.dc.create_base_dashboard(show_df=show_df)
        layout = [
            {
                "content": "chart",
                "chart_element": self.dc.basic_scatter(
                    color=alt.value("#4E1E1E"),
                    selection=False,
                ),
            },
            {
                "content": "chart",
                "chart_element": self.dc.create_feature_importance_barchart(
                    selection=False
                ),
            },
            {
                "content": "chart",
                "chart_element": self.dc.create_cluster_comparison_bar_dropdown(),
            },
            {"content": "chart", "chart_element": self.dc.create_tsne_scatter()},
            {
                "content": "chart",
                "chart_element": self.dc.create_tsne_scatter(importance=True),
            },
        ]
        if self.dc.show_explanations:
            layout.insert(1, {"content": "markdown", "file": "explanation1.md"})
            layout.insert(3, {"content": "markdown", "file": "explanation2.md"})
            layout.insert(5, {"content": "markdown", "file": "explanation3.md"})
            layout.insert(7, {"content": "markdown", "file": "explanation4.md"})
        self.create_page(layout)

    def create_expert_page(self, show_df: bool = False):
        """
        Create the expert dashboard according to the layout dictionary.
        """
        self.dc.create_base_dashboard(show_df=show_df)
        layout = [
            {"content": "chart", "chart_element": self.dc.basic_scatter(self.color)},
            {
                "content": "chart",
                "chart_element": self.dc.create_cluster_comparison_bar_repeat(),
            },
            {"content": "chart", "chart_element": self.dc.create_tsne_scatter()},
            {
                "content": "chart",
                "chart_element": self.dc.create_feature_importance_barchart(),
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
                    self.dc.display_charts(charts)
                    charts = []
                self.dc.dashboard.markdown(read_md(item["file"]))
            elif item["content"] == "image":
                if charts:
                    self.dc.display_charts(charts)
                    charts = []
                self.dc.dashboard.image(read_image(item["file"]))
            elif item["content"] == "chart":
                charts.append(item["chart_element"])
        if charts:
            self.dc.display_charts(charts)

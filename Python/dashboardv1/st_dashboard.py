from dashboard_controller import DashboardController
from RFmodeller import RFmodeller
import pandas as pd
from sklearn.datasets import load_iris
import multiprocessing as mp
from data_operator import DataOperator


def main():
    # Define dataset
    iris = load_iris()
    data = pd.DataFrame(
        {
            "sepal length": iris.data[:, 0],
            "sepal width": iris.data[:, 1],
            "petal length": iris.data[:, 2],
            "petal width": iris.data[:, 3],
            "species": iris.target,
        }
    )
    features = ["sepal length", "sepal width", "petal length", "petal width"]
    # Create RF model
    rfm = RFmodeller(data, features, ["species"], iris.target_names)
    # Create dashboard controller
    dc = DashboardController(data, features)
    # Create tree dataframe
    data_operator = DataOperator(rfm, features)
    tree_df = data_operator.tree_df
    # scatter_chart = dc.basic_scatter(tree_df)
    tsne_chart = dc.create_tsne_scatter(tree_df)
    bar_chart = dc.create_feature_importance_barchart(tree_df)
    cluster_comparison_chart = dc.create_cluster_comparison_bar_plt(tree_df)
    cluster_comparison_chart2 = dc.create_cluster_comparison_bar_plt_2(tree_df)

    dc.create_base_dashboard(tree_df=tree_df)
    dc.display_charts(
        cluster_comparison_chart, cluster_comparison_chart2, tsne_chart, bar_chart
    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()

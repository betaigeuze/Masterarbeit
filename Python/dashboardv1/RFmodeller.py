from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cluster import AgglomerativeClustering
from timeit import default_timer as timer
from datetime import timedelta
import multiprocessing as mp
import pandas as pd
import networkx as nx
import numpy as np


class RFmodeller:
    def __init__(
        self, data: pd.DataFrame, feature_list: list[str], label_list: list[str]
    ):
        self.data = data
        self.features = feature_list
        self.labels = label_list
        (
            self.model,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self.train_model()
        self.directed_graphs = self.create_dot_trees()
        self.clustering = self.calculate_tree_clusters()

    def train_model(self):
        X = self.data[self.features]
        y = self.data[self.labels]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        forest_model = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=0, n_jobs=-1
        )
        forest_model.fit(X_train, y_train)
        return forest_model, X_train, X_test, y_train, y_test

    def create_dot_trees(self) -> list[nx.DiGraph]:
        # TODO:
        # This is also really slow still. Maybe I can find a more efficient way
        # of converting the trees from sklearn objects into dot format and then into a DG.
        directed_graphs = []
        for estimator in self.model.estimators_:
            tree.export_graphviz(estimator, out_file="tree.dot")
            DG = nx.DiGraph()
            DG = nx.nx_agraph.read_dot("tree.dot")
            directed_graphs.append(DG)
        return directed_graphs

    def calculate_tree_clusters(self):
        # TODO:
        # 1. Combine the cluster labels with the trees, so that the rfm object has
        # that information. Do the same for the created trees. Each tree should be
        # assigned linked to the respective entry in the tree_df.

        # Run the calc_dist_matrix method in parallel.
        # I used this idea:
        # https://stackoverflow.com/a/56038389/12355337
        # Every process gets a slice of the list of graphs.
        # sklearn's pdist won't work because it needs numeric value inputs.
        start = timer()
        with mp.Pool() as pool:
            distance_matrix = np.array(
                pool.map(self.calc_dist_matrix_parallel, self.directed_graphs)
            )

        # calculate clusters
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="average"
        ).fit(distance_matrix)
        stop = timer()
        print(
            f"Time spent in calculcate_tree_clusters: {timedelta(seconds=stop-start)}"
        )
        return clustering

    def calc_dist_matrix_parallel(self, directed_graph: nx.DiGraph) -> np.ndarray:
        # This is still not optimal, because every row and column value is being calculated
        # It would be possible (and smarter) to calculate half the matrix and then mirror it.
        # However, this is difficult, because of the calculations happening in parallel.
        # It would be necessary to use a shared memory array to store the results.
        row_distances = np.zeros(len(self.directed_graphs))
        for i, graph1 in enumerate(self.directed_graphs):
            if graph1 != directed_graph:
                # Get this out of the loop and create a dict maybe?
                # No need to compute this on every loop iteration.
                root1 = self.get_root(graph1)
                root2 = self.get_root(directed_graph)
                row_distances[i] = nx.graph_edit_distance(
                    graph1, directed_graph, timeout=0.01, roots=(root1, root2)
                )
            else:
                row_distances[i] = 0
        # Remove possible nans, resulting from timeouts (or so I assume).
        # We currently use an arbitrary measure of twice the maximum distance.
        # This is fine for now, but should be adjusted to something more reasonable.
        max = np.nanmax(row_distances, axis=0)
        row_distances[np.isnan(row_distances)] = max * 2
        return row_distances

    def get_root(self, graph: nx.DiGraph) -> str:
        return [n for n, d in graph.in_degree() if d == 0][0]

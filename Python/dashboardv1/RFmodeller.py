from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist
from collections import deque
import multiprocessing as mp
import pandas as pd
import networkx as nx
import numpy as np


class RFmodeller:
    def __init__(self, data: pd.DataFrame, feature_list: list = None):
        self.data = data
        self.features = feature_list
        (
            self.model,
            self.train_X,
            self.val_X,
            self.train_y,
            self.val_y,
        ) = self.train_model()
        self.directed_graphs = self.create_dot_trees()
        self.clustering = self.calculate_tree_clusters()

    # Regression...? Should be a classification problem
    def train_model(self):
        self.data = self.data.dropna(axis=0)
        y = self.data.Price
        X = self.data[self.features]
        train_X, val_X, train_y, val_y = train_test_split(
            X, y, random_state=0, test_size=0.25
        )
        forest_model = RandomForestRegressor(
            n_estimators=50, random_state=1, max_depth=4, n_jobs=-1
        )
        forest_model.fit(train_X, train_y)
        return forest_model, train_X, val_X, train_y, val_y

    def create_dot_trees(self) -> list[nx.DiGraph]:
        # TODO:
        # This is also really slow still. Maybe I can find a more efficient way
        # of converting the trees from into dot format and then into a DG.
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
        # that information.
        # 2. Find a more efficient way to do this. It scales very poorly with tree
        # depth. Check if it also scales as bad with the number of trees.

        # Problem here is:
        # The unoptimized distance calculation performs very poorly.
        # We therefore choose the optimized version, which returns an iterator.
        # This is what makes the deque part necessary. The deque + .pop() returns
        # the last element of the iterator, which is the lowest value.
        # => the lowest GED (graph edit distance)
        # We also can not use the scipy distance matrix method because it needs
        # vector inputs, but we only have 2 graph objects.

        # Run the calc_dist_matrix method in parallel.
        # I used this idea:
        # https://stackoverflow.com/a/56038389/12355337
        # Every process gets a slice of the list of graphs.
        with mp.Pool() as pool:
            distance_matrix = np.array(
                pool.map(self.calc_dist_matrix_parallel, self.directed_graphs)
            )

        # OUTDATED, Only here for reference
        # Unparallelized:
        # distance_matrix = self.calc_dist_matrix(self.directed_graphs)

        # calculate clusters
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="average"
        ).fit(distance_matrix)
        return clustering

    def calc_dist_matrix_parallel(self, directed_graph: nx.DiGraph) -> np.ndarray:
        # This is still not optimal, because every row and column value is being calculated
        # It would be possible (and smarter) to calculate half the matrix and then mirror it.
        # However, this is difficult, because of the calculations happening in parallel.
        dist_matrix = np.zeros(len(self.directed_graphs))
        for i, graph1 in enumerate(self.directed_graphs):
            if graph1 != directed_graph:
                result_generator = nx.optimize_edit_paths(
                    graph1, directed_graph, timeout=1
                )
                dist_matrix[i] = deque(result_generator, maxlen=1).pop()[2]
            else:
                dist_matrix[i] = 0
        return dist_matrix

    # OUTDATED, only here for reference
    def calc_dist_matrix(self, directed_graphs: list[nx.DiGraph]) -> np.ndarray:
        dist_matrix = np.empty([len(directed_graphs)] * 2, nx.DiGraph)
        for i, graph1 in enumerate(directed_graphs):
            for j, graph2 in enumerate(directed_graphs):
                if graph1 != graph2:
                    dist_matrix[i, j] = deque(
                        nx.optimize_graph_edit_distance(graph1, graph2),
                        maxlen=1,
                    ).pop()
                else:
                    dist_matrix[i, j] = 0
        return dist_matrix

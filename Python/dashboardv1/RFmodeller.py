from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist
from collections import deque
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
            n_estimators=10, random_state=1, max_depth=4, n_jobs=-1
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

        distance_matrix = np.asarray(
            [
                [
                    deque(
                        nx.optimize_graph_edit_distance(p1, p2, upper_bound=10.0),
                        maxlen=1,
                    ).pop()
                    if p1 != p2
                    else 0.0
                    for p2 in self.directed_graphs
                ]
                for p1 in self.directed_graphs
            ]
        )

        # calculate clusters
        clustering = AgglomerativeClustering(
            affinity="precomputed", linkage="average"
        ).fit(distance_matrix)
        return clustering

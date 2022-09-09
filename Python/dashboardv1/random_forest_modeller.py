"""
warnings, timeit, datetime, chainmap and numpy are mostly used for utility stuff.
multiprocessing is used for parallelization of the graph edit distance.
sklearn is used for the random forest classifier, the clustering and the
tsne embedding.
pandas is handling the dataframes in the background
networkx is used for the graph edit distance
streamlit is only used in this class for caching and the loading spinner
pygraphviz is used to convert the sklearn tree to pygraph and then networkx
"""
import warnings
from timeit import default_timer as timer
from datetime import timedelta
from collections import ChainMap
import multiprocessing as mp
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import pandas as pd
import networkx as nx
import numpy as np
import numpy.typing as npt
import streamlit as st
import pygraphviz as pgv


class RFmodeller:
    """
    Handles the creation of the random forest model, the clustering and the tsne embedding.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_list: list[str],
        target: list[str],
        target_names: list[str],
        n_estimators: int = 100,
    ):
        self.data = data
        self.features = feature_list
        self.target = target
        self.target_names = target_names
        (
            self.model,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self.train_model(n_estimators)
        self.directed_graphs = self.create_dot_trees()
        self.distance_matrix = self.compute_distance_matrix()
        (
            self.clustering,
            self.cluster_df,
        ) = self.calculate_tree_clusters()
        (self.tsne_embedding, self.tsne_df) = self.calculate_tsne_embedding()
        self.silhouette_scores_df = self.calculate_silhouette_scores_df()

    def train_model(self, n_estimators=100):
        """
        Standard RF classification model
        """
        # TODO: Add support for categorical feature input as in the mushroom dataset
        # I could do this by checking for the session state
        # However it is probably best to instead add a field in the DataLoader class
        # to indicate whether and which categorical features are present.
        # That way it would work on any dataset, which is configured correctly.
        x = self.data[self.features]
        y = self.data[self.target]
        # Have to run this with the .values on X and y, to avoid passing the series with
        # field names etc.
        x_train, x_test, y_train, y_test = train_test_split(
            x.values, y.values, test_size=0.3, random_state=123
        )
        # 100 estimators and depth of 6 works fine with around 15 secs of processing time.
        forest_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=123,
            oob_score=True,
            n_jobs=-1,
        )
        forest_model.fit(x_train, y_train.ravel())  # type: ignore
        return forest_model, x_train, x_test, y_train, y_test

    def create_dot_trees(self) -> list[nx.DiGraph]:
        """
        Transform the sklearn estimators of Tree class to nxDiGraphs
        """
        # TODO:
        # Not a priority!
        # Changed to avoid using .dot files
        # Need a to verify that information is preserved.
        directed_graphs = []
        for estimator in self.model.estimators_:
            pgv_tree_string = tree.export_graphviz(
                estimator, feature_names=self.features
            )
            pgv_digraph = pgv.AGraph(directed=True)
            pgv_digraph = pgv.AGraph(pgv_tree_string)
            nx_digraph = nx.DiGraph()
            nx_digraph = nx.nx_agraph.from_agraph(pgv_digraph)
            directed_graphs.append(nx_digraph)
        return directed_graphs

    def calculate_tsne_embedding(self):
        tsne = TSNE(
            n_components=2,
            perplexity=5,
            early_exaggeration=4,
            learning_rate=100,  # type: ignore
            n_iter=1000,
            random_state=123,
            metric="precomputed",
            init="random",
            verbose=0,
        )
        tsne_embedding = tsne.fit_transform(self.distance_matrix)
        tsne_df = pd.DataFrame(tsne_embedding, columns=["Component 1", "Component 2"])
        return tsne_embedding, tsne_df

    def calculate_tree_clusters(self):
        clustering = DBSCAN(
            # best combination so far: eps=0.1, min_samples=2
            eps=0.3,
            min_samples=3,
            metric="precomputed",
            n_jobs=-1,
            algorithm="brute",
            p=2,
        ).fit(self.distance_matrix)

        cluster_df = pd.DataFrame(
            {
                "cluster": clustering.labels_,
                "tree": list(range(len(self.directed_graphs))),
            }
        )
        return clustering, cluster_df

    @st.cache(suppress_st_warning=True)
    def compute_distance_matrix(self):
        """
        Calculate the pairwise distance matrix for the directed graphs
        We use graph edit distance as the distance metric.
        """
        st.spinner()
        start = timer()
        # I used this idea: https://stackoverflow.com/a/56038389/12355337
        # Every process gets a slice of the list of graphs.
        # sklearn's pdist won't work because it needs numeric value inputs.
        with mp.Pool() as pool:
            distance_matrix_rows = pool.map(
                self.calc_dist_matrix_parallel, self.directed_graphs
            )
        # Transform list of dicts into a single dict.
        distance_matrix_dict = dict(ChainMap(*distance_matrix_rows))
        # Assemble distance matrix based on line indices
        distance_matrix = np.zeros(
            (len(self.directed_graphs), len(self.directed_graphs))
        )
        # Transform the dict into a numpy array matrix
        for i in range(len(distance_matrix_dict)):
            distance_matrix[i] = distance_matrix_dict.get(i)
        # https://stackoverflow.com/questions/16444930/copy-upper-triangle-to-lower-triangle-in-a-python-matrix
        distance_matrix = (
            distance_matrix + distance_matrix.T - np.diag(np.diag(distance_matrix))
        )
        distance_matrix = remove_possible_nans(distance_matrix)
        if not self.dist_matr_shape_ok(distance_matrix):
            raise ValueError(
                "RFModeller: Error after calculating distance matrix. Distance matrix shape is not correct."
            )
        stop = timer()
        print(
            f"Time spent in calculcate_distance_matrix: {timedelta(seconds=stop-start)}"
        )

        return distance_matrix

    def calc_dist_matrix_parallel(
        self, directed_graph: nx.DiGraph
    ) -> dict[int, npt.NDArray[np.float64]]:
        """
        One directed graph will be sent to this method per process.
        We calculate the distances per row.
        Each row is then collected by the multiprocessing pool, which results in the
        distance matrix.
        """
        # This is the smart version of the above method.
        # TODO: Verify how graph_edit_distance works
        # Does it take node labels, split points, etc. into account?
        row_distances = np.zeros(len(self.directed_graphs))
        dg_index = self.directed_graphs.index(directed_graph)
        for i, graph1 in enumerate(self.directed_graphs[dg_index:]):
            if i == 0:
                row_distances[i] = 0
            else:
                # Get this out of the loop and create a dict maybe?
                # No need to compute this on every loop iteration.
                root1 = get_root(graph1)
                root2 = get_root(directed_graph)
                row_distances[i + dg_index] = nx.graph_edit_distance(
                    graph1, directed_graph, timeout=0.04, roots=(root1, root2)
                )

        return {dg_index: row_distances}

    def dist_matr_shape_ok(self, distance_matrix: np.ndarray):
        return distance_matrix.shape == (
            len(self.directed_graphs),
            len(self.directed_graphs),
        )

    def calculate_silhouette_scores_df(self):
        cluster_silhouette_score = silhouette_score(
            X=self.distance_matrix,
            labels=self.cluster_df["cluster"],
            metric="precomputed",
            sample_size=None,
        )
        print(f"Silhouette score: {cluster_silhouette_score}")
        return pd.DataFrame(
            silhouette_samples(
                X=self.distance_matrix,
                labels=self.cluster_df["cluster"].values,
                metric="precomputed",
            ),
            columns=["Silhouette Score"],
        )


def remove_possible_nans(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Remove possible nans, resulting from timeouts in the nx.graph_edit_distance
    (or so I assume).
    """
    # We currently use an arbitrary measure of twice the maximum distance.
    # This is fine for now, but should be adjusted to something more reasonable.
    nan_count = np.count_nonzero(np.isnan(distance_matrix))
    if nan_count > distance_matrix.shape[0]:
        warnings.warn(
            f"{nan_count} NaNs in distance matrix. Consider adjusting timeout parameter in nx.graph_edit_distance."
        )
    square_max = pow(np.nanmax(distance_matrix), 2)
    distance_matrix = np.nan_to_num(distance_matrix, nan=square_max)
    return distance_matrix


def get_root(graph: nx.DiGraph) -> str:
    # TODO: check if this is really no problem and if in_degree is always > 0
    return [n for n, d in graph.in_degree() if d == 0][0]  # type: ignore

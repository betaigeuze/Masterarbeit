import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf1 = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf1.fit(X_train, y_train)

clf2 = DecisionTreeClassifier(max_leaf_nodes=3, random_state=10)
clf2.fit(X_train, y_train)

DG1 = nx.DiGraph()
DG2 = nx.DiGraph()

clf1_graphviz = tree.export_graphviz(clf1, out_file="tree1.dot")
clf2_graphviz = tree.export_graphviz(clf2, out_file="tree2.dot")

DG1 = nx.nx_agraph.read_dot(clf1_graphviz)
DG2 = nx.nx_agraph.read_dot(clf2_graphviz)

print(nx.graph_edit_distance(DG1, DG2))

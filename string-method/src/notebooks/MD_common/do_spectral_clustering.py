import argparse
import sys
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn import cluster, datasets
from scipy.optimize import minimize
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from scipy.spatial.distance import squareform, pdist
from scipy.sparse.linalg import eigsh, lobpcg
from mpl_toolkits.mplot3d import Axes3D
from spherical_kmeans import SphericalKMeans
from sklearn.utils.graph import graph_shortest_path
from sklearn.metrics import silhouette_score
import networkx as nx
import os
from MD_spectral_clustering import SpectralClustering

parser = argparse.ArgumentParser(
    epilog='Perform spectral clustering with spherical k-means on given data. Annie Westerlund 2017.');

clustering = SpectralClustering();
clustering.initialization(parser);
clustering.cluster();

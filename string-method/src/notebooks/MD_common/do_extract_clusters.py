import argparse
import sys
from itertools import combinations
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, leaders
import MD_fun
import CreateRepresentativeStructureFromCluster as CRS
import math
from MD_extract_clusters import ExtractClusters

parser = argparse.ArgumentParser(
    epilog='Use trajectory and cluster indices to construct clusters and representative structure. Annie Westerlund 2016.');
# parser.add_argument('-ind','--index_file',help='The cluster index file.',default='cluster_indices.txt');

clusterer = ExtractClusters();
clusterer.main(parser);

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


class ExtractClusters:
    save_folder_ = "";
    index_file_name_ = "";

    figure_counter_ = 1;
    file_end_name_ = "";

    fun = MD_fun.MD_functions();
    clusterSaver = CRS.SaveClusters();

    def readClusterIndicesFromFile(self, traj):
        cluster_indices = np.loadtxt(self.index_file_name_);
        dx = int(np.ceil(float(traj.n_frames) / float(len(cluster_indices))));

        traj = traj[::dx];

        print traj
        return cluster_indices, traj;

    def clusterData(self, traj):

        cluster_indices, traj = self.readClusterIndicesFromFile(traj);

        # Save structures and trajectories
        self.clusterSaver.createClusterTrajectories(traj, cluster_indices, self.save_folder_, self.file_end_name_);
        self.clusterSaver.getMinRMSDstructures(traj, cluster_indices, self.save_folder_, self.file_end_name_);
        return;

    def main(self, parser, input_args=None, input_traj=None):
        parser.add_argument('-ind', '--index_file', help='The cluster index file.', default='cluster_indices.txt');
        traj, args = self.fun.initialize_trajectory(parser, input_args);
        self.file_end_name_ = args.file_end_name;
        self.save_folder_ = args.out_directory;
        self.index_file_name_ = args.index_file;
        if input_traj is None:
            self.clusterData(traj);
        else:
            self.clusterData(input_traj)

# parser = argparse.ArgumentParser(epilog='Use trajectory and cluster indices to construct clusters and representative structure. Annie Westerlund 2016.');

# clusterer = ExtractClusters(); 
# clusterer.main(parser);

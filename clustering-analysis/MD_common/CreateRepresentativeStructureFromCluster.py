from itertools import combinations
import mdtraj as md
import numpy as np
import math
import os


class SaveClusters():
    def createClusterTrajectories(self, traj, cluster_indices, save_folder="", file_end_name=""):

        if not os.path.exists(save_folder + "clustered_frames/"):
            os.makedirs(save_folder + "clustered_frames/")

        print "Write trajectories to file"

        maxClusters = cluster_indices.max()
        print "Number of clusters: " + str(maxClusters)
        
        for i in range(1, int(maxClusters) + 1):
            continue #disabled for now, quick fix
            tmpTraj = traj[cluster_indices == i]

            print "Original cluster: "
            print tmpTraj

            tmpTraj.save_dcd(save_folder + "clustered_frames/cluster_" + file_end_name + str(i) + ".dcd")

            # Save first and second half of trajectory
            fInd = int(np.floor(tmpTraj.n_frames / 2))
            tmpTraj1 = tmpTraj[0:fInd]
            tmpTraj2 = tmpTraj[fInd + 1::]

            tmpTraj1.save_dcd(save_folder + "clustered_frames/cluster_" + file_end_name + str(i) + "_half1.dcd")
            tmpTraj2.save_dcd(save_folder + "clustered_frames/cluster_" + file_end_name + str(i) + "_half2.dcd")

        return

    def getRepresentativeCentroidStructures(self, traj, cluster_indices, save_folder="", file_end_name=""):

        if not os.path.exists(save_folder + "clustered_frames/"):
            os.makedirs(save_folder + "clustered_frames/")

        print "write representative structure"
        maxClusters = cluster_indices.max()
        for i in range(1, int(maxClusters) + 1):
            tmpTraj = traj[cluster_indices == i]
            meanTraj = tmpTraj[0]

            # Compute mean coordinates
            meanTraj.xyz = np.mean(tmpTraj.xyz, axis=0)
            cluster_rmsds = md.rmsd(tmpTraj, meanTraj)

            # Find the frame which is closest to mean structure
            minIndex = np.argmin(cluster_rmsds)
            representative_structure = tmpTraj[minIndex]

            # Write frame to file
            representative_structure.save_dcd(
                save_folder + "clustered_frames/repr_cluster_" + file_end_name + str(i) + ".dcd")

        return

    def getMinRMSDstructures(self, traj, cluster_indices, save_folder="", file_end_name=""):

        if not os.path.exists(save_folder + "clustered_frames/"):
            os.makedirs(save_folder + "clustered_frames/")

        print "Compute cluster representative structures"
        maxClusters = cluster_indices.max()
        center_indices = np.zeros(int(maxClusters))

        for i in range(1, int(maxClusters) + 1):
            print "Writing structure for cluster: " + str(i)
            tmpTraj = traj[cluster_indices == i]
            meanTraj = tmpTraj[0]

            cluster_rmsds = np.zeros((int(tmpTraj.n_frames), int(tmpTraj.n_frames)))

            # Compute RMSD with respect to the jth frame
            for j in range(0, int(tmpTraj.n_frames)):
                cluster_rmsds[j, j + 1::] = md.rmsd(tmpTraj[j + 1::], tmpTraj[j])

            cluster_rmsds = cluster_rmsds + cluster_rmsds.T
            cluster_rmsds = np.sum(cluster_rmsds, axis=0)

            # Find the frame which is closest to all other structures
            minIndex = np.argmin(cluster_rmsds)
            representative_structure = tmpTraj[minIndex]
            center_indices[i - 1] = minIndex
            # Write frame to file
            representative_structure.save_dcd(
                save_folder + "clustered_frames/reps_cluster_" + file_end_name + str(i) + ".dcd")

        np.savetxt(save_folder + "center_indices" + file_end_name + ".txt", center_indices)

        return

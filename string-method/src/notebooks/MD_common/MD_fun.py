import mdtraj as md
import argparse
import numpy as np
import math as math
from sklearn.decomposition import PCA
from sklearn import linear_model
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import squareform
from sklearn.metrics import mutual_info_score
from sklearn.neighbors.kde import KernelDensity
from joblib import Parallel, delayed
from timeit import default_timer as timer
import multiprocessing
import os


class MD_functions:
    global global_cache
    save_folder = '';

    figure_counter = 1;
    simulation_time = 0;
    file_name_end = "";
    native_state_holo = 0;
    native_state_apo = 0;
    temperature = 300;
    boltzmann_const = 0.0019872041;  # kcal/(mol*K)

    nResidShift = 5;  # Removed the first 5 residues => have to shift residue number
    sub_units = [];

    def initialize_trajectory(self, parser, input_args=None):

        parser.add_argument('-top', '--topology_file', help='Input 1 topology file (.gro, .pdb, etc)', type=str,
                            default='');
        parser.add_argument('-trj', '--trajectory_files', help='Input trajectory files (.xtc, .dcd, etc)', nargs='+',
                            default='');
        parser.add_argument('-fe', '--file_end_name', type=str, help='Output file end name (optional)', default='');
        parser.add_argument('-od', '--out_directory', type=str,
                            help='The directory where data should be saved (optional)', default='');

        # parser.add_argument('-nhtrj','--nat_holo_traj',help='Input native holo trajectory file (optional)',default='');
        # parser.add_argument('-nhtop','--nat_holo_top',help='Input native holo topology file (optional)',default='');

        # parser.add_argument('-natrj','--nat_apo_traj',help='Input native apo trajectory file (optional)',default='');
        # parser.add_argument('-natop','--nat_apo_top',help='Input native apo topology file (optional)',default='');

        parser.add_argument('-build', '--build_subunits', help='Superpose the sub-units (optional).',
                            action='store_true');
        parser.add_argument('-sub_unit_top', '--sub_unit_topology_files',
                            help='All topology files for sub-units. Used when superposing sub-units (optional).',
                            nargs='+', default='');

        parser.add_argument('-dt', '--dt', help='Keep every dt frame.', default=1);

        args = parser.parse_args() if input_args is None else parser.parse_args(input_args);

        # Get command line input parameters
        self.save_folder = args.out_directory;
        self.file_name_end = args.file_end_name;

        # Put / at end of out directory if not present. Check so that folder extsts, otherwise construct it.
        if self.save_folder != '':
            if self.save_folder[-1] != '/':
                self.save_folder += '/';

            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder);

        print('Saving output files in directory: ' + self.save_folder);

        self.figure_counter = 1;

        if args.trajectory_files is not None and len(args.trajectory_files) > 0:
            if not (args.build_subunits):
                # Get the main trajectory
                traj = self.getTrajectory(args.topology_file, args.trajectory_files);
            else:
                # Build sub-units if specified
                traj, self.sub_units = self.buildTrajectoryFromSubunits(args.sub_unit_topology_files);
            traj = traj[::int(args.dt)];
        else:
            traj = None
        # # Set native trajectories
        # if args.nat_holo_top != '':
        #     if args.nat_holo_traj != '':
        #         nat_traj = md.load(args.nat_holo_traj, top = args.nat_holo_top, stride=1);
        #         self.native_state_holo = nat_traj;
        #         print('Holo native trajectory set');
        #     else:
        #         self.native_state_holo = traj;
        #         print('Warning! Native holo state is the same as input trajectory. Use -nhtop and -nhtrj to set topology and trajectory files');
        # else:
        #     self.native_state_holo = traj;
        #     print('Warning! Native holo state is the same as input trajectory. Use -nhtop and -nhtrj to set topology and trajectory files.');



        # if args.nat_apo_top != '':
        #     if args.nat_apo_traj != '':
        #         nat_traj = md.load(args.nat_apo_traj, top = args.nat_apo_top, stride=1);
        #         self.native_state_apo = nat_traj;
        #         print('Apo native trajectory set');
        #     else:
        #         self.native_state_apo = traj;
        #         print('Warning! Native apo state is the same as input trajectory. use -natop and -natrj to set topology and trajectory files.');
        # else:
        #     self.native_state_apo = traj;
        #     print('Warning! Native apo state is the same as input trajectory. Use -natop and -natrj to set topology and trajectory files.');

        print('File end name: ' + self.file_name_end);

        # Keep every dt:th frame (dt=1 by default)

        return traj, args;

    def getTrajectory(self, topology_file, trajectory_files):
        tmpTrajectoryString = "Trajectory files: ";
        tmpTopologyString = "Topology files: " + topology_file;

        # Print file names in string
        for i in range(0, len(trajectory_files)):
            tmpTrajectoryString += trajectory_files[i] + " ";

        print tmpTrajectoryString;
        print tmpTopologyString;

        # Stack trajectories
        traj = md.load(trajectory_files[0], top=topology_file, stride=1)

        for i in range(1, len(trajectory_files)):
            print "Stacking extra trajectory: " + str(i);

            tmpTrajectory = md.load(trajectory_files[i], top=topology_file, stride=1);
            traj = traj.join(tmpTrajectory);
            print "Number of frames: " + str(traj.n_frames);

        return traj;

    def buildTrajectoryFromSubunits(self, topology_files):
        print('Join sub-units');
        traj = md.load_pdb(topology_files[0]);
        sub_units = [traj];
        for i in range(1, len(topology_files)):
            tmpTrajectory = md.load_pdb(topology_files[i], stride=1);
            traj = traj.stack(tmpTrajectory);
            sub_units.append(tmpTrajectory);

        print('Write to file');
        traj.save_pdb(self.save_folder + self.file_name_end + ".pdb");
        print(self.save_folder + self.file_name_end + ".pdb");
        return traj, sub_units;

    def getSubUnits(self):
        return self.sub_units;

    def getFileEndName(self):
        return self.file_name_end;

    def getSaveFolder(self):
        return self.save_folder;

    def saveTime(self):
        # Set parameters
        if traj.n_frames > 1:
            self.simulation_time = traj.time * traj.timestep / 1000.0;  # Time [nanoseconds]
        else:
            self.simulation_time = np.zeros(1);

        # Save simulation time
        np.savetxt(self.save_folder + "time_" + self.file_name_end + ".txt", self.simulation_time);
        return;

    def filterTrajectories(self, traj, startID, endID):
        # Trims the trajectories given start and end resids.
        selection = 'residue ' + str(startID) + ' to ' + str(endID);

        indices = traj.topology.select(selection);
        traj = traj.atom_slice(indices);

        indices = self.native_state_holo.topology.select(selection);
        self.native_state_holo = self.native_state_holo.atom_slice(indices);

        indices = self.native_state_apo.topology.select(selection);
        self.native_state_apo = self.native_state_apo.atom_slice(indices);

        print self.native_state_apo
        print self.native_state_holo
        return traj;

    def getStartAndEndResidues(self, traj):
        # Get the highest start residue and smallest end residue IDs for the trajectories.
        startID_vector = np.zeros(3);
        endID_vector = np.zeros(3);

        # Get start and end for input trajectory
        res = traj.topology.residue(0);
        tmpID = filter(str.isdigit, str(res))
        startID_vector[0] = int(tmpID) + 1;

        res = traj.topology.residue(-1);
        tmpID = filter(str.isdigit, str(res))
        endID_vector[0] = int(tmpID) - 1;

        # # Get start and end for native holo trajectory              
        # res = self.native_state_holo.topology.residue(0);
        # tmpID = filter(str.isdigit, str(res))
        # startID_vector[1] = int(tmpID) + 1;

        # res = self.native_state_holo.topology.residue(-1);
        # tmpID = filter(str.isdigit, str(res))
        # endID_vector[1] = int(tmpID) - 1;

        # # Get start and end for native apo trajectory
        # res = self.native_state_apo.topology.residue(0);
        # tmpID = filter(str.isdigit, str(res))
        # startID_vector[2] = int(tmpID) + 1;

        # res = self.native_state_apo.topology.residue(-1);
        # tmpID = filter(str.isdigit, str(res))
        # endID_vector[2] = int(tmpID) - 1;

        # Compute the start and end ID
        startID = int(np.max(startID_vector));
        endID = int(np.min(endID_vector));

        return startID, endID;

    def hydrophobic_Cb_coordination_number(self, traj, topology, plotter):
        # Coordination number of hydrophobic Cb contacts
        CB = topology.select(
            "protein and name CB and (resname ALA or resname VAL or resname LEU or resname ILE or resname PRO or resname PHE or resname MET or resname TRP)");
        CB_pairs = np.array(
            [(i, j) for (i, j) in combinations(CB, 2)
             if abs(topology.atom(i).residue.index - \
                    topology.atom(j).residue.index) > 3])

        r = md.compute_distances(traj, CB_pairs, periodic=False);
        hydr_contacts = np.sum(r[:, :] < 0.6, axis=1);
        np.savetxt(self.save_folder + 'hydr_contacts_' + self.file_name_end + '.txt', hydr_contacts);

        # Plot coordination numbers
        self.producePlots(hydr_contacts, "Hydrophobic contacts", plotter);
        return;

    def C_alpha_coordination_number(self, traj, topology, plotter):
        # Coordination number of CA contacts
        CA_ind = topology.select("protein and name CA");
        CA_pairs = np.array(
            [(i, j) for (i, j) in combinations(CA_ind, 2)
             if abs(topology.atom(i).residue.index - \
                    topology.atom(j).residue.index) > 3])
        r = md.compute_distances(traj, CA_pairs, periodic=False);
        CA_contacts = np.sum(r[:, :] < 0.6, axis=1);
        np.savetxt(self.save_folder + 'c_alpha_contacts_' + self.file_name_end + '.txt', CA_contacts);

        self.producePlots(CA_contacts, "Coordination number of C_alpha contacts", plotter);
        return;

    def DSSP_structure(self, traj, plotter, doPlotting=False, saveDeltaDSSP=False):

        traj_holo = self.native_state_holo[0];
        traj_apo = self.native_state_apo;

        # Compute the secondary structure frequency of the trajectory
        dssp_helix, dssp_coil, dssp_strand = self.secondary_structure_frequency(traj, plotter);

        if doPlotting or saveDeltaDSSP:
            dssp_holo_hel, dssp_holo_coil, dssp_holo_strand = self.DSSP_content(traj_holo, plotter);
            dssp_apo_hel, dssp_apo_coil, dssp_apo_strand = self.DSSP_content(traj_apo, plotter);

        if doPlotting:
            fig = plotter.figure(self.figure_counter);
            axis = fig.add_subplot(131);
            axis.plot(dssp_helix, color="red", label='Helix', marker="o");
            axis.plot(dssp_coil, color="blue", label='Coil', marker="o");
            axis.plot(dssp_strand, color="green", label='Strand', marker="o");
            plotter.title("DSSP " + self.file_name_end);
            plotter.xlabel("Residue ID");
            plotter.ylabel("Frequency");

            axis = fig.add_subplot(132);
            axis.plot(dssp_holo_hel - dssp_helix, color="red", label='Helix', marker="o");
            axis.plot(dssp_holo_coil - dssp_coil, color="blue", label='Coil', marker="o");
            axis.plot(dssp_holo_strand - dssp_strand, color="green", label='Strand', marker="o");
            plotter.title("dDSSP Holo");
            plotter.xlabel("Residue ID");
            plotter.ylabel("Frequency");

            axis = fig.add_subplot(133);
            axis.plot(dssp_apo_hel - dssp_helix, color="red", label='Helix', marker="o");
            axis.plot(dssp_apo_coil - dssp_coil, color="blue", label='Coil', marker="o");
            axis.plot(dssp_apo_strand - dssp_strand, color="green", label='Strand', marker="o");
            plotter.title("dDSSP Apo");
            plotter.xlabel("Residue ID");
            plotter.ylabel("Frequency");

            plotter.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

        np.savetxt(self.save_folder + 'DSSP_hel_' + self.file_name_end + '.txt', dssp_helix);
        np.savetxt(self.save_folder + 'DSSP_coil_' + self.file_name_end + '.txt', dssp_coil);
        np.savetxt(self.save_folder + 'DSSP_strand_' + self.file_name_end + '.txt', dssp_strand);

        if saveDeltaDSSP:
            np.savetxt(self.save_folder + 'dDSSP_holo_hel_' + self.file_name_end + '.txt', dssp_holo_hel - dssp_helix);
            np.savetxt(self.save_folder + 'dDSSP_holo_coil_' + self.file_name_end + '.txt', dssp_holo_coil - dssp_coil);
            np.savetxt(self.save_folder + 'dDSSP_holo_strand_' + self.file_name_end + '.txt',
                       dssp_holo_stran - dssp_strand);
            np.savetxt(self.save_folder + 'dDSSP_apo_hel_' + self.file_name_end + '.txt', dssp_apo_hel - dssp_helix);
            np.savetxt(self.save_folder + 'dDSSP_apo_coil_' + self.file_name_end + '.txt', dssp_apo_coil - dssp_coil);
            np.savetxt(self.save_folder + 'dDSSP_apo_strand_' + self.file_name_end + '.txt',
                       dssp_apo_stran - dssp_strand);

        self.figure_counter += 1;

        return;

    def secondary_structure_frequency(self, traj, plotter):

        dssp = md.compute_dssp(traj, simplified=True)
        alpha_content = np.sum(dssp == "H", axis=1)
        strand_content = np.sum(dssp == "E", axis=1)
        coil_content = np.sum(dssp == "C", axis=1)

        residue_dssp_helix = np.zeros(len(dssp[0, :]));
        residue_dssp_coil = np.zeros(len(dssp[0, :]));
        residue_dssp_strand = np.zeros(len(dssp[0, :]));

        # Derive secondary structure content distribution
        for i in range(0, len(dssp[:, 0])):
            for j in range(0, len(dssp[0, :])):
                if dssp[i, j] == 'H':
                    residue_dssp_helix[j] += 1;
                elif dssp[i, j] == 'C':
                    residue_dssp_coil[j] += 1;
                elif dssp[i, j] == 'E':
                    residue_dssp_strand[j] += 1;

        nFrames = len(dssp[:, 0]);

        # Normalize
        for i in range(0, len(residue_dssp_helix)):
            residue_dssp_helix[i] /= nFrames;
            residue_dssp_coil[i] /= nFrames;
            residue_dssp_strand[i] /= nFrames;

        return residue_dssp_helix, residue_dssp_coil, residue_dssp_strand;

    def entropyDihedral(self, traj, plotter):
        indices, angles = md.compute_psi(traj);
        nFrames = len(angles[:, 0]);
        entropy = np.zeros(nFrames);

        for i in range(0, nFrames):
            tmpDihedrals = angles[i, :];
            pdf, edges = np.histogram(tmpDihedrals, 100, normed=True);
            for j in range(0, len(pdf)):
                if pdf[j] > 0:
                    entropy[i] -= pdf[j] * math.log(pdf[j]);

        np.savetxt(self.save_folder + 'dihedral_angle_entropy_' + self.file_name_end + '.txt', entropy);

        # Plot dihedral angle entropy
        self.producePlots(entropy, "Dihedral entropy", plotter);
        return;

    def computeVDA(self, traj, index_choice1, index_choice2, index_choice3, index_choice4):

        # Extract the indices from choice and average
        CA_ind1 = traj.topology.select(index_choice1);
        CA_ind2 = traj.topology.select(index_choice2);
        CA_ind3 = traj.topology.select(index_choice3);
        CA_ind4 = traj.topology.select(index_choice4);

        if len(CA_ind1) > 0 and len(CA_ind2) > 0 and len(CA_ind3) > 0 and len(CA_ind4):
            # Get the points
            CA_pts1 = traj.atom_slice(CA_ind1).xyz;
            CA_pts2 = traj.atom_slice(CA_ind2).xyz;
            CA_pts3 = traj.atom_slice(CA_ind3).xyz;
            CA_pts4 = traj.atom_slice(CA_ind4).xyz;

            ion1_pos = np.mean(CA_pts1, axis=1);
            ion2_pos = np.mean(CA_pts2, axis=1);
            ion3_pos = np.mean(CA_pts3, axis=1);
            ion4_pos = np.mean(CA_pts4, axis=1);

            # set ion to trajectory 
            Ca_ion_traj = traj;

            ind1 = Ca_ion_traj.topology.select("protein and residue 22 and name CA")
            ind2 = Ca_ion_traj.topology.select("protein and residue 22 and name CA")
            ind3 = Ca_ion_traj.topology.select("protein and residue 22 and name CA")
            ind4 = Ca_ion_traj.topology.select("protein and residue 22 and name CA")
            inds = [[ind1[0], ind2[0], ind3[0], ind4[0]]];
            Ca_ion_traj.topology.atom(ind1[0]).xyz = ion1_pos;
            Ca_ion_traj.topology.atom(ind2[0]).xyz = ion2_pos;
            Ca_ion_traj.topology.atom(ind3[0]).xyz = ion3_pos;
            Ca_ion_traj.topology.atom(ind4[0]).xyz = ion4_pos;

            # Compute virtual dihedral angle on atom positions
            VDA = md.compute_dihedrals(Ca_ion_traj, inds);
            np.savetxt(self.save_folder + 'VDA_' + self.file_name_end + '.txt', VDA);
        return;

    def computeDRIDdistances(self, traj, plt):
        # Compute a distance matrix between all frames in the trajectory using DRID as metric.
        print "Computing the DRID matrix";
        nFrames = traj.n_frames;
        DRID_distances = np.zeros((nFrames, nFrames));
        for i in range(0, nFrames):
            DRID_distances[i, i::] = self.DRID(traj[i::], traj.topology, plt, traj[i]);

        DRID_distances = DRID_distances + DRID_distances.T;
        return DRID_distances;

    def computeRMSDdistances(self, traj, precentered=False):
        # Compute a distance matrix between all frames in the trajectory using RMSD as metric.
        print "Computing the RMSD matrix";
        nFrames = traj.n_frames;
        RMSD_distances = np.zeros((nFrames, nFrames));
        for i in range(0, nFrames):
            RMSD_distances[i, ::] = md.rmsd(traj, traj, i);

        RMSD_distances = RMSD_distances + RMSD_distances.T;
        print RMSD_distances;
        return RMSD_distances;

    def saveDRIDdistances(self, traj, plotter):
        # Compute DRID distance matrix and save the results.
        DRID_distances = self.computeDRIDdistances(traj, plotter);
        print "Saving matrix";
        dist_mat = squareform(DRID_distances);
        np.savetxt(self.save_folder + 'DRID_distances_compressed_' + self.file_name_end + '.txt', dist_mat);
        return;

    def saveRMSDdistances(self, traj, choice='protein'):
        # Compute RMSD distance matrix and save the results.
        atom_indices = traj.topology.select(choice);

        traj.atom_slice(atom_indices, inplace=True);

        print "Atom choice: " + choice;
        print traj;

        traj.center_coordinates();
        RMSD_distances = self.computeRMSDdistances(traj, precentered=True);
        dist_mat = squareform(RMSD_distances);
        np.savetxt(self.save_folder + 'RMSD_distances_compressed_' + self.file_name_end + '.txt', dist_mat);
        return;

    def getContactMap(self, distanceMatrix, cutoff, doNormalization=False, makeBinary=False):
        # Construct a contact map from a distance matrix using a cutoff. Contact map can be made binary as well.
        nRows = len(distanceMatrix[::, 0]);
        contactMap = distanceMatrix;

        for i in range(0, nRows):
            tmpVec = distanceMatrix[i, ::];
            contactMap[i, ::] = tmpVec * (tmpVec <= cutoff);

        # Normalize contact map
        if doNormalization:
            contactMap = contactMap / cutoff;

        # Make the contacts binary
        if makeBinary:
            contactMap = contactMap > 0;

        return contactMap;

    def getAllCalphaDistances(self, traj, startInd=-1, endInd=-1, query="protein and name CA"):

        # Construct the distance matrices (nFrames-residue-residue) with the distance 
        # between residues defined as the minimum distance between all heavy atoms of the two residues.
        nFrames = int(traj.n_frames);
        nResidues = int(traj.n_residues);
        allInds = traj.topology.select(query);  # CHANGED BY Oliver

        # Do atom selections, save list with all heavy atoms.
        # for i in range(0,nResidues):
        #     # OBS! Choice is now done on "residue". Can be a problem for multi-chain proteins. 
        #     # Then, switch to resid or feed chains separately.
        #     choice = "protein and resid " + str(i) + " and name CA";
        #     tmpInd = traj.topology.select(choice);
        #     if len(tmpInd) == 0:
        #         print("WARN No atoms matching select query '" + choice + "'. Maybe it's not part of the protein. Skipping residue from distance matrix...")
        #         continue
        #     allInds.append(tmpInd);


        nResidues = int(len(allInds));

        distanceMatrices = np.zeros((nFrames, nResidues, nResidues));

        # Compute distance matrix
        for i in range(0, nResidues):
            if i % 100 == 1:
                print("Distance for residue " + str(i) + "/" + str(nResidues));
            for j in range(i + 1, nResidues):
                # Get all atom pairs            
                atom_pairs = np.zeros((1, 2));
                atom_pairs[0, 0] = allInds[i];
                atom_pairs[0, 1] = allInds[j];

                distances = md.compute_distances(traj, atom_pairs, periodic=False);

                if len(distances) == 0:
                    print('The chosen residue does not exist!');

                # The distance between residues is min distance between all heavy atoms. 
                # Take residual to get rid of cut-off.
                minDistance = np.min(distances, axis=1);
                res = 1 / minDistance
                distanceMatrices[::, i, j] = res;
                distanceMatrices[::, j, i] = res;
        return distanceMatrices;

    def getAllSideChainMinDistances(self, traj, startInd, endInd):

        # Construct the distance matrices (nFrames-residue-residue) with the distance 
        # between residues defined as the minimum distance between all heavy atoms of the two residues.
        nFrames = int(traj.n_frames);
        nResidues = int(traj.n_residues);
        allInds = [];

        # Do atom selections, save list with all heavy atoms.
        for i in range(0, nResidues):
            # OBS! Choice is now done on "residue". Can be a problem for multi-chain proteins. 
            # Then, switch to resid or feed chains separately.
            choice = "protein and resid " + str(i) + "and !(type H)";
            tmpInd = traj.topology.select(choice);
            allInds.append(tmpInd);

        nResidues = int(len(allInds));

        distanceMatrices = np.zeros((nFrames, nResidues, nResidues));

        # Compute distance matrix
        for i in range(0, nResidues):
            for j in range(i, nResidues):

                atomInd1 = allInds[i];
                atomInd2 = allInds[j];

                # Get all atom pairs            
                atom_pairs = np.zeros((len(atomInd1) * len(atomInd2), 2));
                counter = 0;
                for k in range(0, len(atomInd1)):
                    for l in range(0, len(atomInd2)):
                        atom_pairs[counter, 0] = atomInd1[k];
                        atom_pairs[counter, 1] = atomInd2[l];
                        counter += 1;

                atom_pairs = atom_pairs[0:counter - 1, ::];

                distances = md.compute_distances(traj, atom_pairs, periodic=False);

                if len(distances) == 0:
                    print('The chosen residue does not exist!');

                # The distance between residues is min distance between all heavy atoms. Take mean over all frames.
                distanceMatrices[::, i, j] = np.min(distances, axis=1);
                distanceMatrices[::, j, i] = np.min(distances, axis=1);
        return distanceMatrices;

    def computeFrameToFrameSideChainContacts(self, traj, choice='protein'):

        atom_indices = traj.topology.select(choice);
        traj.atom_slice(atom_indices, inplace=True);
        print('Compute frame to frame sidechain contact map difference');
        print(traj);

        # Compute frame-frame residue contact map norm.
        nFrames = int(traj.n_frames);

        frame2frameContacts = np.zeros((nFrames, nFrames));

        startID, endID = self.getStartAndEndResidues(traj);

        distanceMatrices = self.getAllSideChainMinDistances(traj, startID, endID);

        print('Compute frame to frame distances');
        for i in range(0, nFrames):
            if i % 100 == 1:
                print("frame " + str(i) + '/' + str(nFrames));
            tmpMap1 = self.getContactMap(distanceMatrices[i, ::, ::], 0.8);
            for j in range(i + 1, nFrames):
                tmpMap2 = self.getContactMap(distanceMatrices[j, ::, ::], 0.8);
                frame2frameContacts[i, j] = np.linalg.norm((tmpMap1 - tmpMap2), 2);

        frame2frameContacts = frame2frameContacts + frame2frameContacts.T;

        print(frame2frameContacts);
        distances = squareform(frame2frameContacts);
        # Save distance matrix to file
        np.savetxt(self.save_folder + 'frame_to_frame_side_chain_contacts_' + self.file_name_end + '.txt', distances);
        return;

    def computeFrameToFrameCalphaContacts(self, traj, choice='protein', async=True):

        atom_indices = traj.topology.select(choice);
        traj.atom_slice(atom_indices, inplace=True);
        print('Compute frame to frame calpha sidechain contact map difference');
        print('Atom choice: ' + choice);
        print(traj);

        # Compute frame-frame residue contact map norm.
        nFrames = int(traj.n_frames);

        frame2frameContacts = np.zeros((nFrames, nFrames));

        startID, endID = self.getStartAndEndResidues(traj);

        distanceMatrices = self.getAllCalphaDistances(traj, startID, endID);

        # for sharing array in memory https://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing/5550156#5550156
        # frame2frameContacts = multiprocessing.Array(ctypes.c_double, nFrames*nFrames)
        # frame2frameContacts = np.ctypeslib.as_array(shared_array_base.get_obj())
        # frame2frameContacts = frame2frameContacts.reshape(10, 10)
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cpu_count)

        start = timer()
        print('Compute frame to frame calpha distances. async = ' + str(async));
        tasks = []
        for i in range(0, nFrames):
            # build a list of tasks
            tmpMap1 = distanceMatrices[i, ::, ::];
            for j in range(i + 1, nFrames):
                # plotNum += 1
                tmpMap2 = distanceMatrices[j, ::, ::];
                tasks.append((tmpMap1, tmpMap2))
        if async:
            results = [pool.apply_async(computeContactNorm, t) for t in tasks]
        else:
            results = [computeContactNorm(t[0], t[1]) for t in tasks]
        idx = 0
        for i in range(0, nFrames):
            # build a list of tasks
            for j in range(i + 1, nFrames):
                result = results[idx].get() if async else results[idx]
                # frame2frameContacts[i,j] = np.linalg.norm((tmpMap1-tmpMap2),2);
                frame2frameContacts[i, j] = result;
                idx = idx + 1
        frame2frameContacts = frame2frameContacts + frame2frameContacts.T;
        # print(frame2frameContacts);
        distances = squareform(frame2frameContacts);
        print("Done computing frame to frame calpha distances! Took " + str(timer() - start) + " seconds")
        # Save distance matrix to file
        np.savetxt(self.save_folder + 'frame_to_frame_CA_contacts_' + self.file_name_end + '.txt', distances);
        return distances

    def computeAverageSideChainMinDistanceMap(self, traj, choice='protein'):
        # Construct the average distance matrix (residue-residue) with the distance between residues defined as the minimum distance between all heavy atoms of the two residues.
        atom_indices = traj.topology.select(choice);
        traj.atom_slice(atom_indices, inplace=True);
        print('Compute average sidechain contact map');
        print('Atom choice: ' + choice);
        print(traj);
        startID, endID = self.getStartAndEndResidues(traj);
        distanceMatrix = np.mean(self.getAllSideChainMinDistances(traj, startID, endID), axis=0);

        print distanceMatrix;

        # Save distance matrix to file
        np.savetxt(self.save_folder + 'distance_matrix_min_' + self.file_name_end + '.txt', distanceMatrix);
        return;

    def computeAverageSideChainDistanceMap(self, traj, startInd, endInd):
        # Compute the average distance matrix between all residues
        nResidues = int(traj.n_residues);
        nFrames = int(traj.n_frames);

        side_chain_ind = np.zeros(nResidues);
        distanceMatrix = np.zeros((nResidues, nResidues));

        # Do selections, last C-atom that does not have name C.
        for i in range(startInd, endInd):
            # OBS! "Residue" choice works only for 1-chain proteins. Switch to "resid" for multi-chain.
            choice = "protein and residue " + str(i) + " and type C and not name C";
            tmpInd = traj.topology.select(choice);
            side_chain_ind[i - startInd] = tmpInd[-1];

        # Get all atom pairs
        atom_pairs = np.zeros((nResidues * nResidues, 2));
        counter = 0;
        for i in range(0, nResidues):
            for j in range(i, nResidues):
                atom_pairs[counter, 0] = side_chain_ind[i];
                atom_pairs[counter, 1] = side_chain_ind[j];
                counter += 1;

        atom_pairs = atom_pairs[0:counter - 1, ::];

        # Compute distances
        distances = md.compute_distances(traj, atom_pairs, periodic=False);

        # Construct square form matrix of distances
        counter = 0;
        for i in range(0, nResidues):
            for j in range(i, nResidues):
                distanceMatrix[i, j] = np.mean(distances[::, counter]);
                distanceMatrix[j, i] = np.mean(distances[::, counter]);
                counter += 1;

        # Save distance matrix to file
        np.savetxt(self.save_folder + 'distance_matrix_' + self.file_name_end + '.txt', distanceMatrix);
        return;

    def computeCentroidDistance(self, traj, residues1, residues2, plotter, file_tag='', showPlots=False):
        nFrames = int(traj.n_frames);

        # Compute the sidechain centroids
        first_point = np.mean(residues1.xyz, axis=1);
        second_point = np.mean(residues2.xyz, axis=1);

        distances = np.zeros(nFrames);
        for i in range(0, nFrames):
            distances[i] = np.linalg.norm(first_point[i, ::] - second_point[i, ::]);

        np.savetxt(self.save_folder + 'distances_between_residues_' + file_tag + self.file_name_end + '.txt',
                   distances);
        return;

    def computeProbeDistances(self, traj, probe_residues1, probe_residues2, plotter, showPlots=False):
        nFrames = int(traj.n_frames);

        # Compute the sidechain centroids

        atom_indices1 = traj.topology.select(
            'protein and residue ' + str(probe_residues1[0]) + ' and sidechain and !(type H)');
        atom_indices2 = traj.topology.select(
            'protein and residue ' + str(probe_residues1[1]) + ' and sidechain and !(type H)');

        first_point = np.mean(traj.atom_slice(atom_indices1).xyz, axis=1);
        second_point = np.mean(traj.atom_slice(atom_indices2).xyz, axis=1);

        centroid1 = np.zeros((nFrames, 3));
        for i in range(0, nFrames):
            centroid1[i, 0] = (first_point[i, 0] + second_point[i, 0]) / 2;
            centroid1[i, 1] = (first_point[i, 1] + second_point[i, 1]) / 2;
            centroid1[i, 2] = (first_point[i, 2] + second_point[i, 2]) / 2;

        atom_indices1 = traj.topology.select(
            'protein and residue ' + str(probe_residues2[0]) + ' and sidechain and !(type H)');
        atom_indices2 = traj.topology.select(
            'protein and residue ' + str(probe_residues2[1]) + ' and sidechain and !(type H)');

        first_point = np.mean(traj.atom_slice(atom_indices1).xyz, axis=1);
        second_point = np.mean(traj.atom_slice(atom_indices2).xyz, axis=1);

        centroid2 = np.zeros((nFrames, 3));
        distances = np.zeros(nFrames);
        for i in range(0, nFrames):
            centroid2[i, 0] = (first_point[i, 0] + second_point[i, 0]) / 2;
            centroid2[i, 1] = (first_point[i, 1] + second_point[i, 1]) / 2;
            centroid2[i, 2] = (first_point[i, 2] + second_point[i, 2]) / 2;

            distances[i] = np.sqrt((centroid1[i, 0] - centroid2[i, 0]) ** 2 + (centroid1[i, 1] - centroid2[i, 1]) ** 2 + \
                                   (centroid1[i, 2] - centroid2[i, 2]) ** 2);

        if showPlots:
            self.producePlots(distances, 'Distances', plotter);

        np.savetxt(self.save_folder + 'probe_distances_' + self.file_name_end + '.txt', distances);
        return;

    def computeHelixDirection(self, traj, part, plotter, show_plots=False, save_data=True):
        # Computes the direction of the helix (traj) as the principal eigenvector.
        # Get the atoms to use in PCA
        C_alpha_indices = traj.topology.select("protein and (name CA or name N or name C)");
        CA_points = traj.atom_slice(C_alpha_indices).xyz;
        nFrames = int(traj.n_frames);

        xy_coeff = np.zeros((nFrames, 2));
        xz_coeff = np.zeros((nFrames, 2));

        helix_direction = np.zeros((nFrames, 3));

        for i in range(0, nFrames):
            data = CA_points[i, ::, ::];

            # Subtract centroid
            mu = data.mean(axis=0);
            data = data - mu;
            pca = PCA();
            pca.fit(data);
            eigenvector = pca.components_[0, ::];

            # Pick eigenvector that points in the direction of the atom sequence
            ref_line = data[-1, ::] - data[0, ::];
            ref_line_pi_deg = data[0, ::] - data[-1, ::];

            # Normalize the vectors         
            norm_ = (ref_line[0] ** 2 + ref_line[1] ** 2 + ref_line[2] ** 2);
            ref_line = ref_line / norm_;
            ref_line_pi_deg = ref_line_pi_deg / norm_;

            # Compute distance between eigenvector and reference lines
            diff1 = (ref_line[0] - eigenvector[0]) ** 2 + (ref_line[1] - eigenvector[1]) ** 2 + (ref_line[2] -
                                                                                                 eigenvector[2]) ** 2;
            diff2 = (ref_line_pi_deg[0] - eigenvector[0]) ** 2 + (ref_line_pi_deg[1] - eigenvector[1]) ** 2 + (
                                                                                                              ref_line_pi_deg[
                                                                                                                  2] -
                                                                                                              eigenvector[
                                                                                                                  2]) ** 2;

            # If the eigenvector is more similar to diff2, we flip it
            ch_dir = False;
            if diff1 > diff2:
                old_ev = eigenvector;
                eigenvector = -eigenvector;
                diff1 = (ref_line[0] - eigenvector[0]) ** 2 + (ref_line[1] - eigenvector[1]) ** 2 + (ref_line[2] -
                                                                                                     eigenvector[
                                                                                                         2]) ** 2;
                diff2 = (ref_line_pi_deg[0] - eigenvector[0]) ** 2 + (ref_line_pi_deg[1] - eigenvector[1]) ** 2 + (
                                                                                                                  ref_line_pi_deg[
                                                                                                                      2] -
                                                                                                                  eigenvector[
                                                                                                                      2]) ** 2;

            # Reset data
            data = data + mu;

            # Plot the line
            if show_plots:
                fig = plotter.figure(self.figure_counter);
                ax = fig.add_subplot(111, projection='3d');
                ax.scatter(data[::, 0], data[::, 1], data[::, 2])
                v = np.zeros((2, 3));
                v[0, ::] = mu - eigenvector;
                v[1, ::] = mu + eigenvector;
                ax.plot(v[::, 0], v[::, 1], v[::, 2]);
                plotter.title(part);
                plotter.show();

            if ch_dir:
                fig = plotter.figure(self.figure_counter);
                ax = fig.add_subplot(111, projection='3d');
                ax.scatter(data[::, 0], data[::, 1], data[::, 2])
                v = np.zeros((2, 3));
                v2 = np.zeros((2, 3));
                v[0, ::] = mu;
                v[1, ::] = mu + eigenvector;
                v2[0, ::] = mu;
                v2[1, ::] = mu - eigenvector;
                ax.plot(v[::, 0], v[::, 1], v[::, 2]);
                ax.plot(v2[::, 0], v2[::, 1], v2[::, 2], color='r');

                v[0, ::] = mu;
                v[1, ::] = mu + old_ev;
                v2[0, ::] = mu;
                v2[1, ::] = mu - old_ev;
                ax.plot(v[::, 0], v[::, 1], v[::, 2], marker='o', color='b');
                ax.plot(v2[::, 0], v2[::, 1], v2[::, 2], color='r', marker='o');
                plotter.title(part);
                plotter.show();

            # Store helix vector
            helix_direction[i, ::] = eigenvector;

        if show_plots:
            plotter.title(part);
            plotter.show();
        if save_data:
            np.savetxt(self.save_folder + 'helix_direction_' + part + '_' + self.file_name_end + '.txt',
                       helix_direction);
        return helix_direction;

    def computeDihedralSimilarityMeasure(self, traj, native):

        indices_psi, angles_psi = md.compute_psi(traj);
        indicesNative_psi, anglesNative_psi = md.compute_psi(native);

        indices_phi, angles_phi = md.compute_phi(traj);
        indicesNative_phi, anglesNative_phi = md.compute_phi(native);

        nFrames = len(angles_psi[:, 0]);
        nAngles = len(angles_psi[0, :]);
        correlation = np.zeros(nFrames);

        for i in range(0, nFrames):
            for j in range(0, nAngles):
                # Compute correlation to corresponding dihedral angle in native
                correlation[i] += 0.25 * (1.0 + math.cos(angles_psi[i, j] -
                                                         anglesNative_psi[0, j])) / nAngles + 0.25 * (
                1.0 + math.cos(angles_phi[i, j] - anglesNative_phi[0, j])) / nAngles;

        return correlation;

    def correlationToNativeDihedral(self, traj, plotter, atom_sel='all', domain_name='', nativeHolo=True):
        # Compute similarity between one frame and the native 4CAL frame in terms of similarity between corresponding dihedral angles.

        atom_indices = traj.topology.select(atom_sel);

        if len(atom_indices) > 0:
            trajChoice = traj.atom_slice(atom_indices);
            if nativeHolo:
                nativeChoice = self.native_state_holo.atom_slice(atom_indices);

                # Compute correlation of corresponding dihedral angles in the domain
                correlation = self.computeDihedralSimilarityMeasure(trajChoice, nativeChoice);
                np.savetxt(
                    self.save_folder + 'holo_' + domain_name + '_dihedral_angle_correlation_' + self.file_name_end + '.txt',
                    correlation);
            else:
                nativeChoice = self.native_state_apo.atom_slice(atom_indices);

                # Compute correlation of corresponding dihedral angles in the domain
                correlation = self.computeDihedralSimilarityMeasure(trajChoice, nativeChoice);
                np.savetxt(
                    self.save_folder + 'holo_' + domain_name + '_dihedral_angle_correlation_' + self.file_name_end + '.txt',
                    correlation);

        return correlation;

    def correlationNeighborDihedral(self, traj, plotter):
        # Compare similarity between dihedrals to neighboring dihedrals
        indices, angles = md.compute_psi(traj);

        nFrames = len(angles[:, 0]);
        nAngles = len(angles[0, :]);
        correlation = np.zeros(nFrames);

        for i in range(0, nFrames):
            for j in range(1, nAngles):
                # Check correlation values
                # if i < 1:
                # if 0.5*(1.0 + math.cos(angles[i,j]-angles[i,j-1])) > 0.95:
                # print (angles[i,j]-angles[i,j-1])*180/math.pi;
                #   print angles[i,j], angles[i,j-1];

                # Compute correlation
                correlation[i] += 0.5 * (1.0 + math.cos(angles[i, j] - angles[i, j - 1])) / nAngles;

        np.savetxt(self.save_folder + 'dihedral_angle_neighbor_correlation_' + self.file_name_end + '.txt',
                   correlation);

        # Plot dihedral angle correlation
        self.producePlots(correlation, "Dihedral neighbor correlation", plotter);
        return;

    def radiusOfGyration(self, traj, plotter, atom_sel='all', domain_name=''):

        atom_indices = traj.topology.select(atom_sel);
        trajChoice = traj.atom_slice(atom_indices);

        # radiusOfGyrationHolo = md.compute_rg(self.native_state_holo[0]);
        radiusOfGyration = md.compute_rg(trajChoice);

        if domain_name != '':
            np.savetxt(self.save_folder + domain_name + '_radius_gyration_' + self.file_name_end + '.txt',
                       radiusOfGyration);
        else:
            np.savetxt(self.save_folder + 'radius_gyration_' + self.file_name_end + '.txt', radiusOfGyration);

        # Plot radius of gyration
        self.producePlots(radiusOfGyration, "Radius of gyration", plotter);
        return;

    def solventAccessibleSurfaceArea(self, traj):
        print('Computing average SASA');
        SASA = md.shrake_rupley(traj, mode='residue');
        SASA = np.mean(SASA, axis=0);
        np.savetxt(self.save_folder + 'SASA_' + self.file_name_end + '.txt', SASA);
        return;

    def compute_DRID_holo(self, traj, topology, plotter):

        DRID_holo = self.DRID(traj, topology, plotter);
        np.savetxt(self.save_folder + 'DRID_holo_' + self.file_name_end + '.txt', DRID_holo);

        # Plot the DRID moments
        self.producePlots(DRID_holo, "DRID distance from native holo", plotter);

        return;

    def DRID(self, traj, topology, plotter, native=""):

        if native == "":
            native = self.native_state_holo;

        C_alpha = topology.select("protein and name CA");  # C_alpha indices
        drid_vectors = md.compute_drid(traj, atom_indices=C_alpha);

        C_alpha_native = native.topology.select("protein and name CA");
        drid_native = md.compute_drid(native, atom_indices=C_alpha_native);

        nFrames = len(drid_vectors[:, 0]);
        nAtoms = len(drid_vectors[0, :]) / 3.0;

        DRID_distance = np.zeros(nFrames);

        # first_moment_mean = np.zeros(len(drid_vectors[:,1]));
        # second_moment_mean = np.zeros(len(drid_vectors[:,1]));
        # third_moment_mean = np.zeros(len(drid_vectors[:,1]));

        # for i in range(0,len(drid_vectors[:,1])):
        # first_moment_mean[i] = np.mean(drid_vectors[i,0::3]);
        # second_moment_mean[i] = np.mean(drid_vectors[i,1::3]);
        # third_moment_mean[i] = np.mean(drid_vectors[i,2::3]);

        for i in range(0, nFrames):
            DRID_distance[i] = 1 / (3.0) * np.sum((drid_vectors[i, ::3] - drid_native[0, ::3]) ** 2 + (
            drid_vectors[i, 1::3] - drid_native[0, 1::3]) ** 2 + (drid_vectors[i, 2::3] - drid_native[0, 2::3]) ** 2);

        """fig = plotter.figure(self.figure_counter);
        axis = fig.add_subplot(231);
        axis.plot(self.simulation_time, first_moment_mean);
        plotter.title("DRID mean distance 1");
        plotter.xlabel("Simulation time [ns]");
        plotter.ylabel("Distance");

        axis = fig.add_subplot(232);
        axis.plot(self.simulation_time, second_moment_mean);
        plotter.title("DRID mean moment 2");
        plotter.xlabel("Simulation time [ns]");
        plotter.ylabel("Distance");

        axis = fig.add_subplot(233);
        axis.plot(self.simulation_time, third_moment_mean);
        plotter.title("DRID mean moment 3");
        plotter.xlabel("Simulation time [ns]");
        plotter.ylabel("Distance");


        axis = fig.add_subplot(234);
        pdf, edges = np.histogram(first_moment_mean, 50, normed=True);
        binWidth = edges[1] - edges[0];
        axis.bar(edges[:-1], pdf*binWidth, binWidth);

        plotter.title("Histogram");
        plotter.xlabel("Moment 1 mean distibution");
        plotter.ylabel("Frequency");

        axis = fig.add_subplot(235);
        pdf, edges = np.histogram(first_moment_mean, 50, normed=True);
        binWidth = edges[1] - edges[0];
        axis.bar(edges[:-1], pdf*binWidth, binWidth);

        plotter.title("Histogram");
        plotter.xlabel("Moment 2 mean distibution");
        plotter.ylabel("Frequency");

        axis = fig.add_subplot(236);
        pdf, edges = np.histogram(first_moment_mean, 50, normed=True);
        binWidth = edges[1] - edges[0];
        axis.bar(edges[:-1], pdf*binWidth, binWidth);

        plotter.title("Histogram");
        plotter.xlabel("Moment 3 mean distibution");
        plotter.ylabel("Frequency");"""

        return DRID_distance;

    def compute_energy_profile(self, data, title, normalizationPoint, plotter):

        pdf, edges = np.histogram(data, 50, normed=True);

        normalizationPoint = 0;  # May remove

        nPoints = len(pdf);
        energy = np.zeros(nPoints);
        normalizationValue = 0;
        for i in range(0, nPoints):
            if pdf[i] > 0:
                energy[i] = -self.boltzmann_const * self.temperature * np.log(pdf[i]);
            else:
                energy[i] = -self.boltzmann_const * self.temperature * np.log(0.001);

            if edges[i] < normalizationPoint and edges[i + 1] > normalizationPoint:
                normalizationValue = energy[i];

        fig = plotter.figure(self.figure_counter);
        plotter.plot(edges[:-1], energy - normalizationValue);
        plotter.title("Energy " + title);
        plotter.xlabel(title);
        plotter.ylabel("Energy");

        np.savetxt(self.save_folder + "energy_" + title + "_" + self.file_name_end + ".txt",
                   energy - normalizationValue);
        np.savetxt(self.save_folder + "energy_edges_" + title + "_" + self.file_name_end + ".txt", edges);

        self.figure_counter += 1;
        return;

    def compute_Qdiff(self, traj, plotter, makePlots=False):
        # Compute Qdiff to see if state is closer to  4CAL or Apo
        Q_apo = self.best_hummer_q(traj, self.native_state_apo, plotter);
        Q_4CAL = self.best_hummer_q(traj, self.native_state_holo, plotter);

        Qdiff = Q_4CAL - Q_apo;

        np.savetxt(self.save_folder + 'Q_4cal_' + self.file_name_end + '.txt', Q_4CAL);
        np.savetxt(self.save_folder + 'Q_diff_' + self.file_name_end + '.txt', Qdiff);

        # Plot Q_4CAL
        self.producePlots(Q_4CAL, "Q_4cal", plotter);
        # self.compute_energy_profile(Q_4CAL, "Q_4cal", 0, plotter);

        return;

    def producePlots(self, data, orderParameterName, plotter):

        dataSort = np.linspace(np.min(data), np.max(data), 1000)[:, np.newaxis];

        fig = plotter.figure(self.figure_counter);
        """axis = fig.add_subplot(121);
        axis.plot(self.simulation_time, data);
        plotter.title(orderParameterName + " time series");
        plotter.xlabel("Simulation time [ns]");
        plotter.ylabel(orderParameterName);"""

        axis = fig.add_subplot(122);
        pdf, edges = np.histogram(data, 30, normed=True);

        sigma = (edges[1] - edges[0]) * 0.71;
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(data[:, np.newaxis]);
        log_dens = kde.score_samples(dataSort);

        # binWidth = edges[1] - edges[0];
        # axis.bar(edges[:-1], pdf*binWidth, binWidth);

        # plotter.title("Histogram");

        axis.fill(dataSort[::, 0], np.exp(log_dens), fc='#AAAAFF');
        plotter.xlabel(orderParameterName);
        plotter.ylabel("Frequency");

        self.figure_counter += 1;

        return;

    def best_hummer_q(self, traj, native, plotter):
        """Compute the fraction of native contacts according the definition from
        Best, Hummer and Eaton [1]
        Parameters
        ----------
        traj : md.Trajectory
        The trajectory to do the computation for
        native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used
        Returns
        -------
        q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`
        References
        ----------
        ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
        """
        BETA_CONST = 50  # 1/nm
        LAMBDA_CONST = 1.8
        NATIVE_CUTOFF = 0.45  # nanometers

        # get the indices of all of the heavy atoms
        heavy = native.topology.select_atom_indices('heavy')
        # get the pairs of heavy atoms which are farther than 3
        # residues apart
        heavy_pairs = np.array(
            [(i, j) for (i, j) in combinations(heavy, 2)
             if abs(native.topology.atom(i).residue.index - \
                    native.topology.atom(j).residue.index) > 3])

        # compute the distances between these pairs in the native state
        heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs, periodic=False)[0]
        # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
        native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]

        # now compute these distances for the whole trajectory
        r = md.compute_distances(traj, native_contacts, periodic=False)
        # and recompute them for just the native state
        r0 = md.compute_distances(native[0], native_contacts, periodic=False)

        q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)

        # np.savetxt(self.save_folder + 'fraction_native_contacts_' + self.file_name_end + '.txt',q)

        self.figure_counter += 1;
        return q;


def computeContactNorm(tmpMap1, tmpMap2):
    # print(global_cache.distanceMatrices)
    return np.linalg.norm((tmpMap1 - tmpMap2), 2)
    # return {'idx1':i,'idx2': j, 'res': np.linalg.norm((tmpMap1-tmpMap2),2)}

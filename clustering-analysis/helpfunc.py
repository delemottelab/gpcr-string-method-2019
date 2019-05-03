from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dill #needed for saving lambdas with pickle
import pickle
import mdtraj as md
import numpy as np
import logging
import itertools
import matplotlib.pyplot as plt
import os
import getpass
save_traj_enabled = False
save_csv_enabled = False
logger = logging.getLogger(__name__)
timestep_size = 0.18 / 1000  #in microseconds
kb = 0.0019872041 #kilocalories/mol
T = 310.15 #body temperature in kelvin

def get_project_path():
    return os.path.expanduser("~") + "/projects/gpcr"

class Simulation(object):
    def __init__(self, params):
        if 'timestep' in params:
            params['timestep'] = timestep_size
        self.__dict__.update(params)
        self._clusterpath = "simulations/dror-anton-2011/clustering/"
    def _getPath(self):
        tmp = "pnas2011b-" + self.condition + "-" + self.number + "-no-water-no-lipid"
        return "simulations/dror-anton-2011/DESRES-Trajectory_" + tmp + "/" + tmp + "/"
    path = property(_getPath)
    
    def _getClusterpath(self):
        return self._clusterpath
    def _setClusterpath(self,clusterpath):
        self._clusterpath = clusterpath
    def _getId(self):
        return self.condition + "-" + self.number + "-" + self.name
    id = property(_getId)
 
    def __str__(self):
        return str(self.__dict__)
    
    clusterpath = property(_getClusterpath, _setClusterpath)

def normalize(values):
    max_val = values.max()
    min_val = values.min()
    scale = max_val - min_val
    offset = min_val
    return (values - offset) / max(scale,1e-10)

def find_atom(element, residue, atom_name, traj, query=None):
    if query is None:
        atoms = traj.topology.atoms
    else:
        atoms = [traj.topology.atom(idx) for idx in  traj.topology.select(query)]
    res = None
    if not isinstance(residue, basestring):
        residue = str(residue)
    for atom in atoms:
        if atom.element.symbol == element and str(
                    atom.residue) == residue and atom.name == atom_name:
            # print(atom)
            if res is not None:
                logger.warn("Multiple atoms found %s, %s", atom, res)
            res = atom
    return res

def to_vmd_query(atoms, atom_name=None):
    """Set atom name if you all atoms are of the same kind"""
    q = ""
    if atom_name is not None:
        q = "name " + atom_name + " and resid "
        for a in atoms:
            q += str(a.residue.resSeq) + " "
    else:       
        for idx, a in enumerate(atoms):
            q += (" or " if idx > 0 else "") + "(resid " + \
                str(a.residue.resSeq) + " and name " + a.name + ")"
    return q

def to_vmd_selection(resSeq):
    """Returns command, selection variable name"""
    return "set resId{} [atomselect top \"resid {}\"];".format(resSeq, resSeq), "resId{}".format(resSeq)

def to_vmd_beta_value(resSeq, beta):
    command, variable = to_vmd_selection(resSeq)
    return command + "$" + variable + " set beta {};".format(beta)

def filter_atoms(atoms, ref_atoms):
    """
    Returns atoms which name matched the name i ref_atoms as well as the once which did not match.
    Matching is done on name, i.e. str(atom)
    """
    ref_atom_names = [str(a) for a in ref_atoms]
    missing_atoms = []
    matching_atoms = []
    # Atoms in inactive not in simu
    for atom in atoms:
        if str(atom) not in ref_atom_names:
            # print(atom)
            missing_atoms.append(atom)
        else:
            # print("FOUND IT", atom)
            matching_atoms.append(atom)
    return matching_atoms, missing_atoms


def find_duplicates(atoms):
    atom_names = [str(a) for a in atoms]
    return [a for a in atoms if atom_names.count(str(a)) > 1]

def get_atoms(query, top, sort=True):
    res = [top.atom(idx) for idx in top.select(query)]
    return np.array(sorted(res, key=str) if sort else res)

def select_atoms(query, traj):
    atoms = traj.top.select(query)
    return traj.atom_slice(atoms)

def select_atoms_incommon(query, top, ref_top, warn_missing_atoms=True):
    """
    Matches atoms returned by the query for both topologies by name and returns the atom indices for the respective topology
    """
    logger = logging.getLogger(__name__)
    atoms = get_atoms(query, top)
    ref_atoms = get_atoms(query, ref_top)
    ref_atoms, missing_atoms = filter_atoms(ref_atoms, atoms)
    if warn_missing_atoms and len(missing_atoms) > 0:
        logger.warn("%s atoms in reference not found topology. They will be ignored. %s", len(
            missing_atoms), missing_atoms)
    atoms, missing_atoms = filter_atoms(atoms, ref_atoms)
    if warn_missing_atoms and len(missing_atoms) > 0:
        logger.warn("%s atoms in topology not found reference. They will be ignored. %s", len(
            missing_atoms), missing_atoms)
    duplicate_atoms = find_duplicates(atoms)
    if warn_missing_atoms and len(duplicate_atoms) > 0:
        logger.warn("%s duplicates found in topology: %s",len(duplicate_atoms), duplicate_atoms)
    duplicate_atoms = find_duplicates(ref_atoms)
    if warn_missing_atoms and len(duplicate_atoms) > 0:
        logger.warn("%s duplicates found in reference: %s", len(duplicate_atoms), duplicate_atoms)
    if warn_missing_atoms and len(atoms) != len(ref_atoms):
        logger.warn("number of atoms in result differ %s vs %s",
                    len(atoms), len(ref_atoms))
    return [a.index for a in atoms], [a.index for a in ref_atoms]




def save_csv(file_name, results, simulation, header=None, separator=';'):
    logger = logging.getLogger(__name__)
    if not save_csv_enabled:
        logger.info("Save to CSV is disabled")
        return
    filename = simulation.path + 'results/' + file_name + '.csv'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    def to_csv_line(o):
        if o is None:
            return ""
        elif isinstance(o, tuple):
            res = ""
            for idx, x in enumerate(o):
                res += str(x) if idx == 0 else separator + str(x)
            return res
        else:
            return str(o)
    with open(filename, 'w') as f:
        if header is not None:
            f.write(to_csv_line(header) + "\n")
        for o in results:
            f.write(to_csv_line(o) + "\n")
    logger.info(filename + " written to disk")




def save_traj(simu_traj, simulation, filename):
    logger = logging.getLogger(__name__)
    if not save_traj_enabled:
        logger.info("Save traj disabled")
        return
    simu_traj[0].save(simulation.path + filename + ".pdb")
    simu_traj.save(simulation.path + filename + ".dcd")
    logger.info("Saved files '%s.*' to %s", filename, simulation.path)

def compute_distance_CA_atoms(res1, res2, traj, periodic=True):
    if isinstance(res1, int):
        q = "(resSeq %s or resSeq %s) and name CA" % (res1, res2)
        # print(q)
        atoms = tuple(traj.top.select(q))
    else:
        # Deprecated legace code---
        atom1 = find_atom('C', res1, 'CA', traj)
        atom2 = find_atom('C', res2, 'CA', traj)
        atoms = (atom1.index, atom2.index)
        # print(atoms)
    dists = md.compute_distances(
        traj,
        [atoms],
        periodic=periodic)
    return dists

def compute_COM_distance(query1, query2, simu_traj):
    com1 = md.compute_center_of_mass(simu_traj.atom_slice(simu_traj.top.select(query1)))
    com2 = md.compute_center_of_mass(simu_traj.atom_slice(simu_traj.top.select(query2)))
    #TODO fix so that we handle periodic BC correctly here
    return np.linalg.norm(com1 - com2, axis=1)

def find_state_changes(data):
    if len(data) == 0:
        return np.empty(0)
    result = np.empty((len(data),))
    result[0] = False
    for idx in range(1, len(data)):
        x0 = data[idx - 1]
        x1 = data[idx]
        if np.isnan(x0) and np.isnan(x1):
            result[idx] = False
        else:
            result[idx] = x0 != x1
    return result


def plot_state_changes(time, data, description="State changes"):
    result = find_state_changes(data)
    plt.figure(figsize=(16, 3))
    #plt.bar(time, result, align="center", edgecolor='w')
    plt.plot(time, result, marker="", linestyle="-")
    plt.xlabel("Time")
    plt.yticks([], [])
    plt.title(description)
    plt.show()

def cluster_scatterplot(simulation, xdata, ydata, xlabel="", ylabel="", title="", alpha=1.0, figsize=(6,6), markersize=25):
    plt.figure(figsize=figsize)
    length = len(simulation.cluster_indices)
    cluster_reps_x = np.empty(length)
    cluster_reps_y = np.empty(length)
    for i in range(0, len(simulation.cluster_rep_indices)):
        cluster = i + 1
        xvalues = np.empty(length)
        yvalues = np.empty(length)
        for idx in range(0, length):
            if simulation.cluster_indices[idx] == cluster:
                x = xdata[idx]
                y = ydata[idx]
            else:
                x = np.nan
                y = np.nan
            xvalues[idx] = x
            yvalues[idx] = y
            if idx == simulation.cluster_rep_indices[i]:
                cluster_reps_x[idx] = x
                cluster_reps_y[idx] = y
            elif idx not in simulation.cluster_rep_indices:
                cluster_reps_x[idx] = np.nan
                cluster_reps_y[idx] = np.nan
        plt.scatter(xvalues, yvalues, label="Cluster #" +
                    str(cluster), alpha=alpha, s=markersize)
    plt.scatter(cluster_reps_x, cluster_reps_y,
                label="Cluster Representations", marker="D", edgecolor='black',
                facecolor=(0, 0, 0, 0.0), linewidth=2, s=markersize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()
    
def cluster_plot(cluster_simu, cvs):
    #     for cv in cvs:
    #         plt.plot(cv.eval(cluster_simu.traj), label=cv.id)
    #         plt.legend()
    #         plt.show()
    for cvi, cvj in itertools.combinations(cvs, 2):
        cluster_scatterplot(
            cluster_simu,
            cvi.eval(cluster_simu.traj),
            cvj.eval(cluster_simu.traj),
            xlabel=cvi.id,
            ylabel=cvj.id,
            title="Cluster plot",
            alpha=1.0,
            figsize=(6, 6),
            markersize=25)
        
def compute_rmsds(simu_traj, ref_traj, q, warn_missing_atoms=True):
    simu_atoms, ref_atoms = select_atoms_incommon(
        q, simu_traj.topology, ref_traj.topology, warn_missing_atoms=warn_missing_atoms)
    rmsds = md.rmsd(simu_traj.atom_slice(simu_atoms),
                           ref_traj.atom_slice(ref_atoms))
    return rmsds   

def compute_active_inactive_rmsd(simu_traj, active_traj, inactive_traj, q):
    return compute_rmsds(simu_traj, active_traj, q), compute_rmsds(simu_traj, inactive_traj, q) 

def vmd_bond(atom1, atom2, varname="x"):
    return ("label add Bonds $%s/%s $%s/%s;") % (varname, atom1.index, varname, atom2.index)


def persist_object(o, filename):
    with open(filename + '.pkl', 'wb') as output:
        pickle.dump(o, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename + '.pkl', 'rb') as input:
        o = pickle.load(input)
    return o

def get_gromacs_indices(traj, resids, element='CA'):
    q = "name " + element + " and ("
    for i, r in enumerate(resids):
        q += "resSeq " + str(r) + (" or " if i < len(resids) -1 else ")")
    atoms = get_atoms(q, traj.topology, sort=False)
    return [(a,a.index+1) for a in atoms]

def fix_pbc(traj, atom_q="name CA and (resSeq 131 or resSeq 268)", epsilon=1e-3):
    """
    Make a test and see if the distances between the atoms is affected by the periodic boundary conditions.
    If so, center the trajectory so that the protein is not broken
    """
    atoms = traj.top.select(atom_q)
    d_pbc = md.compute_distances(
        traj,
        [atoms],
        periodic=True)
    d_nopbc = md.compute_distances(
        traj,
        [atoms],
        periodic=False)
    diff = np.absolute(d_pbc - d_nopbc)
    diff = diff[diff > epsilon]
    if len(diff) > 0:
        #We need to align the molecule 
        logger.debug("Aligning protein molecule to fix PBCs")
        return center_protein(traj)
    else:
        return traj

    
def center_protein(traj, inplace=True):
    """Center protein so that the molecule is not broken by periodic boundary conditions"""
    if next(traj.top.bonds, None) is None:
        logger.debug("No bonds in topology, creating standard bonds for alignment")
        traj.top.create_standard_bonds()
    return traj.image_molecules(inplace=inplace)

def fix_topology(topology):
    """
    Fix broken anton trajectories, typically used with excluding hydrogen atoms from trajectory
    see https://gist.github.com/evanfeinberg/6ee0421f8e3bd0b2e9a4
    and https://github.com/mdtraj/mdtraj/issues/822 
    """
    new_top = topology.copy()

    residues = {}
    for chain in new_top.chains:
        #print chain
        for residue in chain.residues:
            resname = str(residue)
            if resname in residues.keys():
                residues[resname].append(residue)
            else:
                residues[resname] = [residue]

    for resname in residues.keys():
        fragments = residues[resname]
        if len(fragments) > 1:
            main_fragment = fragments[0]
            new_atom_list = []
            new_atom_list += main_fragment._atoms
            for i in range(1,len(fragments)):
                fragment = fragments[i]
                for atom in fragment.atoms:
                    atom.residue = main_fragment
                new_atom_list += fragment._atoms
                fragment._atoms = []
                fragment.chain = main_fragment.chain
            main_fragment._atoms = new_atom_list

    return new_top
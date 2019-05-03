from __future__ import absolute_import, print_function, division

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

from colvars import *
from utils.helpfunc import *
import utils
import mdtraj as md

logger = logging.getLogger("createCVs")


def create_rmsd_cvs(directory, activename, inactivename, outpath):
    q_CA = "name CA"
    active_traj = md.load(directory + activename)
    inactive_traj = md.load(directory + inactivename)
    active_rmsd_cv = RmsdCV("active_rmsd", active_traj, q_CA, False)
    inactive_rmsd_cv = RmsdCV("inactive_rmsd", inactive_traj, q_CA, False)
    cvs = [active_rmsd_cv, inactive_rmsd_cv]
    cvs = normalize_cvs(cvs, trajs=[active_traj, inactive_traj])
    if outpath is not None:
        persist_object(cvs, outpath)
        logger.info("Saved CVs to file %s", outpath)
    return cvs, active_traj, inactive_traj


def create_5cvs(normalize=True, pbc=True):
    cvs = np.array([
        CADistanceCv("ALA76-CYS327", 76, 327, pbc),
        CADistanceCv("GLU268-LYS147", 268, 147, pbc),
        CADistanceCv("CYS327-LYS147", 327, 147, pbc),
        CADistanceCv("LEU115-LEU275", 115, 275, pbc),
        CADistanceCv("LYS267-PHE223", 267, 223, pbc)
    ])
    if normalize:
        active_traj = utils.load_pdb("3p0g")
        inactive_traj = utils.load_pdb("2rh1")
        # Note that the original CVs were normalized by a FreeMD trajectory and not just the reference structures.
        for cv in cvs:
            cv.normalize(trajs=[active_traj, inactive_traj])
    return cvs


def create_beta1_5cvs(normalize=True, pbc=True, beta2_cvs_dir="../gpcr/cvs/cvs-len5_good/"):
    # Mapping from beta2
    # ALA76->ALA84
    # CYS327->SER346 #Cysteine has the same structure as serine
    # GLU268 -> GLU285
    # LYS147-> ARG155 #switched from basic polar Lysin to basic polar Arginine
    # LEU115 -> LEU123
    # LEU275 -> LEU289
    # LYS267 -> ARG284 #switched from basic polar Lysin to basic polar Arginine
    # PHE223 -> TYR231 #Phenylalanine is a precursor for tyrosine
    cvs = np.array([
        CADistanceCv("ALA84-SER346", 84, 346, pbc),
        CADistanceCv("GLU285-ARG155", 285, 155, pbc),
        CADistanceCv("SER346-ARG155", 346, 155, pbc),
        CADistanceCv("LEU123-LEU289", 123, 289, pbc),
        CADistanceCv("ARG284-TYR231", 284, 231, pbc)
    ])
    if normalize:
        logger.info("Normalizing beta1 cvs with scales from beta2 CVs")
        # TODO maybe somehow include inactive equilibrated beta1 structure to make sure cvs are either 0 or 1 for that structure?
        # beta2_cvs = create_5cvs(normalize=True, pbc=pbc)
        beta2_cvs = load_object(beta2_cvs_dir + "cvs")
        for i, beta2_cv in enumerate(beta2_cvs):
            beta1_cv = cvs[i]
            beta1_cv._norm_offset = beta2_cv._norm_offset
            beta1_cv._norm_scale = beta2_cv._norm_scale
    return cvs


def create_beta1_npxxy_cvs():
    helix_63_dist_cv = CADistanceCv("TM6-TM3(CA)", 139, 289)
    inactive_traj = md.load("/data/oliver/pnas2011b-Dror-gpcr/equilibration/4gpo-beta1-charmm-gui/step1_pdbreader.pdb")
    q_npxxy = "name CA and protein and resSeq 341 to 346"
    inactive_rmsd_cv = RmsdCV("NPxxY-inactive-rmsd", inactive_traj, q_npxxy)
    return [helix_63_dist_cv, inactive_rmsd_cv]


def create_ionic_lock_cv(normalize=True):
    id_prefix = "ARG131-GLU268"
    return create_COM_distance_cv(id_prefix, res1=131, res2=268, res1_atom="CA", res2_atom="CA", normalize=normalize)


def create_YY_cv(normalize=True):
    id_prefix = "Y219-Y326"
    return create_COM_distance_cv(id_prefix, res1=219, res2=326, res1_atom="CZ", res2_atom="CZ", normalize=normalize)


def create_COM_distance_cv(id_prefix, res1, res2, res1_atom, res2_atom, normalize=True):
    active_traj = utils.load_pdb("3p0g")
    inactive_traj = utils.load_pdb("2rh1")
    if res1_atom == "CA" and res2_atom == "CA":
        # Optimization
        cv = CADistanceCv(id_prefix + "(CA-CA)", res1, res2)
    elif res1_atom == None and res2_atom == None:
        cv = COMDistanceCv(id_prefix + "(COM)", "protein and resSeq " + str(res1), "protein and resSeq " + str(res2))
    else:
        cv = COMDistanceCv(
            id_prefix + "({}-{})".format(res1_atom, res2_atom),
            "protein and resSeq {} and name {}".format(res1, res1_atom),
            "protein and resSeq {} and name {}".format(res2, res2_atom)
        )
    if normalize:
        cv.normalize(trajs=[active_traj, inactive_traj])
    return cv


def create_DRY_cvs(normalize=True):
    active_traj = utils.load_pdb("3p0g")
    inactive_traj = utils.load_pdb("2rh1")
    q = "chainid 0 and protein and (resSeq 130 to 132)"
    active_DRY_rmsd = RmsdCV("rmsd-active-DRY", active_traj, q)
    inactive_DRY_rmsd = RmsdCV("rmsd-inactive-DRY", inactive_traj, q)
    if normalize:
        active_DRY_rmsd.normalize(trajs=[active_traj, inactive_traj])
        inactive_DRY_rmsd.normalize(trajs=[active_traj, inactive_traj])
    return active_DRY_rmsd, inactive_DRY_rmsd


def create_loose_coupling_cvs(normalize=True):
    active_traj = utils.load_pdb("3p0g")
    inactive_traj = utils.load_pdb("2rh1")
    ligand_cv = LigandBindingCV(active_traj)
    connector_cv = ConnectorRegionCV(active_traj, inactive_traj)
    gprotein_cv = CADistanceCv("helix_63_dist(CA)", 131, 272)
    if normalize:
        ligand_cv.normalize(trajs=[active_traj, inactive_traj])
        connector_cv.normalize(trajs=[active_traj, inactive_traj])
        gprotein_cv.normalize(trajs=[active_traj, inactive_traj])
    return ligand_cv, connector_cv, gprotein_cv


def save_stringpath(cvs, startpoint, endpoint, outpath):
    stringpath = utils.get_straight_path(eval_cvs(startpoint, cvs), eval_cvs(endpoint, cvs), number_points=10,
                                         height_func=None)
    np.savetxt(outpath, stringpath)
    logger.info("Straight path with these CVs\n%s", stringpath)
    return stringpath


if __name__ == "__main__":
    # cvspath = "../gpcr/cvs/rmsd-cvs/"
    # rmsd_cvs, active_traj, inactive_traj = create_rmsd_cvs("../gpcr/reference_structures/", "active_CA.pdb",
    #                                                        "inactive_CA.pdb", cvspath + "new")
    # save_stringpath(rmsd_cvs, active_traj, inactive_traj, cvspath + "string-paths/new.txt")
    # logger.debug("Done")
    # print(eval_cvs(md.load("../gpcr/.string_simu/1/1/i1p1-minimization.gro"), rmsd_cvs, rescale=True))
    cvspath = "../gpcr/cvs/beta1-cvs/"
    beta1_cvs = create_beta1_5cvs(normalize=True, pbc=True)
    persist_object(beta1_cvs, cvspath + "cvs")


class LigandBindingCV(CV):
    """
    projection of the Ser207 5.46 CA atom position (in a given snapshot) onto the vector between the Ser207 5.46
    and Gly315 7.42 CA atoms (in the active structure) as a quantity that summarized the position of Ser207 5.46 ;
    zero is defined as the position of Ser207 5.46 in the active structure, and increasing values are defined to be displacement away from Gly315 7.42 .
    We called this quantity displacement of Ser207 5.46 away from helix 7.
    Trajectory windows with an average displacement greater than 0.6 Angstrom were classified as having an inactive ligand-binding site,
    whereas the remainder were classified as having an active ligand-binding site.
    """

    def __init__(self, active_traj, displacement_limit=0.06):
        # TODO bind ref_traj to variable instead
        CV.__init__(self, "LigandBinding", lambda traj: self.compute_ligand_binding(traj))
        self.displacement_limit = displacement_limit
        self.active_traj = active_traj
        self.active_ser = find_atom('C', 'SER207', 'CA', self.active_traj)
        self.active_gly = find_atom('C', 'GLY315', 'CA', self.active_traj)
        # compute vector from active state
        # TODO maybe use mdtraj compute_displacements instead
        unitvec = (
                      self.active_traj.xyz[0, self.active_gly.index] -
                      self.active_traj.xyz[0, self.active_ser.index]
                  ) * -1
        # normalize it
        self.unitvec = unitvec / np.linalg.norm(unitvec)
        # print(unitvec)
        # set the refrencene projection to zero
        self.reference_proj = np.dot(self.active_traj.xyz[0, self.active_ser.index], unitvec)
        # print(reference_proj)

    def find_ligand_site_atoms(self, traj):
        ligand_residues = [
            93, 109, 110, 113, 114, 117, 118, 191, 192, 193, 195, 199, 200, 203, 204,
            207, 286, 289, 290, 293, 305, 308, 309, 312, 316
        ]
        resid_query = "chainid 0 and protein and name CA and ("
        first = True
        for r in ligand_residues:
            resid_query += ("" if first else " or ") + "residue " + str(r)
            first = False
        resid_query += ")"
        return select_atoms_incommon(resid_query, traj.topology, self.active_traj.topology)

    def align_ligand_atoms(self, traj):
        simu_ligand_atoms, active_ligand_atoms = self.find_ligand_site_atoms(traj)
        ligand_aligned_traj = traj.superpose(
            self.active_traj[0],
            frame=0,
            atom_indices=simu_ligand_atoms,
            ref_atom_indices=active_ligand_atoms)
        return ligand_aligned_traj

    def compute_ligand_binding(self, traj):
        ligand_aligned_traj = self.align_ligand_atoms(traj)
        traj_ser = find_atom('C', 'SER207', 'CA', ligand_aligned_traj)
        #traj_gly = find_atom('C', 'GLY315', 'CA', ligand_aligned_traj)
        ligand_projection = np.empty(len(ligand_aligned_traj), dtype=float)
        for idx, ser_pos in enumerate(ligand_aligned_traj.xyz[:, traj_ser.index]):
            proj = np.dot(ser_pos, self.unitvec) - self.reference_proj
            # print(proj)
            ligand_projection[idx] = proj
        return ligand_projection


class ConnectorRegionCV(CV):
    """
    Trajectory windows were classified as having an inactive conformation of the connector region if the average rmsd
    of the 15 nonsym- metric, non-hydrogen atoms of Ile121 3.40 and Phe282 6.44 relative to the inactive structure was
    lower than the average rmsd of these atoms relative to the active structure;
    the remainder were classified as having an active conformation of the connector region.
    """

    def __init__(self, active_traj, inactive_traj):
        CV.__init__(self, "ConnectorRegion", lambda traj: self.compute_connector_region_rmsd_diff(traj))
        self.active_traj = active_traj
        self.inactive_traj = inactive_traj

    def compute_connector_region_rmsd_diff(self, traj):
        """negative when inactive"""
        active_rmsd, inactive_rmsd = self.compute_connector_region_rmsd(traj)
        return inactive_rmsd - active_rmsd

    def compute_connector_region_rmsd(self, traj):
        q_ILE121 = "element C and chainid 0 and protein and (residue 121 or residue 282) and (resname ILE or resname PHE)"
        return compute_active_inactive_rmsd(traj, self.active_traj, self.inactive_traj, q_ILE121)


class WaterSolubilityCV(CV):
    """
    Trajectory windows were classified as having an inactive conformation of the connector region if the average rmsd
    of the 15 nonsym- metric, non-hydrogen atoms of Ile121 3.40 and Phe282 6.44 relative to the inactive structure was
    lower than the average rmsd of these atoms relative to the active structure;
    the remainder were classified as having an active conformation of the connector region.
    """

    def __init__(self, resSeq, cutoff):
        CV.__init__(self, "water_solubility_res{}_{}nm".format(resSeq, cutoff),
                    lambda traj: self.compute_water_solubility(traj))
        self.resSeq = resSeq
        self.cutoff = cutoff

    def compute_water_solubility(self, traj):
        waters = traj.top.select("water and element O")
        atom = traj.top.select("chainid 0 and resSeq {} and name CA".format(self.resSeq))[0]
        atom_pairs = [(atom, w) for w in waters]
        dists = md.compute_distances(traj, atom_pairs)
        water_count = np.empty((len(traj),))
        for idx, frame_dists in enumerate(dists):
            count = frame_dists[frame_dists < self.cutoff].size
            water_count[idx] = count
        return water_count


def invert_cv(cv, power=1):
    if power < 1:
        raise Exception(
            "Exponent power must be >= 1. TODO support this (changing scaling factors will be more complicated in that case)")
    inv_cv = CV("1/" + cv.id, lambda traj: 1 / cv.rescale(cv.eval(traj)) ** power)
    # The normalization is a bit tricky
    cv_min = cv._norm_offset
    cv_max = cv_min + cv._norm_scale
    inv_min, inv_max = 1 / cv_max ** power, 1 / cv_min ** power
    inv_cv.normalize(scale=inv_max - inv_min, offset=inv_min)
    return inv_cv


def create_inverted_cvs(cvs, power=1):
    inverted_cvs = []
    for cv in cvs:
        inverted_cvs.append(invert_cv(cv, power=power))
    inverted_cvs = np.array(inverted_cvs)
    return inverted_cvs

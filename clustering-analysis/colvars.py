from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from helpfunc import *

class CV(object):
    """
    Collective variable class with an id and a generator to convert every frame in a trajectory to a numerical value
    """
    def __init__(self, id, generator, name=None):
        self._id = id
        self._generator = generator
        self._norm_offset = 0
        self._norm_scale = 1
        self.name = id if name is None else name
        
    @property    
    def id(self):
        return self._id
    
    @id.setter
    def id(self, id):
        self._id = id
        
    @property
    def generator(self):
        return self._generator
    
    @generator.setter
    def generator(self, generator):
        self._generator = generator
        
    def __str__(self):
        return "CV(id=%s)" % (self.id)
    
    def rescale(self, point):
        return self._norm_scale*point + self._norm_offset
    
    def normalize(self, trajs=None, scale=1, offset=0):
        """
        Scales the CV according to the following value (eval(traj)-offset)/scale
        
        Your CVs should be normalized to values between 0 and 1,
        meaning that 'scale' is the difference between the max and min value along the traj
        and 'offset' is the minimum value
        """
        if trajs is not None and len(trajs) > 0:
            max_val, min_val = None, None
            for t in trajs:
                evals = self._generator(t)
                tmax = evals.max()
                tmin = evals.min()
                if max_val is None or max_val < tmax:
                    max_val = tmax
                if min_val is None or min_val > tmin:
                    min_val = tmin
            scale = max_val - min_val
            offset = min_val
        self._norm_offset = offset
        self._norm_scale = scale
        return scale, offset

    def eval(self, traj):
        return (self._generator(traj) - self._norm_offset)/self._norm_scale

class RmsdCV(CV):
    """
    Class for RMSD based CVs to a reference structure
    """
    def __init__(self, id, ref_traj, atom_selection_q, warn_missing_atoms=True):
        CV.__init__(self, id, lambda traj: compute_rmsds(traj, ref_traj, atom_selection_q, warn_missing_atoms=warn_missing_atoms))
        self._atom_selection_q = atom_selection_q
    @property
    def atom_selection_q(self):
        return self._atom_selection_q
    
    def __str__(self):
        return "CV(id=%s)" % (self.id)
    
class ColorCv(CV):
    def __init__(self,graph, id, color1,color2, CV_generator):
        CV.__init__(self, id, lambda traj: CV_generator(traj, graph, color1, color2))
        self.colorid = "<|color " + str(color1) + "-" + str(color2) + " distance|>"        
    def __str__(self):
        return "CV(id=%s,color=%s)" % (self.id, self.colorid)
    

class CADistanceCv(CV):
    def __init__(self, id, res1, res2, periodic=True):
        CV.__init__(self, id, lambda traj: compute_distance_CA_atoms(res1, res2, traj, periodic))
        self.res1 = res1
        self.res2 = res2
        self.periodic = periodic

    def __str__(self):
        return "CADistCV(id=%s),%s-%s,periodic=%s" % (self.id, self.res1, self.res2, self.periodic)
    
class COMDistanceCv(CV):
    """Distance between center of mass (COM) between two groups of atoms"""

    def __init__(self, id, query1, query2):
        CV.__init__(self, id, lambda traj: compute_COM_distance(query1, query2,traj))
        self.query1 = query1
        self.query2 = query2

    def __str__(self):
        return "COMDistanceCv(id=%s),group-lengths:(%s,%s)" % (
            self.id, len(self.query1), len(self.query2))
    
class ClosestIonCV(CV):
    def __init__(self, resSeq, ion_resname="SOD"):
        CV.__init__(self, "closest_{}_res{}".format(ion_resname, resSeq), lambda traj: self.compute_closest_ion_dist(traj))
        self.resSeq = resSeq
        self.ion_resname = ion_resname

    def compute_closest_ion_dist(self, traj):
        ions = get_atoms("resname {}".format(self.ion_resname), traj.top, sort=False)
        ion_res = get_atoms("protein and resSeq {} and name CA".format(self.resSeq), traj.top, sort=False)[0].residue
        res_pairs = [(ion_res.index, ion.residue.index) for ion in ions]
        dists, contact_atoms = md.compute_contacts(traj, contacts=res_pairs, scheme="closest-heavy", ignore_nonprotein=False)
        closest_dists = np.empty((len(traj),))
        for idx, frame_dists in enumerate(dists):
            closest_dists[idx] = frame_dists.min()
        return closest_dists


class ClosestMoleculeCV(CV):
    """
    Compute the closest distance to a certain type of molecules (water, ions etc.) from a residue
    """

    def __init__(self, protein_resSeq, molecule_resname, scheme="closest-heavy", residue_query="protein and resSeq {} and name CA"):
        CV.__init__(
            self,
            "closest_{}_res{}".format(molecule_resname, protein_resSeq),
            lambda traj: self.compute_closest_molecule_dist(traj),
            name="Closest {} res{}".format(molecule_resname, protein_resSeq)
        )
        self.protein_resSeq = protein_resSeq
        self.molecule_resname = molecule_resname
        self.scheme = scheme
        self.residue_query = residue_query

    def compute_closest_molecule_dist(self, traj):
        mol_atoms = get_atoms("resname {}".format(self.molecule_resname), traj.top, sort=False)
        mol_res_idx = set([a.residue.index for a in mol_atoms])
        protein_res = get_atoms(
            self.residue_query.format(self.protein_resSeq),
            traj.top,
            sort=False
        )[0].residue
        res_pairs = [(protein_res.index, mm) for mm in mol_res_idx]
        dists, contact_atoms = md.compute_contacts(traj,
                                                   contacts=res_pairs,
                                                   scheme=self.scheme,
                                                   ignore_nonprotein=False)
        closest_dists = np.empty((len(traj),))
        for idx, frame_dists in enumerate(dists):
            min_dist = frame_dists.min()
            if min_dist <= 0:
                #raise BadFrameException("Closest molecule (dist={}) is below critical limit".format(min_dist), idx)
                closest_dists[idx] = np.nan
            else:
                closest_dists[idx] = min_dist
        return closest_dists


class CloseMoleculeCountCV(CV):
    """
    Computes the number of molecules (water, lipids etc.) within a certain cutoff from a residue
    """

    def __init__(self, protein_resSeq, cutoff, molecule_resname, molecule_atom_name, res_atom_name="CA",
                 compute_contacts=False,
                 scheme="closest-heavy"):
        CV.__init__(
            self,
            "count_{}_res{}_{}nm_{}-{}".format(molecule_resname, protein_resSeq, cutoff,
                                               "" if compute_contacts else res_atom_name,
                                               scheme if compute_contacts else molecule_atom_name),
            lambda traj: self.count_close_molecules(traj),
            name="#{} within {} of residue {}".format(molecule_resname, cutoff, protein_resSeq)
        )
        self.protein_resSeq = protein_resSeq
        self.molecule_resname = molecule_resname
        self.scheme = scheme
        self.cutoff = cutoff
        self.compute_contacts = compute_contacts
        self.res_atom_name = res_atom_name
        self.molecule_atom_name = molecule_atom_name
        self.generator = self.count_close_molecules

    def count_close_molecules(self, traj):
        mols = get_atoms("resname {} and name {}".format(self.molecule_resname, self.molecule_atom_name), traj.top,
                         sort=False)
        protein_atom = get_atoms(
            "protein and resSeq {} and name {}".format(self.protein_resSeq,
                                                       "CA" if self.res_atom_name is None else self.res_atom_name),
            traj.top,
            sort=False
        )[0]
        if self.compute_contacts:
            res_pairs = [(protein_atom.residue.index, mm.residue.index) for mm in mols]
            dists, contact_atoms = md.compute_contacts(traj, contacts=res_pairs, scheme=self.scheme,
                                                       ignore_nonprotein=False)
        else:
            atom_pairs = [(protein_atom.index, mm.index) for mm in mols]
            dists = md.compute_distances(traj, atom_pairs)
        molecule_count = np.empty((len(traj),))
        for idx, frame_dists in enumerate(dists):
            count = frame_dists[frame_dists < self.cutoff].size
            if count > 100:
                #raise BadFrameException("Too many close molecule with count={}, id={}".format(count, self.id), idx)
                molecule_count[idx] = np.nan
            else:
                molecule_count[idx] = count
        return molecule_count

    
class DihedralAtomIndexCv(CV):
    def __init__(self, id, atom_indices):
        CV.__init__(self, id, lambda traj: self.compute_dihedral(traj))
        if len(atom_indices) != 4:
            raise Exception("Expected 4 atoms. Got %s" % len(atom_indices))
        self.atom_indices = [atom_indices]

    def compute_dihedral(self, traj):
        # see http://mdtraj.org/1.9.0/examples/ramachandran-plot.html
        return md.compute_dihedrals(traj, self.atom_indices).squeeze()

    def __str__(self):
        return "DihedralAtomIndexCv(%s,atoms=%s)" % (self.id, self.atom_indices[0])


class ContactCv(CV):
    def __init__(self, id, res1, res2, scheme="closest-heavy", periodic=True, ignore_nonprotein=True):
        CV.__init__(self, id, lambda traj: self.compute_contact(traj))
        self.res1 = res1
        self.res2 = res2
        self.scheme = scheme
        self.periodic = periodic
        self.ignore_nonprotein = ignore_nonprotein

    def __str__(self):
        return "ContactCv(id=%s),%s-%s,%s,periodic=%s" % (self.id, self.res1, self.res2, self.scheme, self.periodic)
    
    def compute_contact(self, traj):
        res1_idx, res2_idx = -1, -1
        for residue in traj.topology.residues:
            if not self.ignore_nonprotein or residue.is_protein:
                if residue.resSeq == self.res1:
                    res1_idx = residue.index
                    if res2_idx > -1:
                        break
                elif residue.resSeq == self.res2:
                    res2_idx = residue.index
                    if res1_idx > -1:
                        break
        dists, atoms = md.compute_contacts(traj, contacts=[[res1_idx, res2_idx]], scheme=self.scheme,
                                           periodic=self.periodic, 
                                           ignore_nonprotein=self.ignore_nonprotein)
        return dists
    
def eval_cvs(traj, cvs):
    res = np.empty((len(traj),len(cvs)))
    for i, cv in enumerate(cvs):
        res[:,i] = np.squeeze(cv.eval(traj))
    return res


def normalize_cvs(cvs, simulations=None, trajs=None):
    if simulations is not None:
        trajs= [s.traj for s in simulations]
    if trajs is not None:
        for cv in cvs:
            cv.normalize(trajs)
    return cvs

def rescale_points(cvs, points):
    if len(points.shape) == 1:
        return np.array([cv.rescale(p) for cv,p in zip(cvs, points)])
    else:
        res = np.empty(points.shape)
        for i, point in enumerate(points):
            for j, cv in enumerate(cvs):
                res[i,j] = cv.rescale(point[j])
        return res

def rescale_evals(evals, cvs):
    if len(evals.shape) == 1:
        return cvs[0].rescale(evals)
    res = np.empty(evals.shape)
    for i, cv in enumerate(cvs):
        res[:, i] = cv.rescale(evals[:, i])
    return res
    
def crosscorrelate_cvs(simu, cvs1, cvs2, plot_limit=0, output=True, number_simus=1):
    def compute_correlation(eval1, eval2):
        return np.corrcoef(eval1, eval2)[0, 1]

    def correlation_sort(cvs, correlations):
        return sorted(
            zip(cvs, correlations),
            cmp=lambda (cv1, corr1), (cv2, corr2): int(1000 * (corr1 - corr2))
        )

    all_corr = []
    max_correlations_cvs2 = np.zeros(
        (len(cvs2),), )
    evals1 = eval_cvs(simu.traj, cvs1)
    evals2 = eval_cvs(simu.traj, cvs2)
    index = 1
    for i, cvi in enumerate(cvs1):
        val1 = evals1[:, i]
        max_corr = None
        for j, cvj in enumerate(cvs2):
            val2 = evals2[:, j]
            corr = compute_correlation(val1, val2)
            label = cvi.id + "-" + cvj.id
            index += 1
            all_corr.append((cvi, cvj, corr))
            if abs(corr) > max_correlations_cvs2[j]:
                max_correlations_cvs2[j] = abs(corr)
    if output:
        table = "cv1\tcv2\t\t\tcorrelation\n"
        # Find those with correlation close to zero, these are uncorrelated!
        for cvi, cvj, corr in sorted(all_corr,
                                     cmp=lambda (cv11, cv12, e1), (cv21, cv22, e2): int(10000 * (abs(e1) - abs(e2)))):
            table += "%s\t%s\t%s\n" % (cvi.id, cvj.id, corr)
        # logger.info("Correlation table for all correlations\n%s", table)
        # take the second set of CVs with the lowest correlation to all the first set of CVs
        table = "cv\t\t\tmax corr\tDesc.\n"
        # iterate through max_corr and change the correlation if they are internally correlated
        for i in range(len(max_correlations_cvs2)):
            val1 = evals2[:, i]
            for j in range(i + 1, len(max_correlations_cvs2)):
                corri, corrj = max_correlations_cvs2[[i, j]]
                val2 = evals2[:, j]
                cross_corr = abs(compute_correlation(val1, val2))
                if corri < corrj:
                    if cross_corr > corrj:
                        # this means that these two are internally more correlated
                        # since corri is more unique, we update corrj to a higher correlation value
                        max_correlations_cvs2[j] = cross_corr
                elif cross_corr > corri:
                    max_correlations_cvs2[i] = cross_corr
        max_corr = correlation_sort(cvs2, max_correlations_cvs2)
        ticks = np.arange(0, len(evals2), len(evals2) / number_simus)
        for index, (cv, corr) in enumerate(max_corr):
            table += "%s\t%s\t%s\n" % (cv.id, corr, cv)
            if index < plot_limit:
                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(plot_limit, 1, index + 1)
                ax.set_xticks(ticks)
                plt.plot(evals1, '--', alpha=0.3)
                plt.title("Max correlation=%s" % corr)
                cv_vals = eval_cvs(simu.traj, [cv])
                plt.plot(cv_vals, color="black", alpha=0.5)
                plt.legend([c.id for c in cvs1] + [cv.id])
                # plt.ylabel("Normalized value")
                plt.grid()
        logger.info("Correlation table for max correlation per cv\n%s", table)
        if plot_limit > 0:
            plt.show()
    return max_corr, np.array(all_corr)

class DistanceCv(CV):
    def __init__(self, id, res1, res2, res1_atom, res2_atom,
                 periodic=True,
                 select_query="protein and chainid 0 and ((resSeq %s and name %s) or (resSeq %s and name %s))"):
        CV.__init__(self, id, lambda traj: self.compute_distances(traj))
        self.res1 = res1
        self.res2 = res2
        self.res1_atom = res1_atom
        self.res2_atom = res2_atom
        self.periodic = periodic
        self.select_query = select_query

    def __str__(self):
        return "DistCV(id=%s),%s%s-%s%s,periodic=%s)" % (
            self.id, self.res1, self.res1_atom, self.res2, self.res2_atom, self.periodic)

    def compute_distances(self, traj):
        atoms = self.get_atoms(traj)
        dists = md.compute_distances(traj,
                                     atom_pairs=[atoms],
                                     opt=True,
                                     periodic=self.periodic)

        for idx, d in enumerate(dists):
            if d <= 0 and (self.res1 != self.res2 or self.res1_atom != self.res2_atom):
                raise BadFrameException("{}. Closest molecule (dist={}) is below critical limit".format(self.id, d),
                                        idx)
        return dists

    def get_atoms(self, traj):
        q = self.select_query % (self.res1, self.res1_atom, self.res2, self.res2_atom)
        # print(q)
        return traj.top.select(q)


    
class BadFrameException(Exception):

    def __init__(self, message, frame_idx):
        Exception.__init__(self, message)
        self.frame_idx = frame_idx

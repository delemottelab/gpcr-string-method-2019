from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from colvars.cv_utils import *

logger = logging.getLogger(__name__)


class CV(object):
    """
    Collective variable class with an id and a generator to convert every frame in a trajectory to a numerical value
    """

    def __init__(self, id, generator):
        self._id = id
        self._generator = generator
        self._norm_offset = 0
        self._norm_scale = 1

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

    def rescale(self, point):
        """Scales back to the original physical values"""
        return self._norm_scale * point + self._norm_offset

    def scale(self, point):
        """Scales to the normalized values"""
        return (point - self._norm_offset) / self._norm_scale

    def eval(self, traj):
        return (self._generator(traj) - self._norm_offset) / self._norm_scale


class DependentCV(CV):
    """
    A CV which depends on other CVs, i.e. they evaluate the output of previous CVs and do not process trajectories
    """

    def __init__(self, id, generator, original_cvs):
        CV.__init__(self, id, generator)
        self.original_cvs = original_cvs

    def __str__(self):
        return "CV(id=%s,depends_on_cv_ids=%s)" % (self.id, [cv.id for cv in self.original_cvs])

    def eval(self, data):
        if type(data) is not np.ndarray:
            traj = data
            data = eval_cvs(traj, self.original_cvs)
        return (self._generator(data) - self._norm_offset) / self._norm_scale


class RmsdCV(CV):
    """
    Class for RMSD based CVs to a reference structure
    """

    def __init__(self, id, ref_traj, atom_selection_q, warn_missing_atoms=False):
        # TODO bind ref_traj to variable instead
        CV.__init__(self, id, lambda traj: compute_rmsds(traj, ref_traj, atom_selection_q, self.warn_missing_atoms))
        self.warn_missing_atoms = warn_missing_atoms
        self._atom_selection_q = atom_selection_q

    @property
    def atom_selection_q(self):
        return self._atom_selection_q

    def __str__(self):
        return "RMSDCV(id=%s)" % self.id


class CADistanceCv(CV):
    def __init__(self, id, res1, res2, periodic=True):
        CV.__init__(self, id, lambda traj: compute_distance_CA_atoms(res1, res2, traj, periodic))
        self.res1 = res1
        self.res2 = res2
        self.periodic = periodic

    def __str__(self):
        return "CADistCV(id=%s),%s-%s,periodic=%s)" % (
            self.id, self.res1, self.res2, self.periodic if hasattr(self, "periodic") else "NA")


class COMDistanceCv(CV):
    """
    Distance between center of mass (COM) between two groups of atoms
    """

    def __init__(self, id, query1, query2):
        CV.__init__(self, id, lambda traj: compute_COM_distance(query1, query2, traj))
        self.query1 = query1
        self.query2 = query2

    def __str__(self):
        return "COMDistanceCv(id=%s),group-lengths:(%s,%s)" % (
            self.id, len(self.query1), len(self.query2))


class ContactCv(CV):
    def __init__(self, id, res1, res2, scheme="closest-heavy", periodic=True):
        CV.__init__(self, id, lambda traj: self.compute_contact(traj))
        self.res1 = res1
        self.res2 = res2
        self.scheme = scheme
        self.periodic = periodic

    def __str__(self):
        return "%s(id=%s),%s-%s,%s,periodic=%s" % (
            self.__class__.__name__, self.id, self.res1, self.res2, self.scheme, self.periodic)

    def compute_contact(self, traj):
        res1_idx, res2_idx = None, None
        for residue in traj.topology.residues:
            if residue.is_protein:
                if residue.resSeq == self.res1:
                    res1_idx = residue.index
                    if res2_idx is not None and res2_idx > -1:
                        break
                elif residue.resSeq == self.res2:
                    res2_idx = residue.index
                    if res1_idx is not None and res1_idx > -1:
                        break
        if res1_idx is None:
            raise ValueError("No residue with id {}".format(self.res1))
        if res2_idx is None:
            raise ValueError("No residue with id {}".format(self.res2))
        dists, atoms = md.compute_contacts(traj, contacts=[[res1_idx, res2_idx]], scheme=self.scheme,
                                           periodic=self.periodic)
        return dists


class InverseContactCv(ContactCv):
    """
    Same as ContactCv but with the inverse of the distance
    """

    def __init__(self, id, res1, res2, scheme="closest-heavy", periodic=True):
        ContactCv.__init__(self, id, res1, res2, scheme=scheme, periodic=periodic)

    def compute_contact(self, traj):
        dists = ContactCv.compute_contact(self, traj)
        return 1 / dists


class MaxDistanceCv(CV):
    """
    Max distance between any two atoms on the two residues
    """

    def __init__(self, id, res1, res2, periodic=True, atom_query="protein and resSeq {}"):
        CV.__init__(self, id,
                    lambda traj: compute_furthest_distance(res1, res2, traj, periodic=periodic, atom_query=atom_query))
        self.res1 = res1
        self.res2 = res2
        self.periodic = periodic
        self.atom_query = atom_query

    def __str__(self):
        return "MaxDistCV(id=%s),%s-%s,periodic=%s,q=%s" % (
            self.id, self.res1, self.res2, self.periodic, self.atom_query)


class StringIndexCv(DependentCV):
    """Chooses the closest point in some CV space on a given string. The output is the string value"""

    def __init__(self, id, stringpath, simulation_cvs, interpolate=False):
        DependentCV.__init__(self,
                             id,
                             lambda cv_evals: self.compute_string_index(cv_evals),
                             simulation_cvs)
        self.stringpath = stringpath
        self.interpolate = interpolate

    def compute_string_index(self, cv_evals):
        # For every frame, find closest point on string

        indices = np.empty((len(cv_evals),))
        for i, frame in enumerate(cv_evals):
            dists = np.linalg.norm(self.stringpath - frame, axis=1)
            closest = np.argmin(dists)
            if self.interpolate:
                if closest == 0:
                    second_closest = 1
                elif closest == len(self.stringpath) - 1:
                    second_closest = closest - 1
                else:
                    if dists[closest + 1] < dists[closest - 1]:
                        second_closest = closest + 1
                    else:
                        second_closest = closest - 1
                total_dist = dists[closest] + dists[second_closest]
                indices[i] = closest * (1 - dists[closest] / total_dist) + second_closest * (
                            1 - dists[second_closest] / total_dist)
            else:
                indices[i] = closest
        return indices


class ColorCv(CV):
    def __init__(self, graph, id, color1, color2, CV_generator):
        CV.__init__(self, id, lambda traj: CV_generator(traj, graph, color1, color2))
        self.colorid = "<|color " + str(color1) + "-" + str(color2) + " distance|>"

    def __str__(self):
        return "CV(id=%s,color=%s)" % (self.id, self.colorid if hasattr(self, 'colorid') else self.id)

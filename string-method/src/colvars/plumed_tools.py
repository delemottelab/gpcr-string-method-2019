from __future__ import absolute_import, division, print_function

import collections
import inspect

from colvars.generic_cvs import *

logger = logging.getLogger(__name__)


def _to_plumed_atom_indices(atom_indices):
    """
    Converts between 0-indexed mdtraj indices to 1-indexed plumed indices
    """
    return np.array(atom_indices) + 1


def _CADistanceCV_to_plumedCv(cv, idx, topology):
    atom_indices = get_CA_atom_indices(cv.res1, cv.res2, topology)
    atom_indices = _to_plumed_atom_indices(atom_indices)
    if len(atom_indices) < 2:
        raise Exception("Only {} atoms found for CV {}".format(len(atom_indices), cv))
    if not cv.periodic:
        logger.warn(
            "You are having non periodic boundary conditions for a distance. " \
            + " You may need to align your molecule somehow in plumed." \
            + " For example WHOLEMOLECULES ENTITY0=atom1_index,atom2_index,...")
    return "DISTANCE ATOMS={},{} {}LABEL=cv{}".format(
        atom_indices[0],
        atom_indices[1],
        "" if cv.periodic else "NOPBC ",
        idx
    )


def convert_to_plumed_restraints(cvs, topology, kappa=10000.0):
    """

    :param cvs:
    :param topology:
    :param kappa: array or integer. If array, there is one kappa per CV
    :return: string of the content of the restraints file
    """
    ncvs = len(cvs)
    if isinstance(kappa, collections.Iterable):
        if len(kappa) != len(cvs):
            raise Exception("Number of cvs and number of kappas must be the same")
        kappas = kappa
    else:
        kappas = [kappa for i in range(ncvs)]

    definitions = ""
    for idx, cv in enumerate(cvs):
        if inspect.isclass(CADistanceCv):
            plumed_cv = _CADistanceCV_to_plumedCv(cv, idx, topology)
            definitions += "#{}\n{}\n".format(cv.id, plumed_cv)
        else:
            raise Exception("Object {} cannot be converted to plumed restraint. Please implement a conversion")
    label = "restraint"
    restraints = "RESTRAINT ARG=" \
                 + ",".join("cv{}".format(i) for i in range(ncvs)) \
                 + " AT=" \
                 + ",".join("$cv{}_center".format(i) for i in range(ncvs)) \
                 + " KAPPA=" \
                 + ",".join("{}".format(k) for k in kappas) \
                 + " LABEL={}".format(label)
    outputs = "PRINT ARG={}.bias,{}.force2 FILE=plumed-restrained.out STRIDE=500".format(label, label)
    plumed_file_content = "{}\n\n{}\n{}".format(
        definitions,
        restraints,
        outputs
    )
    # logger.info("plumed_file_content: %s", plumed_file_content)
    return plumed_file_content


if __name__ == "__main__":
    import system_setup.create_cvs as create_cvs

    logger.info("Started")
    #cvs = create_cvs.create_5cvs(normalize=False, pbc=True)
    traj = md.load("../../gpcr/reference_structures/3p0g-noligand/equilibrated.gro")
    plumed_file_content = convert_to_plumed_restraints(cvs, traj.topology)
    logger.info("Done. plumed_file_content\n\n%s", plumed_file_content)

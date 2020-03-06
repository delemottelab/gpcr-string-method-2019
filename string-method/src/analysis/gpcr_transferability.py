from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('default')
import colvars
from colvars.generic_cvs import ContactCv
import mdtraj as md
import numpy as np
import matplotlib.colors

logger = logging.getLogger("GpcrTrans")


class ReceptorCvComplex(object):
    def __init__(self, name, cvs, active_structure, inactive_structure, active_pdb_name, inactive_pdb_name, color=None):
        self.name = name
        self.cvs = cvs
        self.active_structure = active_structure
        self.inactive_structure = inactive_structure
        self.active_pdb_name = active_pdb_name
        self.inactive_pdb_name = inactive_pdb_name
        self.color = color

    def eval_active(self, cv_indices=None, rescale=True):
        try:
            if self.active_structure is not None:
                evals = colvars.eval_cvs(self.active_structure, self.cvs[cv_indices], rescale)
                return evals
        except ValueError as ex:
            logger.exception(ex)
            logger.error("Could not eval {}/active for cvs {}".format(self.name, cv_indices))
            return None

    def eval_inactive(self, cv_indices=None, rescale=True):
        try:
            if self.inactive_structure is not None:
                evals = colvars.eval_cvs(self.inactive_structure, self.cvs[cv_indices], rescale)
                return evals
        except ValueError as ex:
            logger.exception(ex)
            logger.error("Could not eval {}/inactive for cvs {}".format(self.name, cv_indices))
            return None

    def scale(self):
        if self.active_structure is None and self.inactive_structure is None:
            return
        trajs = []
        if self.active_structure is not None:
            trajs.append(self.active_structure)
        if self.inactive_structure is not None:
            trajs.append(self.inactive_structure)
        for cv in self.cvs:
            try:
                cv.normalize(trajs=trajs)
            except Exception as ex:
                logger.exception(ex)
                logger.error("Failed to scale cv {} for receptor {}", cv.name, self.name)


def create_transfer_gpcr_cvs(type, separator_char=","):
    residue_alignment_file = "../../gpcr/reference_structures/gpcr_pdbs/{}.csv".format(type)
    logger.info("Loaded alignemnt info from %s", residue_alignment_file)
    column_to_receptor = []
    receptor_to_resids = {}
    receptor_to_cvs = {}
    ncvs = 0
    with open(residue_alignment_file) as f:
        i = -1
        for row in f.readlines():
            i += 1
            if i == 0:
                for column_idx, r in enumerate(row.strip().split(separator_char)):
                    column_to_receptor.append(r)
                    receptor_to_resids[r] = []
                    receptor_to_cvs[r] = []
            elif row.startswith(separator_char):  # blank line
                continue
            elif row.startswith("CV"):
                ncvs += 1
            else:  # parse line and find atoms indices
                for column_idx, val in enumerate(row.strip().split(separator_char)):
                    r = column_to_receptor[column_idx]
                    receptor_to_resids[r].append(val if r == "generic" else int(val[1:]))
    logger.info("Mapping the following receptors to residues:\n%s", receptor_to_resids)
    for i in range(ncvs):
        for c, r in enumerate(column_to_receptor):
            cvs = receptor_to_cvs.get(r)
            # Assuming 2 resides per CV
            resid1 = receptor_to_resids.get(r)[i * 2]
            resid2 = receptor_to_resids.get(r)[i * 2 + 1]
            if r == "generic":
                cvs.append((resid1, resid2))
            else:
                cvs.append(ContactCv(r + "({}-{})".format(resid1, resid2), resid1, resid2, scheme="ca", periodic=True))
    return receptor_to_cvs


def load_receptor_structures_for_cvs(receptor_to_cvs, separator_char=","):
    dir = "../../gpcr/reference_structures/gpcr_pdbs/"
    file = dir + "receptor_to_structures.csv"
    receptor_cvs = []
    with open(file) as f:
        for row in f.readlines():
            [receptor, active_pdb, inactive_pdb, color] = row.strip().split(separator_char)
            # print(receptor, active_pdb, inactive_pdb)
            if receptor == "Receptor":  # header
                continue
            active_struct = None if active_pdb in ["", "\n"] else md.load(dir + active_pdb + ".pdb")
            inactive_struct = None if inactive_pdb in ["", "\n"] else md.load(dir + inactive_pdb + ".pdb")
            rcv = ReceptorCvComplex(
                receptor,
                np.array(receptor_to_cvs.get(receptor)),
                active_struct,
                inactive_struct,
                active_pdb,
                inactive_pdb,
                color=color
            )
            receptor_cvs.append(rcv)
    for rcv in receptor_cvs:
        rcv.scale()
    logger.info("Loaded structures and CVs %s", [(rcv.active_pdb_name, rcv.inactive_pdb_name) for rcv in receptor_cvs])
    return receptor_cvs


def visualize(receptor_cvs, rescale=True, title="", fill_alpha=1.):
    # cmap = plt.cm.get_cmap("tab20c", len(receptor_cvs) + 1)
    ncvs = len(receptor_cvs[0].cvs)
    ncols = 2
    nrows = np.ceil(ncvs / (2 * ncols))
    index = 1
    # plt.suptitle(title + " (scaled={})".format(not rescale))
    active_marker, inactive_marker = "*", "^"
    for cvi in range(0, ncvs):
        if cvi % 2 != 0 and cvi != ncvs - 1:
            continue  # Don't plot a CV twice
        cvj = cvi + 1 if cvi < ncvs - 2 else cvi - 1
        cv_indices = [cvi, cvj]
        plt.subplot(nrows, ncols, index, adjustable='box', aspect="auto")
        for r_idx, rcv in enumerate(receptor_cvs):
            cv0, cv1 = rcv.cvs[cv_indices]
            if rcv.name == "generic":
                plt.xlabel("|{}-{}| [nm]".format(cv0[0], cv0[1]))  # , fontsize=label_fontsize)
                plt.ylabel("|{}-{}| [nm]".format(cv1[0], cv1[1]))  # , fontsize=label_fontsize)
                continue
            active_evals = rcv.eval_active(cv_indices=cv_indices, rescale=rescale)
            inactive_evals = rcv.eval_inactive(cv_indices=cv_indices, rescale=rescale)
            # color = cmap(r_idx)
            if active_evals is not None:
                plt.plot(active_evals[:, 0], active_evals[:, 1],
                         markeredgecolor=rcv.color,
                         marker=active_marker,
                         # markersize=marker_size,
                         linestyle="None",
                         markerfacecolor=matplotlib.colors.colorConverter.to_rgba(rcv.color, alpha=fill_alpha),
                         # "None",
                         label=rcv.name if inactive_evals is None else None)
            if inactive_evals is not None:
                plt.plot(inactive_evals[:, 0], inactive_evals[:, 1],
                         markeredgecolor=rcv.color,
                         marker=inactive_marker,
                         # markersize=marker_size,
                         linestyle="None",
                         markerfacecolor=matplotlib.colors.colorConverter.to_rgba(rcv.color, alpha=fill_alpha),
                         # "None",
                         label=rcv.name)
        index += 1
        # plt.tick_params(labelsize=ticks_labelsize)
    # plt.legend(fontsize=label_fontsize, bbox_to_anchor=(1.25, 0.75), loc="upper left", ncol=2)
    plt.legend(bbox_to_anchor=(1.25, 0.75), loc="upper left", ncol=2)
    # plt.show()
    plt.savefig("gpcr_transferability.svg")


if __name__ == "__main__":
    type = "sequence_alignment_feb2019"
    # type="sequence_alignment_gpcrdb"
    receptor_to_cvs = create_transfer_gpcr_cvs(type)
    receptor_cvs = load_receptor_structures_for_cvs(receptor_to_cvs)
    visualize(receptor_cvs, title=type, rescale=True)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import scipy
import scipy.misc

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import shutil
import time
# import dill #needed for saving lambdas with pickle
import pickle
import re
import mdtraj as md
import glob
from utils.reparametrize_string import *

sys.path.append('notebooks/')
sys.path.append('notebooks/MD_common/')

logger = logging.getLogger("utils")
reference_strucs_topdir = "../gpcr/"

ticks_labelsize = 35  # "large"
label_fontsize = 40
legend_fontsize = 35
pdb_fontsize = 20
marker_size = 300
linewidth = 5
simuid_to_color = {
    "holo-optimized": "green",
    "apo-optimized": "orange",
    "apo-curved": "red",
    "holo-curved": "cyan"
}
simuid_to_label = {
    "holo-optimized": "holo",
    "apo-optimized": "apo",
    "apo-curved": "apo'",
    "holo-curved": "holo'"
}
project_dir = "../"


def sorted_alphanumeric(l):
    """
    From https://arcpy.wordpress.com/2012/05/11/sorting-alphanumeric-strings-in-python/
    Sorts the given iterable in the way that is expected.
    Required arguments:
    l -- The iterable to be sorted.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def unit_vector(vector):
    """
    From https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    From https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    Returns the angle in radians between vectors 'v1' and 'v2'::

            #>>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            #>>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            #>>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def load_binary(path):
    """
    Load binary objects
    """
    return pickle.load(open(path, "r"))


def read_file(path):
    with open(path, "r") as myfile:
        data = myfile.read()
    return data


def get_backup_name(path):
    return path.strip("/") + "#" + time.strftime('%Y%m%d-%H%M')


def backup_path(oldpath):
    """Backups/Moves the path to a backup name"""
    if os.path.exists(oldpath):
        newpath = get_backup_name(oldpath)
        if os.path.exists(newpath):
            backup_path(newpath)
        shutil.move(oldpath, newpath)


def makedirs(path, overwrite=False, backup=True):
    if os.path.exists(path):
        if overwrite:
            if backup:
                backup_path(path)
            else:
                shutil.rmtree(path)  # beware, removes all the subdirectories!
            os.makedirs(path)
    else:
        os.makedirs(path)


def copy_and_inject(infile, outfile, params, marker="$%s", start_index=0):
    text = read_file(infile)
    with open(outfile, "w") as out:
        out.write(inject(text, params, marker=marker, start_index=start_index))


def inject(text, params, marker="$%s", start_index=0):
    for i in range(start_index, len(params) + start_index):
        text = text.replace(marker % i, str(params[i - start_index]))
    return text


def load_traj_for_regex(directory,
                        traj_filename,
                        top_filename,
                        stride=1,
                        print_files=False):
    file_list = sorted_alphanumeric(glob.glob(directory + traj_filename))
    logger.info("Loading %s files from directory %s", len(file_list), directory)
    if print_files:
        logger.debug("Trajectories included:\n%s", "\n".join([t for t in file_list]))
    traj = md.load(
        file_list,
        top=glob.glob(directory + top_filename)[0],
        stride=stride)
    return traj


def plot_path(path, label=None, text=None, ncols=None, legend=True, twoD=True, lambda1D=None, lambda2D=None,
              endpoints_only=False, axis_labels=None, scatter=True, boxplot_data=None, ticks=None, xticks=None,
              color=None):
    def do_plot2D(i, j, x, y):
        plt.grid()
        if endpoints_only:
            array_mask = np.array([0, -1])
            plt.plot(x[array_mask], y[array_mask], '--', color='black', alpha=0.3, label=label, linewidth=linewidth)
            plt.scatter(x[-1], y[-1], marker='^', color='black', alpha=0.5, s=marker_size)
        else:
            plt.plot(x, y, '--', alpha=0.5, label=label, linewidth=linewidth, color=color)
            plt.scatter(x, y, alpha=0.5, s=marker_size, color=color)
            plt.scatter(x[-1], y[-1], marker='^', color='black', alpha=0.5, s=marker_size)
        plt.grid()
        if axis_labels is not None and len(axis_labels) > j:
            plt.xlabel(axis_labels[i], fontsize=label_fontsize)
            plt.ylabel(axis_labels[j], fontsize=label_fontsize)
        if text is not None:
            plt.text(x[0], y[0], text, color="black", fontsize=label_fontsize)
        else:
            plt.scatter(x[0], y[0], marker='*', color='black', alpha=0.5, s=marker_size)
        # if legend:
        #     plt.legend(fontsize=label_fontsize)
        if lambda2D is not None:
            lambda2D(i, j, x, y)
        plt.tick_params(labelsize=ticks_labelsize)
        # plt.tight_layout()

    def do_plot1D(i, x):
        xticks1D = np.linspace(0, 1, min(len(x), 5)) if xticks is None else xticks
        ticks1D = [tick * 1 / len(x) for tick in range(len(x))] if ticks is None else ticks
        if boxplot_data is not None:
            bdata = boxplot_data[:, :, i] if len(boxplot_data.shape) > 2 else boxplot_data
            plt.boxplot(bdata, showmeans=False, positions=ticks1D,  widths=1 / (ncols*len(x)),
                        manage_xticks=False)
        plt.plot(ticks1D, x, '--', alpha=0.5, label=label, linewidth=linewidth, color=color)
        if scatter:
            plt.scatter(ticks1D, x, alpha=0.5, s=marker_size, color=color)
        if text is not None:
            plt.text(0, x[0], text, color="black", fontsize=label_fontsize)
        else:
            plt.scatter(0, x[0], marker='*', color='black', alpha=0.5)
        # if legend:
        #     plt.legend(fontsize=label_fontsize)
        if lambda1D is not None:
            lambda1D(i, x)
        if axis_labels is not None:
            if len(axis_labels) == dim:
                plt.ylabel(axis_labels[i], fontsize=label_fontsize)
            elif len(axis_labels) == dim + 1:
                plt.xlabel(axis_labels[0], fontsize=label_fontsize)
                plt.ylabel(axis_labels[i + 1], fontsize=label_fontsize)
            else:
                logger.error("invalid number of axis labels, %s, for %s dimensions. Must be the same or one larger.",
                             len(axis_labels), dim)
        # plt.grid()
        # plt.locator_params(axis='x', nbins=min(len(x), 20))
        plt.xticks(xticks1D)
        plt.xlim((xticks1D[0], xticks1D[-1]))
        plt.tick_params(labelsize=ticks_labelsize)
        # plt.tight_layout()

    if len(path.shape) == 1:
        if twoD:
            dim = path.shape[0]
            path = np.expand_dims(path, axis=0)
        else:
            dim = 1
    else:
        dim = path.shape[1]
    if twoD:
        combinations = int(scipy.misc.comb(dim, 2))
    else:
        combinations = dim
    idx = 1
    if ncols is None:
        ncols = 1 if dim <= 2 else 3
    nrows = int(np.ceil(combinations / ncols))
    for i in range(dim):
        if twoD:
            for j in range(i + 1, dim):
                plot_index = (nrows, ncols, idx)
                # print(plot_index)
                plt.subplot(*plot_index)
                do_plot2D(i, j, path[:, i], path[:, j])
                idx += 1
        else:
            plot_index = (nrows, ncols, idx)
            # print(plot_index)
            plt.subplot(*plot_index)
            data = path if len(path.shape) == 1 else path[:, i]
            do_plot1D(i, data)
            idx += 1
    if legend:
        # SHow legend on last plot only or on the missing plot space if there is one
        if combinations > 1 and combinations % ncols > 0:
            plt.legend(fontsize=legend_fontsize, bbox_to_anchor=(1, -0.1), loc="lower left")
        else:
            plt.legend(fontsize=legend_fontsize)


def plot_endpoints(path, label=None, text=None, ncols=2):
    raise DeprecationWarning("Use plot_path with only_endoints=True")

    def do_plot(x, y):
        plt.plot(x, y, '--', color='black', alpha=0.3, label=label)
        plt.scatter(x[-1], y[-1], marker='^', color='black', alpha=0.5)
        if text is not None:
            plt.text(x[0], y[0], text, color="black", fontsize=12)
        else:
            plt.scatter(x[0], y[0], marker='*', color='black', alpha=0.5)
        plt.legend()
        plt.grid()
        # print(x, y)

    if len(path.shape) == 1:
        do_plot(path[0], path[1])
        return
    dim = path.shape[1]
    combinations = int(scipy.misc.comb(dim, 2))
    idx = 1
    for i in range(dim):
        for j in range(i + 1, dim):
            plot_index = (int(np.ceil(combinations / ncols)), ncols, idx)
            # print(plot_index)
            plt.subplot(*plot_index)
            x, y = path[:, i], path[:, j]
            do_plot(x, y)
            idx += 1


def load_reference_structures(working_dir=None, equilibrated=True):
    if equilibrated:
        return load_reference_structure("3p0g-ligand/equilibrated.gro",
                                        working_dir=working_dir), load_reference_structure(
            "2rh1-noligand/equilibrated.gro", working_dir=working_dir)
    else:
        return load_reference_structure("active_CA.pdb", working_dir=working_dir), load_reference_structure(
            "inactive_CA.pdb", working_dir=working_dir)


def load_reference_structure(filename, working_dir=None):
    if working_dir is None:
        working_dir = reference_strucs_topdir
    return md.load(working_dir + "reference_structures/%s" % filename)


def load_many_reference_structures():
    # active, inactive = load_reference_structures()
    return [
        ('3P0G', load_pdb("3P0G")),
        ('2RH1', load_pdb("2RH1")),
        # ("4GBR", load_pdb("4GBR")),
        ("3PDS", load_pdb("3PDS")),
        # ("3SN6", load_pdb("3SN6")),
        # ("2R4R", load_pdb("2R4R")),
        # ("3NY9", load_pdb("3NY9")),
        # ("2RH1*apo", load_reference_structure("2rh1-noligand/equilibrated.gro")),
        # ("2RH1*apo-prod", load_reference_structure("2rh1-noligand/production.gro")),
        # ("3P0G*apo-prod", load_reference_structure("3p0g-noligand/production.gro")),
        # ("3P0G*apo", load_reference_structure("3p0g-noligand/equilibrated.gro")),
        # ("3P0G*-prod", load_reference_structure("3p0g-ligand/production.gro")),
        # ("3P0G*", load_reference_structure("3p0g-ligand/equilibrated.gro")),
        ("3SN6*", load_reference_structure("3sn6-ligand/equilibrated.gro")),
        ###BETA1
        ("4GPO", load_pdb("4GPO")),
        ("4GPO*",  load_reference_structure("beta1-apo/equilibrated.gro")),
    ]


def load_pdb(name, pdb_dir="/home/oliverfl/projects/gpcr/mega/protein_db/"):
    return md.load(pdb_dir + "%s/%s.pdb" % (name.upper(), name.lower()))


def rint(val):
    """
    Returns an int rounded to the correct value of the float
    """
    return int(np.rint(val))

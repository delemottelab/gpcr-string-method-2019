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
import mdtraj as md
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('notebooks/')
sys.path.append('notebooks/MD_common/')
import colvars
import utils

logger = logging.getLogger("InitialPoints")


def find_closest_frame(point, trajs, cv_evals):
    """
    Find closest frame of a simulation to a point in CV space
    :param point:
    :param cvs:
    :param trajs:
    :param cv_values:
    :return:
    """

    closest_frame = None
    closest_distance = 1e10
    for i, t in enumerate(trajs):
        dists = np.linalg.norm(point - cv_evals[i], axis=1)
        # print(dists.shape, len(t))
        mindist_index = dists.argmin()
        mindist = dists[mindist_index]
        if mindist < closest_distance:
            # logger.debug("Found frame in %s at time %s", simulation.id, t)
            closest_frame = t[mindist_index]
            closest_distance = mindist
    return closest_frame


def find_closest_frames(path, cvs, trajs):
    """Find representative frames for the path CV coordinates"""
    closest_frames = []
    cv_evals = np.array([colvars.eval_cvs(t, cvs) for t in trajs])
    for point in path:
        # TODO could optimze algorithm with a search tree or something
        # TODO could precompute the CV values for all iterations
        closest_frames.append(find_closest_frame(point, trajs, cv_evals))
    return closest_frames


def save_frames(directory, frames, filename="i0p%s-restrained.gro"):
    for i, f in enumerate(frames):
        utils.makedirs(directory % i, overwrite=False, backup=False)
        f.save(directory % i + filename % (i))


def to_single_trajectory(frames):
    t = None
    for f in frames:
        t = f if t is None else t + f
    return t


def plot_frames(cvs, string_path, frames):
    string_traj = to_single_trajectory(frames)
    utils.plot_path(string_path, label="String path coordinates", twoD=False)
    utils.plot_path(colvars.eval_cvs(string_traj, cvs), label="String path from simulation frames", twoD=False)
    plt.legend()
    plt.show()


def create_argparser():
    parser = argparse.ArgumentParser(epilog='String with swarms of trajectories code, Oliver Fleetwood 2017.')
    parser.add_argument('-cp', '--cvs_path', type=str, help='Path to binary CVs', required=True)
    parser.add_argument('-tr', '--traj', action='append', type=str, help='Path to trajectory or trajectories',
                        required=False, default=None)
    parser.add_argument('-to', '--top', action='append', type=str, help='Path to topology or topologies',
                        default=None)
    parser.add_argument('-od', '--out_dir', type=str, help='Path to save output',
                        default=None, required=True)
    parser.add_argument('-sp', '--string_path', type=str, help='string path')
    parser.add_argument("--reparametrize_stringpath", help="Distribute points equidistantly along string and save file",
                        type=lambda x: (str(x).lower() == 'true'),
                        default=False)
    # parser.add_argument('-q', '--atom_query', type=str, help='Query to select relevant atoms needed for CVs',
    #                     default="protein")
    return parser


def parse_args():
    parser = create_argparser()
    return parser.parse_args()


def find_initial_frames(args):
    stringpath = np.loadtxt(args.string_path)
    if args.reparametrize_stringpath:
        logger.info("Reparametrizing stringpath and saving it")
        stringpath = utils.reparametrize_path_iter(stringpath, arclength_weight=None)
        utils.backup_path(args.string_path)
        np.savetxt(args.string_path, stringpath)
    logger.info("Using stringpath %s ", stringpath)
    cvs = colvars.cvs_definition_reader.load_cvs(args.cvs_path + "cvs.json")
    trajs = []
    if args.traj is None:
        raise Exception("argument traj is required")
    for i, t in enumerate(args.traj):
        top = args.top[i] if i < len(args.top) else args.top[0]
        logger.debug("Loading traj and topology from %s, %s", t, top)
        traj = md.load(t, top=top)
        trajs.append(traj)
    logger.info("Loaded %s trajectories with %s frames", len(trajs), sum(len(t) for t in trajs))
    frames = find_closest_frames(stringpath, cvs, trajs)
    logger.info("Found initial frames")
    save_frames(args.out_dir, frames)
    logger.info("Saved. Plotting...")
    plot_frames(cvs, stringpath, frames)
    cmd = "for i in {0..%s}; do mkdir $i; gmx editconf -f ../%s$i.pdb  -o $i/i0p${i}-restrained.gro; done" % (
        len(stringpath), args.out_dir,)
    logger.info("To move the files into the directory, use a command such as:\n%s", cmd)


if __name__ == "__main__":
    logger.info("Starting")
    args = parse_args()
    # args.string_path = args.cvs_path + "string-paths-drorpath/string0.txt"
    # args.string_path = args.cvs_path + "string-paths/straight.txt"
    logger.info("Using args: %s", args)
    # generate_straight_path(args, utils.load_reference_structure("3p0g-ligand-equilibrated.gro"),
    #                        utils.load_reference_structure("2rh1-noligand-equilibrated.gro"), number_points=20)
    find_initial_frames(args)

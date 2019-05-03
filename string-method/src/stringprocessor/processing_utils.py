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
from os.path import abspath, exists
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import utils
import colvars

logger = logging.getLogger("SingleIterationPostProcessor")


def load_swarm(runner, point_idx, swarm_batch_idx, swarm_idx, ignore_missing_files=False,
               traj_filetypes=["xtc", "trr"], fallback_on_restrained_output=True):
    """
    Load a single swarm trajectory
    :param runner:
    :param point_idx:
    :param swarm_batch_idx:
    :param swarm_idx:
    :param ignore_missing_files: return None instead of throwing exception when files are not found
    :param traj_filetypes: types of trajectory file types to try and load
    :param fallback_on_restrained_output: Try to load the restrained last frame and the .gro output for this swarm and create a 2-frame trajectory from this
    :return: trajectory or None
    """
    # Load swarm trajectory
    found_traj = False
    for ftype in traj_filetypes:
        trajpath = runner.point_path(point_idx) + runner.swarm_name(point_idx, swarm_batch_idx, swarm_idx) + "." + ftype
        if exists(trajpath):
            found_traj = True
            break
    if not found_traj:
        # if fallback_on_restrained_output:
        #     return load_restrained_out_swarm(runner, point_idx, swarm_batch_idx, swarm_idx,
        #                                      ignore_missing_files=ignore_missing_files)
        if ignore_missing_files:
            logger.warn("File %s not found. Skipping this swarm", trajpath)
            return None
        else:
            raise IOError("Swarm %s-%s not found for point %s at iteration %s" % (
                swarm_batch_idx, swarm_idx, point_idx, runner.iteration))
    trajpath = abspath(trajpath)
    try:
        swarmtraj = md.load(trajpath, top=runner.topology)
    except Exception as ex:
        logger.exception(ex)
        logger.error("Could not load file %s.", trajpath)
        if fallback_on_restrained_output:
            # Quite often
            return load_restrained_out_swarm(runner, point_idx, swarm_batch_idx, swarm_idx,
                                             ignore_missing_files=ignore_missing_files)
        raise ex
    return swarmtraj


def load_restrained_out_swarm(runner, point_idx, swarm_batch_idx, swarm_idx, ignore_missing_files=False):
    """
    Try to load the restrained last frame and the .gro output for this swarm and create a 2-frame trajectory from this
    :param runner:
    :param point_idx:
    :param swarm_batch_idx:
    :param swarm_idx:
    :param ignore_missing_files: return None if trajectory is not found instead of throwing exception
    :return: a 2-frame trajectory
    """
    restrained_out = load_restrained(runner, point_idx, only_last_frame=True, ignore_missing_files=ignore_missing_files)
    swarm_out = load_swarm(runner, point_idx, swarm_batch_idx, swarm_idx, traj_filetypes=["gro"],
                           ignore_missing_files=ignore_missing_files, fallback_on_restrained_output=False)
    if restrained_out is None or swarm_out is None:
        msg = "%s not found from restrained_out. Skipping this swarm" % runner.swarm_name(point_idx, swarm_batch_idx,
                                                                                          swarm_idx)
        if ignore_missing_files:
            logger.warn(msg)
            return None
        else:
            raise IOError(msg)
    return restrained_out + swarm_out


def load_restrained(runner, point_idx, traj_filetypes=["trr", "xtc", "gro"], ignore_missing_files=False,
                    only_last_frame=True):
    """

    :param runner:
    :param point_idx:
    :param traj_filetypes:  types of trajectory file types to try and load
    :param ignore_missing_files:  return None if trajectory is not found instead of throwing exception
    :param only_last_frame: only return the last frame
    :return:
    """
    found_traj = False
    for ftype in traj_filetypes:
        trajpath = runner.point_path(point_idx) + runner.point_name(point_idx) + "-restrained.%s" % ftype
        if exists(trajpath):
            found_traj = True
            break
    if not found_traj:
        msg = "File %s not found. Skipping this swarm" % trajpath
        if ignore_missing_files:
            logger.warn(msg)
            return None
        else:
            raise IOError(msg)
    restrained = md.load(trajpath, top=runner.topology)
    return restrained[-1] if only_last_frame else restrained


def merge_restrained(runner, traj_filetypes=["trr", "xtc", "gro"]):
    """Merge all restrained simulation endpoints for this iteration"""
    traj = None
    for idx in range(len(runner.stringpath)):
        if runner.fixed_endpoints and (idx == 0 or idx == len(runner.stringpath) - 1):
            continue
        t = load_restrained(runner, idx, traj_filetypes=traj_filetypes)
        if traj is None:
            traj = t
        else:
            traj += t
    return traj


def save_string(string_filepath, iteration, stringpath, append_length=False):
    if append_length:
        suffix = "{}_len{}".format(iteration, len(stringpath))
    else:
        suffix = str(iteration)
    name = string_filepath % suffix
    np.savetxt(name, stringpath)


def create_stringpath_files_of_different_lengths(runner, short_stringpath, number_of_points_to_add=1):
    """Modifies the current iteration's output string and adds a point to it. The original string is saved with a suffix"""
    long_stringpath = utils.change_string_length(short_stringpath, len(short_stringpath) + number_of_points_to_add)
    save_string(runner.string_filepath, runner.iteration, short_stringpath, append_length=True)
    save_string(runner.string_filepath, runner.iteration, long_stringpath, append_length=True)
    logger.info("Added %s points to string for iteration %s. New string length=%s", number_of_points_to_add,
                runner.iteration, len(long_stringpath))
    return long_stringpath


def save_input_coordinate_mapping(string_filepath, iteration, input_coordinate_mapping):
    filepath = string_filepath % (str(iteration) + "-mapping")
    np.savetxt(filepath, input_coordinate_mapping)

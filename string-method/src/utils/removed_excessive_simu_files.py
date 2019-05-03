from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

from analysis.FE_analysis.simu_config import get_args_for_simulation
import utils
import run_simulation as rs

logger = logging.getLogger("remove_files")


def delete_backup_files(runner):
    return delete_for_regex(runner, "#*#")


def delete_for_filetype(runner, filetype):
    return delete_for_regex(runner, "*.{}".format(filetype))


def delete_input_coordinate_files(runner):
    return delete_for_regex(runner, "i*p*in.gro")


def delete_for_regex(runner, regex):
    file_regex = runner.simulation_dir + "*/*/" + regex
    file_list = utils.sorted_alphanumeric(glob.glob(file_regex))
    nfiles_removed = 0
    total_filesize_mb = 0
    size_mb = 1024 * 1024
    for file in file_list:
        try:
            filesize = os.stat(file).st_size / size_mb
            os.remove(file)
            nfiles_removed += 1
            total_filesize_mb += filesize
        except OSError as ex:
            logger.exception(ex)
    logger.info("Deleted %s files for regex %s, totalling %s Mb in file size", nfiles_removed, file_regex,
                total_filesize_mb)
    return nfiles_removed, total_filesize_mb


def start():
    logger.info("----------------Starting removing files------------")
    simulations = [
        "apo-optimized",
        "holo-optimized",
        "endpoints-holo",
        "endpoints-apo",
        "to3sn6-holo",
        "to3sn6-apo",
        "straight-holo-optimized",
        "holo-curved",
        "apo-curved",
        "holo-straight",
        "beta1-apo",
        "pierre-ash79",
        "pierre-asp79_Na"
    ]
    delete_filetypes = [
        "cpt",
        "tpr",
        "log",
        "trr",
    ]
    total_nfiles_removed, total_filesize_mb = 0, 0
    for simu_id in simulations:
        args = get_args_for_simulation(simu_id)
        runner = rs.StringIterationRunner(args)
        for filetype in delete_filetypes:
            nfiles_removed, filesize_mb = delete_for_filetype(runner, filetype)
            total_nfiles_removed += nfiles_removed
            total_filesize_mb += filesize_mb
        nfiles_removed, filesize_mb = delete_input_coordinate_files(runner)
        total_nfiles_removed += nfiles_removed
        total_filesize_mb += filesize_mb
        nfiles_removed, filesize_mb = delete_backup_files(runner)
        total_nfiles_removed += nfiles_removed
        total_filesize_mb += filesize_mb
        # TODO submission files
        # TODO maybe edr files
        # TODO compress everything to xtc and only necessary frames
    logger.info("##########SUMMARY##########\nDeleted %s files in total of total file size %s Gb", total_nfiles_removed,
                total_filesize_mb / 1024)


if __name__ == "__main__":
    start()

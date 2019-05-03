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
from stringprocessor import *
# import matplotlib as mpl
# mpl.use('GTK3Cairo')
from jobs.job_executors import *
from os.path import abspath
import os
import run_simulation as rs
import matplotlib.pyplot as plt
import traceback
import glob
from system_setup import create_cvs

logger = logging.getLogger("ExtraAnalysis")

logger.info(
    """
    Note
    THE CODE IN THIS FILE IS DELIBERATELY NOT MAINTAINED AND REFACTORED THAT MUCH
    It is intended to be used for ad hoc analysis
    """
)


def get_cv_description(cvid, use_simpler_names=False):
    """DEPRECATED, OLD CODE WHEN I HAD BAD CV IDs"""
    if isinstance(cvid, colvars.CV):
        cvid = cvid.id
    if cvid == "<|color 0-1 distance|>":
        cvid = "ALA76-CYS327"
    elif cvid == "<|color 0-2 distance|>":
        cvid = "GLU268-LYS147"
    elif cvid == "<|color 1-2 distance|>":
        cvid = "CYS327-LYS147"
    elif cvid == "<|color 3-4 distance|>":
        cvid = "LEU115-LEU275"
    elif cvid == "<|color 5-6 distance|>":
        cvid = "LYS267-PHE223"
    if use_simpler_names:
        if cvid == "ALA76-CYS327":
            cvid = "TM2-TM7"
        elif cvid == "GLU268-LYS147":
            cvid = "TM6-TM4"
        elif cvid == "CYS327-LYS147":
            cvid = "TM7-TM4"
        elif cvid == "LEU115-LEU275":
            cvid = "TM3-TM6"
        elif cvid == "LYS267-PHE223":
            cvid = "TM6-TM5"
        # Other variables
        if cvid == "helix_63_dist(CA)":  # residue  131, 272
            cvid = "TM6-TM3"
        elif cvid == "LigandBinding":
            cvid = "Ser207 Orientation"
        elif cvid == "water_solubility_res266_0.8nm":
            cvid = "#H20 within 0.8 nm"
        elif cvid == "inactive322-327rmsd":
            cvid = "NPxxY RMSD to inactive"
    return cvid


def plot_restrained_simus(args, rescale=False, show_label=True, show_text=False):
    plot_simu_trajs(args, "-restrained", rescale=rescale, show_label=show_label, show_text=show_text)


def plot_thermalization_simus(args, rescale=False):
    plot_simu_trajs(args, "-thermalization", rescale=rescale)


def plot_simu_trajs(args, type, show_reference_structures=True, rescale=False, show_label=True, show_text=False):
    runner = rs.StringIterationRunner(args)
    plt.title(type + " SIMUS")
    if show_reference_structures:
        plot_reference_structures(runner, rescale=rescale)
    utils.plot_path(colvars.rescale_evals(runner.stringpath, runner.cvs) if rescale else runner.stringpath,
                    label="Initial String")
    trajcount = 0
    for i in range(len(runner.stringpath)):
        trajpath = runner.point_path(i) + runner.point_name(i)
        filename = trajpath + type + ".trr"
        if not os.path.exists(filename):
            logger.warn("File %s not found for point %s. Skipping", filename, i)
            continue
        traj = md.load(abspath(trajpath + type + ".trr"),
                       top=runner.topology)  # trajpath + type + ".gro"))
        description = type + str(i)
        utils.plot_path(colvars.eval_cvs(traj, runner.cvs, rescale=rescale), label=description if show_label else None,
                        text=description if show_text else None)
        trajcount += 1
    if trajcount < 4:
        plt.legend()
    plt.show()


def plot_input_frames(args):
    """PLOT THE INITIAL COORDINATES; BEFORE MINIMIZATION FOR THIS SIMULATION"""
    runner = rs.StringIterationRunner(args)
    plt.title("INPUT FRAMES")
    plot_reference_structures(runner)
    utils.plot_path(runner.stringpath, label="Initial String")
    inittraj = None
    for i in range(len(runner.stringpath)):
        trajpath = runner.point_path(i) + runner.point_name(i)
        infile = trajpath + "-in.gro"
        if not os.path.exists(infile):
            logger.warn("%s files not found for point %s. Skipping", infile, i)
            continue
        intraj = md.load(abspath(infile))
        inittraj = intraj if inittraj is None else inittraj + intraj
    logger.debug("init-traj: %s", inittraj)
    inpath = colvars.eval_cvs(inittraj, runner.cvs)
    utils.plot_path(inpath, label="Init coordinates")
    # utils.plot_endpoints(np.matrix([colvars.eval_cvs(traj, runner.cvs) for traj in inittraj]), label="Init gros2")
    plt.legend()
    plt.show()
    if len(runner.stringpath) == len(inpath):
        diff = np.linalg.norm(runner.stringpath - inpath, axis=1)
    elif len(runner.stringpath) == (len(inpath) + 2):
        # Fixed endpoints
        diff = np.linalg.norm(runner.stringpath[0:-2] - inpath, axis=1)
    else:
        logger.error("Path lengths differ: %s vs %s", len(runner.stringpath))
        return
    plt.plot(diff, label="Distance between input coordinates and string")
    plt.scatter([range(0, len(diff))], diff)
    plt.legend()
    plt.show()


def compute_drifted_string(args, plot=True, iteration=None, twoD=False):
    """Return the unparametrized string after the swarms drift
    :param plot:
    """
    runner = rs.StringIterationRunner(args)
    if iteration is not None:
        runner.init_iteration(iteration)

    plt.title("SWARMS PATHS")
    processor = SingleIterationPostProcessor(runner, save=False, plot=False)
    runner.run()
    driftstring = processor.compute_drifted_string()
    reparametrized_weights = processor.compute_new_stringpath()
    reparametrized_noweights = utils.reparametrize_path_iter(driftstring, arclength_weight=None)
    if plot:
        utils.plot_path(runner.stringpath, label="Input", twoD=twoD)
        utils.plot_path(reparametrized_weights, label="reparametrized weights", twoD=twoD)
        utils.plot_path(reparametrized_noweights,
                        label="reparametrized no weights", twoD=twoD)
        utils.plot_path(driftstring, label="Drifted", twoD=twoD)
        plt.show()
    logger.debug("Convergence to input path between reparametrized with weights %s and without weights %s",
                 np.linalg.norm(reparametrized_weights - runner.stringpath) / np.linalg.norm(runner.stringpath),
                 np.linalg.norm(reparametrized_noweights - runner.stringpath) / np.linalg.norm(runner.stringpath))
    logger.debug("Convergence to drifted path between reparametrized with weights %s and without weights %s",
                 np.linalg.norm(reparametrized_weights - driftstring) / np.linalg.norm(driftstring),
                 np.linalg.norm(reparametrized_noweights - driftstring) / np.linalg.norm(driftstring))
    return driftstring


def plot_swarms_path(args):
    """PLOT THE SWARMS FOR THIS ITERATION"""
    runner = rs.StringIterationRunner(args)
    plt.title("SWARMS PATHS")
    processor = SingleIterationPostProcessor(runner, save=False, plot=True, ignore_missing_files=True)
    processor.run()
    logger.info("Convergence after iteration %s: %s", runner.iteration, processor.converge)


def plot_dror_path(runner):
    drorpath = np.loadtxt(runner.cvs_dir + "dror_path.txt")
    utils.plot_path(drorpath, label="Dror path")
    return drorpath


def compare_dror_path(args):
    """Find all string-paths and plot them"""
    runner = rs.StringIterationRunner(args)
    plt.title("Dror String vs. first/lats string")
    plot_reference_structures(runner)
    drorpath = plot_dror_path(runner)
    first = np.loadtxt(runner.working_dir + runner.string_filepath % 0)
    current = np.loadtxt(runner.working_dir + runner.string_filepath % runner.iteration)
    utils.plot_path(first, label="First")
    utils.plot_path(current, label="Iteration %s" % runner.iteration)
    plt.legend()
    reldiff = np.abs(utils.compute_path_length(drorpath) - utils.compute_path_length(
        current) / utils.compute_path_length(drorpath))
    logger.info("Relative difference between Dror path and final: %s percent", 100 * reldiff)
    plt.show()


def compute_average_string(runner, start_iteration=25, end_iteration=999, rescale=True, legend=True, plot=True,
                           save=True, twoD=True, cv_indices=None):
    """Find all string-paths and compute their average them"""
    if cv_indices is None:
        cv_indices = [i for i in range(runner.stringpath.shape[1])]
    runner.cvs = np.array(runner.cvs)[cv_indices]
    all = []
    last = None
    lastiter = start_iteration
    for i in range(start_iteration, end_iteration + 1):
        try:
            lastiter = i
            single_path = np.loadtxt(runner.string_filepath % i)
            single_path = single_path[:, cv_indices]
            if last is not None and len(last) != len(single_path):
                logger.warn(
                    "Different number of points on string between iterations %s and %s: %s vs %s. Changing length of strings analytically",
                    i - 1, i, len(last), len(single_path))
                # Change length of all strings here
                for i, sp in enumerate(all):
                    all[i] = utils.change_string_length(sp, len(single_path))
            last = single_path
            all.append(single_path)
        except IOError as err:
            logger.error(traceback.format_exc(err))
            logger.info("Did not find string %s in filepath %s. Not looking for sequential strings", i,
                        runner.string_filepath)
            lastiter = i - 1
            break
    if last is None:
        logger.warn("No strings found. Not doing anything more")
        return None, None
    strings = np.empty((len(all), last.shape[0], last.shape[1]))
    for i, s in enumerate(all):
        strings[i, :, :] = colvars.rescale_evals(s, runner.cvs) if rescale else s
    avg_string = np.mean(strings, axis=0)
    logger.info("Average string:\n%s", avg_string)
    iterations_label = "%s-%s" % (start_iteration, lastiter)
    if save:
        np.savetxt(runner.string_filepath % ("average" + iterations_label), avg_string)
    dim = last.shape[1]
    xvalues = np.arange(0, len(avg_string))
    label = "Average string iter %s" % iterations_label
    iterations_label
    if plot:
        plt.title(label)
        for i in range(dim):
            plot_index = (int(np.ceil(dim / 2)), 2, i + 1)
            plt.subplot(*plot_index)
            plt.boxplot(strings[:, :, i], showmeans=True, positions=xvalues)
            plt.plot(xvalues, avg_string[:, i], label=label)
            if legend:
                plt.legend()
                plt.xlabel("inactivation")
                plt.ylabel(get_cv_description(runner.cvs[i].id, use_simpler_names=True))
            plt.grid()
        plt.show()
        # compute the product of stds in all dimensions to estimate the mutidimensional error
        std = np.std(strings, axis=0)
        std_prod = np.prod(std, axis=1)
        std_prod = std_prod / np.max(std_prod)
        plt.plot(xvalues, std_prod, label="STD multidimension")
        plt.scatter(xvalues, std_prod)
        for i, cv in enumerate(runner.cvs):
            plt.plot(xvalues, std[:, i] / np.max(std[:, i]), '--', label=get_cv_description(cv.id) + "STD")
        if legend:
            plt.legend()
            plt.xlabel("String point")
            plt.ylabel("Normalized STD")
        plt.show()
    if plot and twoD:
        utils.plot_path(avg_string, label=label)
        plt.grid()
        plt.show()
    return avg_string, strings


def compute_rolling_average_string_convergence(args, strings_per_average=33, start_iteration=None, end_iteration=9999,
                                               comparison_step=None,
                                               plot=True):
    if start_iteration is None:
        start_iteration = 2  # strings_per_average
    if comparison_step is None:
        comparison_step = strings_per_average
    runner = rs.StringIterationRunner(args)
    all = []
    iterations = []
    convergences = []
    total_avg_string, strings = compute_average_string(runner, start_iteration=start_iteration,
                                                       end_iteration=end_iteration,
                                                       rescale=False, legend=False, plot=False,
                                                       save=False, twoD=False)
    total_avg = np.linalg.norm(total_avg_string)
    last = None
    for i in range(start_iteration, end_iteration + 1):
        try:
            runner.init_iteration(start_iteration)
            path, strings = compute_average_string(runner,
                                                   start_iteration=max(1, i - strings_per_average + 1),
                                                   end_iteration=i,
                                                   rescale=False, legend=False, plot=False,
                                                   save=False, twoD=False)
            if path is None or (i >= strings_per_average and len(strings) != strings_per_average):
                break
            all.append(path)
            compare_index = i - start_iteration - comparison_step
            if compare_index >= 0:
                last = all[compare_index]
                if len(last) != len(path):
                    logger.warn("Different length of previous path for iteration %s. %s vs %s", i, len(last), len(path))
                    last = utils.change_string_length(last, len(path))
                c = np.linalg.norm(last - path) / total_avg
                convergences.append(c)
                iterations.append(i)
        except IOError as err:
            logger.error(err)
            logger.info("Did not find string %s in filepath %s. Not looking for sequential strings", i,
                        runner.string_filepath)
            break
    if len(iterations) == 0:
        logger.warn("Nothing done")
        return None
    convergences = np.array(convergences)
    iterations = np.array(iterations)
    result = np.empty((len(iterations), 2))
    result[:, 0] = iterations
    result[:, 1] = convergences
    # print(convergences)
    all = np.array(all)
    if plot:
        utils.plot_path(result, axis_labels=["Iteration#", "Convergence"], twoD=True,
                        label=utils.simuid_to_label.get(runner.simu_id, runner.simu_id))
        plt.title("rolling avg. for %s strings, comparison to average %s iterations before" % (
            strings_per_average, comparison_step))
        plt.show()

    return result


def plot_all_average_strings(args, strings_per_average=10, start_iteration=1, end_iteration=999,
                             rescale=False, legend=True, twoD=False, plot_strings=True,
                             do_plot_reference_structures=True, plot_boxes=True,
                             cv_indices=None, plot_convergence=True, accumulate=False):
    """Find all string-paths and plot them"""
    runner = rs.StringIterationRunner(args)
    if cv_indices is None:
        cv_indices = [i for i in range(runner.stringpath.shape[1])]
    plt.title("ALL STRINGS")
    if do_plot_reference_structures:
        plot_reference_structures(runner, rescale=rescale, twoD=twoD)
    last, last_iteration = None, start_iteration
    convergences = []
    last_iteration = start_iteration
    for string_index in range(start_iteration, end_iteration, strings_per_average):
        first_index = start_iteration if accumulate else string_index
        last_index = string_index + strings_per_average
        try:
            runner.init_iteration(string_index)
            path, strings = compute_average_string(runner, start_iteration=first_index, end_iteration=last_index,
                                                   rescale=rescale, legend=legend, plot=False,
                                                   save=False, twoD=twoD)
            if path is None:
                break
            path = path[:, cv_indices]
            # path = np.loadtxt(runner.working_dir + runner.string_filepath % i)
            if last is not None:
                if len(last) == len(path):
                    dist = np.linalg.norm(last - path)
                else:
                    dist = np.linalg.norm(utils.change_string_length(last, len(path)) - path)
                    logger.warn("Number points differ between iterations %s and %s", last_iteration, last_index)
                    # convergences.append(np.nan)
                convergence = dist / np.linalg.norm(path)
                logger.info("Converge between iterations %s and %s: %s. Absolute distance: %s",
                            last_iteration, last_index,
                            convergence,
                            dist)
                convergences.append(convergence)
            if plot_strings:
                boxplot = plot_boxes and (string_index + strings_per_average) >= end_iteration
                utils.plot_path(path,
                                label="i%s-i%s" % (first_index, last_index),
                                text=None,
                                scatter=False,
                                boxplot_data=strings[:, :, cv_indices] if boxplot else None,
                                legend=legend,
                                twoD=twoD,
                                ncols=2,
                                axis_labels=([] if twoD else ["inactivation"]) +
                                            [get_cv_description(cv.id, use_simpler_names=True) for cv in
                                             np.array(runner.cvs)[cv_indices]])
                # utils.plot_path(plotpath, label="Stringpath %s" % i, text=None, legend=legend)

            last = path
            last_iteration = last_index
        except IOError as err:
            logger.error(err)
            logger.info("Did not find string %s in filepath %s. Not looking for sequential strings", string_index,
                        runner.string_filepath)
            break
    if last is None:
        return
    # plt.title(runner.simu_id)
    plt.show()
    if plot_convergence:
        plt.plot(convergences)
        plt.ylabel(r'$|\bar{s_i}-\bar{s}_{i+1}|/|\bar{s}_{i+1}|$')
        plt.xlabel(r"i")
        plt.title("Convergence")
        plt.show()


def plot_custom_strings(args, filepaths=[], additional_strings=[], rescale=False, legend=True,
                        twoD=False, cv_indices=None, show_reference_structures=False):
    """Find all string-paths and plot them"""
    runner = rs.StringIterationRunner(args)
    if cv_indices is None:
        cv_indices = [i for i in range(runner.stringpath.shape[1])]
    plt.title("ALL STRINGS")
    if show_reference_structures:
        plot_reference_structures(runner, rescale=rescale, twoD=twoD)
    last = None
    for fp in filepaths:
        try:
            path = np.loadtxt(fp)[:, cv_indices]
            plotpath = colvars.rescale_evals(path, runner.cvs) if rescale else path
            utils.plot_path(plotpath, label=fp.split("/")[-1], text=None, legend=legend, twoD=twoD,
                            axis_labels=[get_cv_description(cv, use_simpler_names=True) for cv in
                                         np.array(runner.cvs)[cv_indices]])
            plt.grid()
            # utils.plot_path(plotpath, label="Stringpath %s" % i, text=None, legend=legend)

            last = path
        except IOError as err:
            tb = traceback.format_exc()
            logger.error(tb)
            logger.info("Did not find string in filepath %s. Not looking for sequential strings", fp)
            break
    for i, path in enumerate(additional_strings):
        plotpath = colvars.rescale_evals(path, runner.cvs) if rescale else path
        utils.plot_path(plotpath, label="a%s" % i, text=None, legend=legend, twoD=twoD,
                        axis_labels=[get_cv_description(cv, use_simpler_names=True) for cv in
                                     np.array(runner.cvs)[cv_indices]])
        plt.grid()
        last = path
    if last is None:
        return
    if legend:
        plt.legend()
    plt.show()


def plot_all_strings(args, plot_frequency=1, start_iteration=1, end_iteration=None, rescale=False, legend=True,
                     twoD=False,
                     plot_restrained=False, cv_indices=None, plot_convergence=True, plot_reference_structures=True):
    """Find all string-paths and plot them"""
    runner = rs.StringIterationRunner(args)
    runner.cvs = np.array(runner.cvs)
    if cv_indices is None:
        cv_indices = [i for i in range(runner.stringpath.shape[1])]
    plt.title("ALL STRINGS")
    if plot_reference_structures:
        plot_reference_structures(runner, rescale=rescale, twoD=twoD)
    last = None
    convergences = []
    for i in range(start_iteration, 2000 if end_iteration is None else end_iteration):
        try:
            runner.init_iteration(i)
            path = runner.stringpath[:, cv_indices]
            # path = np.loadtxt(runner.working_dir + runner.string_filepath % i)
            if last is not None:
                if len(last) == len(path):
                    dist = np.linalg.norm(last - path)
                    convergence = dist / np.linalg.norm(path)
                    logger.info("Converge between iterations %s and %s: %s. Absolute distance: %s", i - 1, i,
                                convergence,
                                dist)
                    convergences.append(convergence)
                else:
                    logger.warn("Number points differ between iterations %s and %s", i, i - 1)
                    convergences.append(np.nan)
            if (i + start_iteration - 1) % plot_frequency == 0:
                plotpath = colvars.rescale_evals(path, runner.cvs[cv_indices]) if rescale else path
                utils.plot_path(plotpath, label="Stringpath %s" % i, text=None, legend=legend, twoD=twoD,
                                axis_labels=[get_cv_description(cv.id, use_simpler_names=True) for cv in
                                             np.array(runner.cvs)[cv_indices]])
                if plot_restrained:
                    restrainedpath = SingleIterationPostProcessor(runner).compute_string_from_restrained()
                    restrainedpath = colvars.rescale_evals(restrainedpath, runner.cvs) if rescale else restrainedpath
                    utils.plot_path(restrainedpath, label="Restrained {}".format(i), twoD=twoD)
                plt.grid()
                # utils.plot_path(plotpath, label="Stringpath %s" % i, text=None, legend=legend)

            last = path
        except IOError as err:
            tb = traceback.format_exc()
            logger.error(tb)
            logger.info("Did not find string %s in filepath %s. Not looking for sequential strings", i,
                        runner.string_filepath)
            break
    if last is None:
        return
    if legend:
        plt.legend()
    plt.show()
    if plot_convergence:
        plt.plot(convergences)
        plt.ylabel(r'$|\bar{s_i}-\bar{s}_{i+1}|/|\bar{s}_{i+1}|$')
        plt.xlabel(r"i")
        plt.title("Convergence")
        plt.show()


def plot_minimizations(args, rescale=False, plot_dist=False):
    """PLOT ALL THE MINIMZATION RESULTS FOR THIS SIMULATION"""
    runner = rs.StringIterationRunner(args)
    plt.title("Minimization results")
    plot_reference_structures(runner, rescale=rescale)
    utils.plot_path(colvars.rescale_evals(runner.stringpath, runner.cvs) if rescale else runner.stringpath,
                    label="Initial String")
    stringdist = np.zeros((len(runner.stringpath),))
    for i in range(len(runner.stringpath)):
        trajpath = runner.point_path(i) + runner.point_name(i) + "-minimization.gro"
        if os.path.exists(trajpath):
            traj = md.load(abspath(trajpath))
            logger.debug("point %s: %s", i, traj)
            val = colvars.eval_cvs(traj, runner.cvs, rescale=rescale)
            utils.plot_path(val, text=str(i))
            stringdist[i] = np.linalg.norm(val - runner.stringpath[i])
        else:
            logger.warn("Minimization result for iteration %s not found", i)
    plt.legend()
    plt.show()
    if plot_dist:
        plt.title("minimization distance to points on string")
        plt.plot(stringdist)
        plt.show()


def load_reference_structures(runner):
    # TODO load more structures
    return utils.load_reference_structures()


def plot_reference_structures(runner, rescale=False, twoD=True):
    active, inactive = load_reference_structures(runner)
    utils.plot_path(colvars.eval_cvs(active, runner.cvs, rescale=rescale), text="Active", twoD=twoD)
    utils.plot_path(colvars.eval_cvs(inactive, runner.cvs, rescale=rescale), text="Inactive", twoD=twoD)


def plot_any_traj(runner, trajpath, rescale=False, show_reference_structures=True,
                  label=None,
                  stride=1, trajformat="trr", show_plot=True, twoD=True):
    traj = md.load(trajpath + "." + trajformat, top=runner.topology, stride=stride)
    if show_reference_structures:
        plot_reference_structures(runner, rescale=rescale, twoD=twoD)
    utils.plot_path(colvars.eval_cvs(traj, runner.cvs, rescale=rescale), label=label,
                    axis_labels=[cv.id for cv in runner.cvs], twoD=twoD)
    if show_plot:
        plt.show()
    return traj


def merge_swarms_to_file(args, filename, iterations=None, ignore_missing_files=True):
    runner = rs.StringIterationRunner(args)
    if iterations is None:
        iterations = [runner.iteration]
    traj = None
    for i in iterations:
        runner.init_iteration(i)
        processor = SingleIterationPostProcessor(runner, ignore_missing_files=ignore_missing_files)
        iterationtraj = processor.merge_swarms()
        if traj is None:
            traj = iterationtraj
        else:
            traj += iterationtraj
        logger.debug("Done with iteration %s, generated traj: %s", i, iterationtraj)
    traj.save(filename)
    return traj


def merge_restrained_to_file(args, filename, iterations=None):
    runner = rs.StringIterationRunner(args)
    if iterations is None:
        iterations = [runner.iteration]
    traj = None
    for i in iterations:
        runner.init_iteration(i)
        iterationtraj = merge_restrained(runner)
        if traj is None:
            traj = iterationtraj
        else:
            traj += iterationtraj
        logger.debug("Done with iteration %s, generated traj: %s", i, iterationtraj)
    traj.save(filename)
    return traj


def show_all_freemd(args, traj_stride=10, startpart=2, endpart=78, file_stride=1, save=False):
    runner = rs.StringIterationRunner(args)
    traj = None
    for i in range(startpart, endpart + 1, file_stride):
        if i < 3:
            filename = "/home/oliverfl/projects/gpcr/simulations/freemd/freemd-dec2017/3p0g_prod4"
        else:
            endstr = "0" + str(i) if i < 10 else str(i)
            filename = "/home/oliverfl/projects/gpcr/simulations/freemd/freemd-dec2017/3p0g_prod4.part00" + endstr
        logger.debug("Loading file %s", filename)
        t = plot_any_traj(runner,
                          filename,
                          rescale=False,
                          # label="part %s" % i,
                          trajformat="trr", stride=traj_stride, show_plot=False)
        if save:
            traj = t if traj is None else traj + t

    if save:
        traj.save("/home/oliverfl/projects/gpcr/simulations/freemd/freemd-dec2017/merged-part%sto%sstride%s.xtc" % (
            startpart, endpart, traj_stride))
    plt.show()


def compute_string_with_new_length(args, new_length, iteration=None, savepath=None, plot=True):
    runner = rs.StringIterationRunner(args)
    if iteration is not None:
        runner.init_iteration(iteration)
    new_stringpath = utils.change_string_length(runner.stringpath, new_length)
    if plot:
        utils.plot_path(runner.stringpath, label="Original path", twoD=True)
        utils.plot_path(new_stringpath, label="New path", twoD=True)
        plt.show()
    logger.info("New string with new length:%s\n%s", new_stringpath)
    if savepath is None:
        print(runner.string_filepath)
        savepath = runner.string_filepath % (str(runner.iteration - 1) + "_len" + str(new_length))
    np.savetxt(savepath, new_stringpath)
    logger.info("Saved to %s", savepath)


def compare_5cvs_pbc_condition(args, point_filetype="gro", tol=1e-5):
    runner = rs.StringIterationRunner(args)
    cvs_pbc = create_cvs.create_5cvs(normalize=False, pbc=True)
    cv_nopbc = create_cvs.create_5cvs(normalize=False, pbc=False)

    # Load all restrained simus.
    fake_iter = 999
    point_dir = runner.point_path(fake_iter, iteration=fake_iter)
    point_path = point_dir + runner.point_name(fake_iter, iteration=fake_iter) + "-restrained." + point_filetype
    point_path = point_path.replace(str(fake_iter), "*")
    all_restrained_paths = sorted(glob.glob(point_path))
    ntraj = len(all_restrained_paths)
    logger.debug("Comparing %s trajectories ", ntraj)
    for traj_idx, traj_path in enumerate(all_restrained_paths):
        if traj_idx % np.ceil(ntraj / 100) == 0:
            logger.debug("Comparing traj %s/%s", traj_idx, ntraj)
        traj = md.load(traj_path, top=runner.topology)
        evals_pbc = colvars.eval_cvs(traj, cvs_pbc)
        evals_nopbc = colvars.eval_cvs(traj, cv_nopbc)
        # For every restrained simu, compare pbc to no pbc. See if they differ.
        for frame_idx, frame_pbc in enumerate(evals_pbc):
            frame_nopbc = evals_nopbc[frame_idx]
            for cv_idx, val_pbc in enumerate(frame_pbc):
                val_nopbc = frame_nopbc[cv_idx]
                if not (val_nopbc - val_pbc) < tol:
                    raise Exception("The two values do not equal: %s, %s. For cv %s and trajectory %s" % (
                        val_pbc, val_nopbc, cvs_pbc[cv_idx], traj_path))


if __name__ == "__main__":
    parser = rs.create_argparser()
    args = parser.parse_args()
    if args.start_mode != 'analysis':
        logger.error("Startmode %s not supported", args.start_mode)
    else:
        # compare_dror_path(args)
        # plot_custom_strings(args, [
        #     # "/home/oliverfl/projects/gpcr/freemd_ligand-path_fixedep.txt",
        #     # "/home/oliverfl/git/string-method/gpcr/cvs/cvs-len5_good/string-paths-drorpath/dror_path.txt",
        #     "/home/oliverfl/projects/gpcr/simulations/strings/april/holo5-drorpath/gpcr/cvs/cvs-len5_good/string-paths/stringaverage150-188.txt",
        #     "/home/oliverfl/projects/gpcr/simulations/strings/christmas/5cvs-straight/gpcr/cvs/cvs-len5_good/string-paths/stringaverage150-200.txt",
        #
        # ],
        #                     rescale=False,
        #                     legend=True,
        #                     twoD=True, cv_indices=None)
        # plot_custom_strings(args,
        #                     # filepaths=["../gpcr/cvs/cvs-len5_good/string-paths-drorpath/dror_path.txt"],
        #                     additional_strings=[
        #                         # utils.reparametrize_path_iter(# np.loadtxt("../gpcr/cvs/cvs-len5_good/string-paths-drorpath/stringaverage93-98.txt")),
        #                         # np.loadtxt(
        #                         #     "/data/oliver/pnas2011b-Dror-gpcr/strings/jan/5cvs-drorpath/gpcr/cvs/cvs-len5_good/string-paths/stringaverage80-99.txt"),
        #
        #                         # utils.reparametrize_path_grid(np.loadtxt("../gpcr/cvs/cvs-len5_good/string-paths-drorpath/dror_path.txt"))
        #
        #                         np.loadtxt(
        #                             "/data/oliver/pnas2011b-Dror-gpcr/strings/april/holo5-drorpath/gpcr/cvs/cvs-len5_good/string-paths/stringaverage150-200.txt"),
        #                         # np.loadtxt(
        #                         #     "/data/oliver/pnas2011b-Dror-gpcr/strings/april/endpoints-holo5/gpcr/cvs/cvs-len5_good/string-paths/string10.txt"),
        #                         # np.loadtxt(
        #                         #     "/data/oliver/pnas2011b-Dror-gpcr/strings/april/endpoints-holo5/gpcr/cvs/cvs-len5_good/string-paths/string5.txt"),
        #                         np.loadtxt(
        #                             "/data/oliver/pnas2011b-Dror-gpcr/strings/jun/holo5-optimized-straight/gpcr/cvs/cvs-len5_good/string-paths/stringaverage160-212.txt"),
        #                         np.loadtxt(
        #                             "/data/oliver/pnas2011b-Dror-gpcr/strings/jun/holo5-optimized-straight/gpcr/cvs/cvs-len5_good/string-paths/stringaverage50-100.txt"),
        #                     ],
        #                     rescale=False,
        #                     legend=True,
        #                     twoD=True, cv_indices=None)
        # plot_all_strings(args, plot_frequency=1, start_iteration=1, end_iteration=300, rescale=True, legend=True,
        #                  twoD=True,cv_indices=None, plot_convergence=False, plot_reference_structures=False)
        # plot_restrained=False, cv_indices=[0,1], plot_convergence=True)
        # plot_all_average_strings(args, strings_per_average=77, start_iteration=1, end_iteration=232,
        #                          rescale=True, legend=True, twoD=False, plot_strings=True,
        #                          do_plot_reference_structures=False, plot_boxes=True,
        #                          cv_indices=None, plot_convergence=False, accumulate=False)
        # compute_rolling_average_string_convergence(args,
        #                                            strings_per_average=21,  # comparison_step=81/51,
        #                                            plot=True)
        # logger.info("Drifted string after swarms (unparametrized):\n%s",
        #             compute_drifted_string(args, plot=True, iteration=10))
        # compute_string_with_new_length(args, iteration=None, new_length=20, savepath=None)
        compute_average_string(rs.StringIterationRunner(args), start_iteration=200, end_iteration=282, rescale=False,
                               legend=True, twoD=True, cv_indices=None)
        # plot_input_frames(args)
        # plot_minimizations(args, rescale=True, plot_dist=False)
        # plot_swarms_path(args)
        # plot_thermalization_simus(args, rescale=True)
        # plot_restrained_simus(args, rescale=False, show_label=False, show_text=True)
        # plot_any_traj(rs.StringIterationRunner(args),
        #               "/home/oliver/slask/3SN6-holo-charmm-gui/gromacs/step7_1to3",
        #               trajformat="xtc",
        #               rescale=False,
        #               show_reference_structures=True,
        #               label="MD",
        #               twoD=False,
        #               stride=1)
        # show_all_freemd(args, traj_stride=5, startpart=2, endpart=78, file_stride=1, save=True)
        # plot_any_traj(args, "/home/oliverfl/projects/gpcr/simulations/targetedmd/rmsd-straight/targetedmd-restrained",
        #               topologypath="../gpcr/confout.gro",
        #               rescale=True,
        #               label="Target MD", trajformat="xtc", stride=5)
        # plot_any_traj(rs.StringIterationRunner(args),
        #               "/data/oliver/pnas2011b-Dror-gpcr/targetedmd/2018/targetedmd-3p0g-3sn6-holo/targetedmd-restrained",
        #               rescale=True,
        #               show_reference_structures=False,
        #               label="Targetedmd",
        #               trajformat="xtc",
        #               stride=1)
        # merge_swarms_to_file(args,
        #                      "/home/oliverfl/projects/gpcr/simulations/strings/jan/5cvs-drorpath/gpcr/.string_simu/1/allswarms.xtc",
        #                      iterations=[1])
        # merge_restrained_to_file(args,
        #                          "/home/oliverfl/projects/gpcr/simulations/strings/jan/5cvs-drorpath/gpcr/.string_simu/1/allrestrained.xtc",
        #                          iterations=[1])
        # compare_5cvs_pbc_condition(args)
        logger.debug("Done")

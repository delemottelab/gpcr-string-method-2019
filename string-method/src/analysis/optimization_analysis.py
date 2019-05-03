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
from analysis.extra_analysis import *

logger = logging.getLogger("OptimizationAnalysis")

logger.info(
    """
    Note
    THE CODE IN THIS FILE IS DELIBERATELY NOT MAINTAINED AND REFACTORED THAT MUCH
    It is intended to be used for ad hoc analysis
    """
)


def analyze_simulation_runtime(args, start_iteration=2, end_iteration=999, point_filetype="gro", swarm_filetype="xtc",
                               swarm_time_ps=10, ignore_points_without_swarms=True):
    runner = rs.StringIterationRunner(args)
    point_counts = []
    swarm_counts = []
    simu_to_restrained_time = {
        "holo-optimized": 20,
        "apo-optimized": 20,
        "holo-curved": 30,
        "apo-curved": 30,
        "holo-straight": 30,
        "looseep-holo-straight": 50
    }
    restrained_time_ps = simu_to_restrained_time[runner.simu_id]
    for iteration in range(start_iteration, end_iteration):
        try:
            runner.init_iteration(iteration)
        except IOError as ex:
            logger.exception(ex)
            break
        startpoint = 1 if runner.fixed_endpoints else 0
        endpoint = len(runner.stringpath) - 1 if runner.fixed_endpoints else len(runner.stringpath)
        point_count = 0
        swarm_count = 0
        for point_idx in range(startpoint, endpoint):
            if runner.fixed_endpoints and (point_idx == 0 or point_idx == len(runner.stringpath) - 1):
                continue
            point_dir = runner.point_path(point_idx)
            point_path = point_dir + runner.point_name(point_idx) + "-restrained." + point_filetype
            if exists(point_path):
                point_count += 1
            swarm_path = point_dir + runner.swarm_name(point_idx, 999, 999).replace("999", "*") + swarm_filetype
            swarm_count += len(glob.glob(swarm_path))
        if not ignore_points_without_swarms or swarm_count > 0:
            point_counts.append(point_count)
            swarm_counts.append(swarm_count)

    point_counts = np.array(point_counts)
    swarm_counts = np.array(swarm_counts)
    npoints = sum(point_counts)
    nswarms = sum(swarm_counts)
    total_time_ns = (restrained_time_ps * npoints + nswarms * swarm_time_ps) / 1000
    logger.info("Total number of restrained points: %s. Total number of swarms: %s. Time in ns: %s", npoints, nswarms,
                total_time_ns)
    plt.subplot(221)
    plt.ylabel("# points", fontsize=utils.label_fontsize)
    plt.xlabel("Iteration #", fontsize=utils.label_fontsize)
    plt.plot(point_counts)
    # plt.grid()
    # plt.show()
    plt.subplot(222)
    plt.ylabel("# trajectories", fontsize=utils.label_fontsize)
    plt.xlabel("Iteration #", fontsize=utils.label_fontsize)
    # plt.grid()
    plt.plot(swarm_counts)  # TODO boxplots with deviations etc.
    # plt.show()
    plt.subplot(223)
    plt.ylabel("Mean swarm size", fontsize=utils.label_fontsize)
    plt.xlabel("Iteration #", fontsize=utils.label_fontsize)
    plt.plot(swarm_counts / point_counts)
    # plt.grid()
    # plt.title("Total simulation time for simulation {}: {} ns.".format(runner.simu_id, total_time_ns))
    plt.show()


def reparametrize_string(runner, cvs, equidistant_points, already_finished=False):
    old_cvs = runner.cvs
    old_equidistant_points = runner.equidistant_points
    runner.cvs = cvs
    runner.equidistant_points = equidistant_points
    if not already_finished:
        runner.swarmProcessor = SwarmProcessor(runner)
        runner.job_executor = FinishedJobExecutor("FinishedJobExecutor", callbacks=[runner.swarmProcessor])
        runner.run()
    processor = SingleIterationPostProcessor(runner, save=False, plot=False)
    processor.run()
    runner.cvs = old_cvs
    runner.equidistant_points = old_equidistant_points
    return processor.new_stringpath


def reparametrize_string_contacts_distance(args, iteration, twoD=True, cv_indices=[0, 1], plot=True, exponent_power=2):
    logger.info("reparametrize_string_contacts_distance")
    runner = rs.StringIterationRunner(args)
    runner.init_iteration(iteration)
    # runner.stringpath = runner.stringpath[:,cv_indices]
    # runner.cvs = np.array(runner.cvs)[cv_indices]
    distance_cvs = runner.cvs
    contact_cvs = create_cvs.create_inverted_cvs(distance_cvs, power=exponent_power)
    # First reparametrize with existing CVs
    distance_stringpath = runner.stringpath
    contact_stringpath = colvars.scale_points(contact_cvs, 1 / colvars.rescale_points(distance_cvs,
                                                                                      runner.stringpath) ** exponent_power)
    ###COMPUTE STRINGS
    # Contacts
    runner.stringpath = contact_stringpath
    contact_string_weighted = reparametrize_string(runner, contact_cvs, False, already_finished=False)
    contact_string_unweighted = reparametrize_string(runner, contact_cvs, True, already_finished=True)
    axis_labels = [get_cv_description(cv.id, use_simpler_names=True) for cv in distance_cvs]
    distcontact_string_unweighted = 1 / colvars.rescale_points(contact_cvs, contact_string_unweighted) ** (
            1 / exponent_power)
    distcontact_string_weighted = 1 / colvars.rescale_points(contact_cvs, contact_string_weighted) ** (
            1 / exponent_power)
    # DISTANCES
    runner.stringpath = distance_stringpath
    distance_string_weighted = reparametrize_string(runner, distance_cvs, False, already_finished=False)
    distance_string_weighted = colvars.rescale_points(distance_cvs, distance_string_weighted)
    distance_string_unweighted = reparametrize_string(runner, distance_cvs, True, already_finished=True)
    distance_string_unweighted = colvars.rescale_points(distance_cvs, distance_string_unweighted)
    ## Compute difference:
    dist_weighted = np.linalg.norm(distcontact_string_weighted - distance_string_weighted)
    dist_unweighted = np.linalg.norm(distcontact_string_unweighted - distance_string_unweighted)
    logger.info("Distance between weighted %s for iteration %s", dist_weighted, iteration)
    logger.info("Distance between unweighted %s for iteration %s", dist_unweighted, iteration)
    ###PLOTTING
    if plot:
        ncols = 1 if len(cv_indices) == 2 and twoD else 2
        utils.plot_path(distcontact_string_unweighted[:, cv_indices],
                        label="(Distance^-{})Not equidistant points".format(exponent_power), ncols=ncols, twoD=twoD,
                        axis_labels=axis_labels)
        utils.plot_path(distance_string_weighted[:, cv_indices],
                        label="(Distance)Not equidistant points", ncols=ncols, twoD=twoD,
                        axis_labels=axis_labels)
        utils.plot_path(distcontact_string_weighted[:, cv_indices],
                        label="(Distance^-{})Equidistant points".format(exponent_power), ncols=ncols, twoD=twoD,
                        axis_labels=axis_labels)
        utils.plot_path(distance_string_unweighted[:, cv_indices],
                        label="(Distance)Equidistant points", ncols=ncols, twoD=twoD,
                        axis_labels=axis_labels)
        plt.title("Simulation {} iteration {}".format(runner.simu_id, iteration))
        plt.show()
    return dist_weighted, dist_unweighted


def compute_diff_contact_distance_reparametrization(args, start_iteration, last_iteration, iteration_step=1,
                                                    exponent_power=2):
    logger.info("compute_diff_contact_distance_reparametrization")

    weighted_diffs, unweighted_diffs = [], []
    iterations = []
    for iteration in range(start_iteration, last_iteration + 1, iteration_step):
        dist_weighted, dist_unweighted = reparametrize_string_contacts_distance(args, iteration, plot=False,
                                                                                exponent_power=exponent_power)
        weighted_diffs.append(dist_weighted)
        unweighted_diffs.append(dist_unweighted)
        iterations.append(iteration)
    weighted_diffs = np.array(weighted_diffs)
    unweighted_diffs = np.array(unweighted_diffs)
    iterations = np.array(iterations)
    axis_labels = ["Iteration", "Difference"]
    utils.plot_path(weighted_diffs,
                    label="Not equidistant points", ncols=1, twoD=False,
                    xticks=iterations, ticks=iterations,
                    axis_labels=axis_labels)
    utils.plot_path(unweighted_diffs,
                    label="Equidistant points", ncols=1, twoD=False,
                    xticks=iterations, ticks=iterations,
                    axis_labels=axis_labels)
    plt.title(args.simu_id + " power=" + str(exponent_power))
    plt.show()
    return weighted_diffs, unweighted_diffs


def compute_swarm_convergence_fraction(args, iteration, swarm_batch_size=None, plot_frequency=1,
                                       compare_to_final=False):
    """Compute the swarm convergence after adding the last 'swarm_batch_size' swarms to the average drift vector"""
    logger.info("compute_swarm_convergence_fraction")
    runner = rs.StringIterationRunner(args)
    if iteration is not None:
        runner.init_iteration(iteration)
    runner.compute_number_of_swarms()
    if swarm_batch_size is None:
        swarm_batch_size = runner.swarm_batch_size
    swarm_processor = SwarmProcessor(runner, ignore_missing_files=False)
    runner.job_executor = FinishedJobExecutor("FinishedJobExecutor", callbacks=[swarm_processor])
    runner.run()
    logger.info("Last swarm_convergence_fraction %s ", swarm_processor.last_swarm_convergence_fraction)
    logger.info("Average swarm_convergence_fraction %s ", swarm_processor.last_swarm_convergence_fraction.mean())
    convergence_fractions = []
    for point_idx in range(swarm_processor.string_length):
        convergences = []
        final_drifts = swarm_processor.compute_swarm_drifts(point_idx)
        final_swarm_coordinates = swarm_processor.swarm_coordinates[point_idx]
        final_avg_drift = swarm_processor.compute_average_drift(point_idx)
        previous_avg_drift = None
        for traj_idx in range(len(final_drifts)):
            if traj_idx == 0 or (traj_idx + 1) % swarm_batch_size != 0:
                continue
            new_swarm_coordinates = final_swarm_coordinates[0:(traj_idx + 1)]
            swarm_processor.swarm_coordinates[point_idx] = new_swarm_coordinates
            new_avg_drift = swarm_processor.compute_average_drift(point_idx)
            if not compare_to_final and previous_avg_drift is not None:
                swarm_convergence_fraction = np.linalg.norm(new_avg_drift - previous_avg_drift) / np.linalg.norm(
                    new_avg_drift)
                convergences.append(swarm_convergence_fraction)
            elif compare_to_final:
                swarm_convergence_fraction = np.linalg.norm(new_avg_drift - final_avg_drift) / np.linalg.norm(
                    final_avg_drift)
                convergences.append(swarm_convergence_fraction)
            previous_avg_drift = new_avg_drift
        # logger.info("swarm_convergence_fraction for last %s swarms for point %s: %s", swarm_batch_size, point_idx, swarm_convergence_fraction)
        # reset values
        swarm_processor.swarm_coordinates[point_idx] = final_swarm_coordinates
        convergence_fractions.append(np.array(convergences))
    final_convergences = []
    for conv in convergence_fractions:
        final_convergences.append(conv[-1] if len(conv) > 0 else np.nan)
    final_convergences = np.array(final_convergences)
    logger.info("Mean and median final convergence %s, %s", np.mean(final_convergences), np.median(final_convergences))
    logger.info("Min, max final convergence %s, %s", final_convergences.min(), final_convergences.max())
    swarm_count = sum([len(conv) for conv in convergence_fractions])
    logger.info("Final swarm count %s", swarm_count)
    xticks = None
    for idx in range(1, len(convergence_fractions), plot_frequency):
        convergences = convergence_fractions[idx]
        ticks = [(ii + 2) * swarm_batch_size for ii in range(len(convergences))]
        if xticks is None or len(ticks) > len(xticks):
            xticks = ticks
        utils.plot_path(convergences, label="point {}".format(idx), ncols=1, twoD=False,
                        axis_labels=["Swarm count",
                                     "Convergence {}".format("final" if compare_to_final else " previous")],
                        ticks=ticks,
                        xticks=xticks)
    plt.title(runner.simu_id + "-iter{}".format(runner.iteration, compare_to_final))
    plt.show()
    # print(convergence_fractions)


if __name__ == "__main__":
    # parser = rs.create_argparser()
    # args = parser.parse_args()
    simulations = [
        "holo-optimized",
        "apo-optimized",
        # "straight-holo-optimized",
        # "endpoints-holo",
        # "to3sn6-holo",
        # "holo-curved",
        # "apo-curved",
        # "holo-straight",
        # "beta1-apo"
    ]
    if args.start_mode != 'analysis':
        logger.error("Startmode %s not supported", args.start_mode)
    else:
        analyze_simulation_runtime(args)
        # compute_swarm_convergence_fraction(args, iteration=3, plot_frequency=4, swarm_batch_size=1,
        #                                    compare_to_final=True)
        # reparametrize_string_contacts_distance(args, iteration=4, cv_indices=[0, 1], exponent_power=3)
        # compute_diff_contact_distance_reparametrization(args, start_iteration=4, last_iteration=35, iteration_step=5,
        #                                                 exponent_power=1)
        logger.debug("Done")

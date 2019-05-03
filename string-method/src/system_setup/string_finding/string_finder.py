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
from utils.reparametrize_string import *
from system_setup.string_finding.density_field import *
from system_setup.string_finding.pnas_simulation_loader import *
import numpy as np

logger = logging.getLogger("stringfinder")

logger.info("Use this file at your own risk. It is in the need of refactoring")


def gradientNd(S, x, h=1e-5):
    """Approximates gradient of S at point x with a central difference method
    #TODO use np.gradient instead
    """
    dx = np.empty(x.shape, dtype=float)
    xh1 = np.copy(x)
    xh2 = np.copy(x)
    ncvs = len(x) if len(x.shape) == 1 else x.shape[1]
    scale = 1 / (2 * h)
    for i in range(0, ncvs):
        xh1[:, i] -= h
        xh2[:, i] += h
        dx[:, i] = scale * (S(xh2.T) - S(xh1.T))
        xh1[:, i] += h
        xh2[:, i] -= h
    return dx


def find_closest_frame(point, cvs, simulation_evals):
    frame = None
    for simulation, cv_values in simulation_evals:
        closest_distance = 1e10
        for t in range(0, len(simulation.traj)):
            dist = np.linalg.norm(point - cv_values[:, t])
            if dist < closest_distance:
                # logger.debug("Found frame in %s at time %s", simulation.id, t)
                frame = simulation.traj[t]
                closest_distance = dist
    return frame


def find_closest_frames(path, cvs, simulation_evals):
    """Find representative frame for the path CV coordinates"""
    closest_frames = []
    for point in path:
        # TODO could optimze algorithm with a search tree or something
        closest_frames.append(find_closest_frame(point, cvs, simulation_evals))
    return closest_frames


def get_initial_path(start, end, number_points=25, S=None):
    logger.warn("Deprecated use utils function for straight path instead")
    dim = len(start)
    pts = np.empty((number_points, dim if S is None else dim + 1))
    for i in range(0, dim):
        pts[:, i] = np.linspace(start[i], end[i], num=number_points)
    if S is not None:
        for row in range(0, number_points):
            pts[row, dim] = S(pts[row, 0:dim])
    return pts


def find_physical_pathway(S,
                          path,
                          iterations,
                          driftsize=1e-7,  # TODO adaptive driftsize
                          fixed_endpoints=True):
    previous_path = path
    h = 1e-5
    new_path = path
    for n in range(0, iterations):
        gradient = gradientNd(S, previous_path, h=h)
        new_path = previous_path - gradient * driftsize
        if fixed_endpoints:
            new_path[[0, -1], :] = previous_path[[0, -1], :]
        # for i, p in enumerate(previous_path):
        #     if fixed_endpoints and (i == 0 or i == (len(previous_path) - 1)):
        #         new_path[i] = previous_path[i]
        #         continue
        #     # gradient = -100*np.array([(0.2-p[0]), (0.2-p[1])]) #test gradient
        #     gradient = gradientNd(S, p, h=h)
        #     # gradient = gradientfunc(start)[0]
        #     # 2 Let them drift along the field's gradient
        #     pxy = p - gradient * driftsize
        #     new_path[i] = pxy
        # # Use new points as input for next iteration
        new_path = reparametrize_path_iter(new_path)
        previous_path = new_path
    return new_path


def compute_physical_pathway(start,
                             end,
                             field,
                             cvs,
                             plot_intermediates=False,
                             number_points=50,
                             fixed_endpoints=True,
                             driftsize=1e-7,
                             convergence_limit=1e-5):
    logger.info(
        "Computing physical pathway from %s to %s. Params: number_points=%s, fixed_endpoints=%s, driftsize=%s",
        start, end, number_points, fixed_endpoints, driftsize)
    S = field
    initial_path = utils.get_straight_path(start, end,
                                           number_points=number_points)  # get_initial_path(start, end, number_points=number_points)
    last_path = initial_path
    for i in range(1, 20):
        iterations = 10  # 500
        final_path = find_physical_pathway(
            S,
            last_path,
            iterations,
            driftsize=driftsize,
            fixed_endpoints=fixed_endpoints)
        convergence = np.linalg.norm(final_path - last_path)
        L = compute_path_length(final_path)
        logger.debug("Convergence: %s after %s iterations. Path length=%s",
                     convergence, (i * iterations), L)
        if convergence < convergence_limit:
            logger.info("String Converged!")
            break
        if plot_intermediates:
            for dimi in range(0, final_path.shape[1]):
                for dimj in range(dimi + 1, final_path.shape[1]):
                    plt.plot(final_path[::, dimi], final_path[::, dimj], label="dim %s-%s" % (dimi, dimj))
                    plt.scatter(final_path[::, dimi], final_path[::, dimj])
            plt.legend()
            plt.show()
        last_path = final_path

    logger.info("Final string coordinates:\n %s", final_path)
    return final_path


def load_rmsd_start_end_frames(start_rmsd, end_rmsd, rmsd_cvs, simulations):
    """Find closest frames for start and end points in RMSD space"""
    logger.debug("Finding start and end frames ")
    values = evaluate_simulations(simulations, rmsd_cvs)
    start_frame = find_closest_frame(start_rmsd, rmsd_cvs, values)
    end_frame = find_closest_frame(end_rmsd, rmsd_cvs, values)
    return start_frame, end_frame


if __name__ == "__main__":
    logger.info("Started")
    traj_type = "apo"  # drorD, drorA
    endpoint_method = "max_min"
    if traj_type == "drorD":
        cvs_dir = cvs_len5path
        cvs = colvars.cvs_definition_reader.load_cvs(cvs_dir + "cvs.json")
        simulation_conditions = [("D", "05"), ("D", "09")]
        all_simulations = load_simulations(simulation_conditions, stride=1,
                                           simulation_dir="/home/oliverfl/projects/gpcr/simulations/dror-anton-2011/")
    elif traj_type == "apo":
        cvs_dir = "/home/oliverfl/projects/gpcr/cvs/"
        cvs = colvars.cvs_definition_reader.load_cvs(cvs_dir + "cvs-freemd_apo_3_clusters-len29.json")
        all_simulations = load_freemd("/home/oliverfl/projects/gpcr/simulations/freemd/3p0g-noligand-charmmgui/",
                                      "allTrr.xtc", "step6.6_equilibration.gro")
    logger.info("Done. Using %s simulations with %s frames", len(all_simulations),
                sum(len(s.traj) for s in all_simulations))
    # Evaulate CVs and create free energy field
    simulation_evals = evaluate_simulations(all_simulations, cvs)
    cv_coordinates = get_cv_coordinates(simulation_evals, cvs)
    density = compute_density_field(simulation_evals, cvs, cv_coordinates)
    frame_count = sum(len(s.traj) for s in all_simulations)
    free_energy = to_free_energy(density, norm=frame_count)

    # Set endpoints
    if endpoint_method == "rmsd":
        rmsd_cvs = load_object("../../../gpcr/cvs/rmsd-cvs/cvs")
        start_frame, end_frame = load_rmsd_start_end_frames(
            np.array([0.65, 1.15]), np.array([1.15, 0.65]), rmsd_cvs, all_simulations)
        startpoint = np.array([cv.eval(start_frame)[0] for cv in cvs])
        endpoint = np.array([cv.eval(end_frame)[0] for cv in cvs])
    elif endpoint_method == "structures":
        ## From equilibrated structures
        active_struct = md.load("../../../gpcr/reference_structures/old/3p0g-ligand/equilibrated.gro")
        inactive_struct = md.load("../../../gpcr/reference_structures/2rh1-noligand/equilibrated.gro")
        startpoint = colvars.eval_cvs(active_struct, cvs)[0]
        endpoint = colvars.eval_cvs(inactive_struct, cvs)[0]
    elif endpoint_method == "max_min":
        startpoint = np.empty((len(cvs),))
        endpoint = np.empty((len(cvs),))
        for i, cv in enumerate(cvs):
            startpoint[i] = cv_coordinates[:, i].max()
            endpoint[i] = cv_coordinates[:, i].min()
    else:
        raise Exception("Invalid method")

    # Find string
    physical_path = compute_physical_pathway(
        startpoint,
        endpoint,
        free_energy,
        cvs,
        plot_intermediates=False,
        fixed_endpoints=True,
        driftsize=1e-3,
        number_points=20,
        convergence_limit=1e-3)
    np.savetxt(cvs_dir + "path-{}-{}.txt".format(traj_type, endpoint_method), physical_path)
    # Visualize it in lower dimensions
    cv_indices = [0, 1]
    cvs = np.array(cvs)[cv_indices]
    simulation_evals = evaluate_simulations(all_simulations, cvs)
    cv_coordinates = get_cv_coordinates(simulation_evals, cvs)
    density = compute_density_field(simulation_evals, cvs, cv_coordinates)
    free_energy = to_free_energy(density, norm=frame_count)
    plot_field(
        cvs,
        cv_coordinates,
        free_energy,
        ngrid=40,
        heatmap=True,
        scatter=False)
    utils.plot_path(physical_path[:, cv_indices], label=None, axis_labels=None, scatter=False)
    plt.show()
    logger.info("Done")

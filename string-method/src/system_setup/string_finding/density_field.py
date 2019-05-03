from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from analysis.extra_analysis import get_cv_description, colvars
from system_setup import create_cvs
from system_setup.create_stringpaths import cvs_len5path
from system_setup.string_finding.pnas_simulation_loader import *
from utils.helpfunc import *

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import utils

logger = logging.getLogger("density_field")


def plot_2d_field(xvalues,
                  yvalues,
                  Vij,
                  cvx,
                  cvy,
                  ngrid,
                  cmap=plt.cm.Greys,
                  heatmap=True,
                  scatter=False):
    if heatmap:
        xmin = np.amin(xvalues)
        xmax = np.amax(xvalues)
        ymin = np.amin(yvalues)
        ymax = np.amax(yvalues)

        X, Y = np.meshgrid(np.linspace(xmin, xmax, ngrid), np.linspace(ymin, ymax, ngrid))
        Z = Vij  # np.reshape(Vij, X.shape)
        Z = Z - Z.min()
        # Z[Z>7] = np.nan
        # plt.figure(figsize=(10,8))
        # Zrange = Z.max() - Z.min()
        # levels = np.arange(Z.min(), Z.min() + Zrange / 2, Zrange / ngrid)
        im = plt.contourf(
            X.T,
            Y.T,
            Z,
            #             np.rot90(Z), #FOR plt.imshow()
            #             levels=levels,
            cmap=cmap,
            extent=[xmin, xmax, ymin, ymax])
        ct = plt.contour(
            X.T,
            Y.T,
            Z,
            #             levels=levels,
            extent=[xmin, xmax, ymin, ymax],
            alpha=0.3,
            colors=('k',))
        # im.cmap.set_under('k')
        # im.set_clim(0, Z.max())
        cbar = plt.colorbar(im, orientation='vertical')
        cbar.set_label(r'$\Delta G$ [kcal/mol]', fontsize=utils.label_fontsize)
        cbar.ax.tick_params(labelsize=utils.ticks_labelsize)
        # plt.ylabel(cvy.id)
        # plt.xlabel(cvx.id)
        plt.grid()
        # plt.show()
    if scatter:
        # gaussian normalized by total number of points
        xy = np.vstack([xvalues, yvalues])
        colors = Vij(xy)
        im = plt.scatter(xvalues, yvalues, c=colors, cmap=cmap)
        plt.colorbar(im, orientation='vertical')
        plt.ylabel(cvy.id)
        plt.xlabel(cvx.id)
        plt.show()


def to_free_energy(density, norm=1, delta=1e-7):
    """ConvertsTODO move to separate module"""
    return lambda x: -kb * 310.15 * np.log(density(x) / norm + delta)


def get_cv_coordinates(simulation_evals, cvs):
    """Put all CV values into a matrix with len(cvs) rows and total number of simulation frames as columns"""
    frame_count = 0
    for simulation, cv_values in simulation_evals:
        frame_count += len(simulation.traj)
    cv_coordinates = np.empty((len(cvs), frame_count))
    logger.debug("Aggregating all simulations")
    for i, cv in enumerate(cvs):
        frames_offset = 0
        for simulation, cv_values in simulation_evals:
            val = cv_values[i]
            traj_size = len(val)
            cv_coordinates[i, frames_offset:(frames_offset + traj_size)] = val
            frames_offset += traj_size
    return cv_coordinates


def integrate_cv(V, cv_idx, width=1):
    return np.sum(V, axis=cv_idx) * width


def integrate_other_cv(V, cv_idx, ncvs):
    for i in reversed(range(ncvs)):
        if i != cv_idx:
            V = integrate_cv(V, cv_idx=i)
    return V


def compute_density_field(simulation_evals, cvs, cv_coordinates):
    stacked = np.vstack(cv_coordinates)
    logger.debug("Computing gaussian field")
    return stats.gaussian_kde(stacked)


def compute_energy_grid(cvs, field, ngrid, cv_coordinates):
    maxvals, minvals = cv_coordinates.max(axis=1), cv_coordinates.min(axis=1)
    gridsizes = (maxvals - minvals) / ngrid
    ncvs = len(cvs)
    V = np.zeros(tuple(np.zeros((ncvs,)).astype(int) + ngrid))
    count = 0
    total = ngrid ** (ncvs)
    modulus = [ngrid ** (ncvs - j - 1) for j in range(ncvs)]
    while count < total:
        idx = ((np.zeros((ncvs,)) + count) / modulus) % ngrid
        idx = idx.astype(int)
        # print(idx)
        if V[tuple(idx)] != 0.0:
            logger.error("Index %s visited twice for matrix of size %s", idx,
                         V.shape)
            raise Exception("Failed at count " + str(count))
        V[tuple(idx)] = field(minvals + idx * gridsizes)
        count += 1
        if 10 * count % total == 0:
            logger.debug(
                "Progress %s percent. Computed potential for %s gridpoints out of %s",
                int(100 * (count / total)), count, total)
    logger.info("Done computing field on an %s grid in %s dimension", ngrid,
                ncvs)
    return V


def plot_field(cvs,
               cv_coordinates,
               field,
               ngrid=10,
               heatmap=True,
               scatter=True):
    """
    inspired by https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    and https://stackoverflow.com/questions/24119920/how-to-plot-a-field-map-in-python
    Using two coordinates at the time. The other coordinates are fixed at their average value
    """
    # average_positions = get_average_positions(cv_coordinates)
    # logger.info("Average coordinate positions %s", average_positions)
    #     logger.info("Fixing to average values for CVs not in plots to: %s",
    #                 average_positions)

    #     def field2d(positions2d):
    #         positions = np.empty((len(cvs), positions2d.shape[1]))
    #         positions[i, :] = positions2d[0, :]
    #         positions[j, :] = positions2d[1, :]
    #         for idx in range(0, len(cvs)):
    #             if idx not in [i, j]:
    #                 positions[idx, :] = average_positions[idx]
    #         return field(positions)
    # TODO integrate out the degrees of freedom instead of using average
    # Something like: V01 = np.sum(V, axis=2)*(max_cvs[2]-min_cvs[2])
    # Comute field along a high dimensional grid:
    V = compute_energy_grid(cvs, field, ngrid, cv_coordinates)
    for i, cvx in enumerate(cvs):
        xvalues = cv_coordinates[i, :]
        for j in range(i + 1, len(cvs)):
            yvalues = cv_coordinates[j, :]
            cvy = cvs[j]
            Vij = V
            for k in reversed(range(len(cvs))):
                if k == i or k == j:
                    continue
                Vij = integrate_cv(Vij, cv_idx=k)
            plot_2d_field(
                xvalues,
                yvalues,
                Vij,
                cvx,
                cvy,
                ngrid,
                heatmap=heatmap,
                scatter=scatter)


if __name__ == "__main__":
    logger.info("Started")
    ####SETUP##########
    # protein, traj_type = "beta1", "equilibration"  # "after_targetedmd", "equilibration", "targetedmd"
    protein, traj_type = "beta2", "apo"
    cvs = colvars.cvs_definition_reader.load_cvs(cvs_len5path + "cvs.json")
    stringpaths = \
        [("Fixed EP", np.loadtxt(cvs_len5path + "string-paths-drorpath/dror_path_fixedep.txt"))] + \
        [("Loose EP", np.loadtxt(cvs_len5path + "string-paths-drorpath/dror_path_looseep.txt"))] + \
        [("Equilibrated EP", np.loadtxt(cvs_len5path + "string-paths-drorpath/dror_path_equilibratedep.txt"))] + \
        [("Holo Final", colvars.scale_points(cvs, np.loadtxt(
            "/home/oliverfl/projects/gpcr/simulations/strings/april/holo5-drorpath/gpcr/cvs/cvs-len5_good/string-paths/stringaverage150-200.txt")))] + \
        [("Apo Final", np.loadtxt(
            "/home/oliverfl/projects/gpcr/simulations/strings/april/apo5-drorpath/gpcr/cvs/cvs-len5_good/string-paths/stringaverage200-281.txt"))] + \
        [("Holo-straight Final", np.loadtxt(
            "/home/oliverfl/projects/gpcr/simulations/strings/jun/holo5-optimized-straight/gpcr/cvs/cvs-len5_good/string-paths/stringaverage300-400.txt"))]
    if protein == "beta2":
        cv_indices = [0, 1]
        cvs = cvs[cv_indices]
        if traj_type == "pnas":
            simulation_conditions = [("D", "05")]  # [("D", "05"), ("D", "09")]  # , [("A", "00")]
            simulation = load_simulations(simulation_conditions, stride=1,
                                          simulation_dir="/home/oliverfl/projects/gpcr/simulations/dror-anton-2011/")
        elif traj_type == "apo":
            cv_indices=[0,1]
            stringpaths = [("apo-new", np.loadtxt("/home/oliverfl/projects/gpcr/cvs/path-apo-max_min.txt"))]
            cvs = colvars.cvs_definition_reader.load_cvs("/home/oliverfl/projects/gpcr/cvs/cvs-freemd_apo_3_clusters-len29.json")[[0,1]]
            simulation = load_freemd("/home/oliverfl/projects/gpcr/simulations/freemd/3p0g-noligand-charmmgui/",
                                     # "allTrr.xtc", "step6.6_equilibration.gro")
                                    "freemd_notwater_notlipid.dcd", "freemd_notwater_notlipid.pdb")
        elif traj_type == "equilibration-production":
            simulation = []
            ##### HOLO 3p0g
            simulation += \
                load_freemd(
                    "/data/oliver/pnas2011b-Dror-gpcr/equilibration/aug/fixed_loop/asp79-holo-protonated-ligand/production/",
                    "3p0g-protonated-ligand-full.xtc",
                    "3p0g-protonated-ligand.gro") + \
                load_freemd(
                    "/data/oliver/pnas2011b-Dror-gpcr/equilibration/aug/fixed_loop/asp79-holo-deprotonated-ligand/charmm-gui/gromacs/",
                    "step6.6_equilibration.trr",
                    "step6.6_equilibration.gro") + \
                load_freemd(
                    "/data/oliver/pnas2011b-Dror-gpcr/equilibration/aug/fixed_loop/asp79-holo-deprotonated-ligand/charmm-gui/gromacs/",
                    "step7_1and2.xtc",
                    "step6.6_equilibration.gro") + \
                load_freemd("/data/oliver/pnas2011b-Dror-gpcr/freemd/3p0g-inactive-endpoint/",
                            "freemd_notwater_notlipid.dcd",
                            "freemd_notwater_notlipid.pdb")  # + \
            #### APO 2rh1
            simulation += \
                load_freemd(
                    "/data/oliver/pnas2011b-Dror-gpcr/equilibration/noligand-jan24/2rh1-charmm-gui/gromacs/",
                    "step7.xtc",
                    "step6.6_equilibration.gro"
                )
            # APO 3p0g
            simulation += \
                load_freemd(
                    "/data/oliver/pnas2011b-Dror-gpcr/equilibration/aug/fixed_loop/apo/charmm-gui/gromacs/production/",
                    "3p0g-apo-full.xtc",
                    "3p0g-apo.gro") + \
                load_freemd(
                    "/data/oliver/pnas2011b-Dror-gpcr/equilibration/aug/fixed_loop/apo/charmm-gui/gromacs/",
                    "step7_1and2.xtc",
                    "step6.6_equilibration.gro") + \
                load_freemd(
                    "/data/oliver/pnas2011b-Dror-gpcr/equilibration/aug/fixed_loop/apo/charmm-gui/gromacs/",
                    "step6.6_equilibration.trr",
                    "step6.6_equilibration.gro")
    elif protein == "beta1":
        cv_indices = [0, 1]
        #cvs = np.array(create_cvs.create_beta1_5cvs(beta2_cvs_dir=cvs_len5path))[cv_indices]
        if traj_type == "equilibration":
            simulation = load_freemd("/data/oliver/pnas2011b-Dror-gpcr/equilibration/4gpo-beta1-charmm-gui/gromacs/",
                                     "step7_1to3_nowater_nolipid.dcd", "step7_1to3_nowater_nolipid.pdb")
        elif traj_type == "after_targetedmd":
            simulation = load_freemd("/data/oliver/pnas2011b-Dror-gpcr/targetedmd/2018/targetedmd-beta1/", "freemd.xtc",
                                     "freemd.gro")
        elif traj_type == "targetedmd":
            simulation = load_freemd("/data/oliver/pnas2011b-Dror-gpcr/targetedmd/2018/targetedmd-beta1/",
                                     "targetedmd-restrained.xtc",
                                     "targetedmd-restrained.gro")
    logger.info("Done. Using %s simulations with %s frames",
                len(simulation), sum(len(s.traj) for s in simulation))
    ###START EVALS#########
    simulation_evals = evaluate_simulations(simulation, cvs)
    cv_coordinates = get_cv_coordinates(simulation_evals, cvs)
    density = compute_density_field(simulation_evals, cvs, cv_coordinates)
    frame_count = sum(len(s.traj) for s in simulation)
    free_energy = to_free_energy(density, norm=frame_count)
    plot_field(
        cvs,
        cv_coordinates,
        free_energy,
        ngrid=40,
        heatmap=True,
        scatter=False)
    # plt.title("{}-{}, CVs{}".format(protein, traj_type, cv_indices))
    axis_labels = np.array([get_cv_description(cv.id, use_simpler_names=True) for cv in cvs])
    # plt.scatter(cv_coordinates[0, :], cv_coordinates[1, :], alpha=0.05, label="Simulation frames", color="orange")
    for (label, sp) in stringpaths:
        utils.plot_path(sp[:, cv_indices], label=label, axis_labels=axis_labels, scatter=False)
    # plt.text(cv_coordinates[0, 0], cv_coordinates[1, 0], "START", color="blue")
    # plt.text(cv_coordinates[0, -1], cv_coordinates[1, -1], "END", color="blue")
    plt.show()
    logger.info("Done")

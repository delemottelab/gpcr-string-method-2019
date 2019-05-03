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

import colvars
import utils
from utils.helpfunc import *

logger = logging.getLogger("InitialPoints")
cvs_len5path = utils.project_dir + "gpcr/cvs/len5/asp79/"
cvs_len5path = utils.project_dir + "gpcr/cvs/len5/asp79/"


def create_straight_path_between_structures(cvs, start_traj, end_traj, number_points, savepath=None, plot=True):
    logger.info("Creating straight path with %s points for cvs %", number_points, [cv.id for cv in cvs])
    start_evals = colvars.eval_cvs(start_traj, cvs).squeeze()
    end_evals = colvars.eval_cvs(end_traj, cvs).squeeze()
    stringpath = utils.get_straight_path(start_evals, end_evals, number_points=number_points)
    if savepath is not None:
        np.savetxt(savepath, stringpath)
        logger.info("Saved stringpath of %s points to %s", len(stringpath), savepath)
    if plot:
        utils.plot_path(stringpath, twoD=False)
        plt.title("Generated stringpath")
        plt.show()
    return stringpath


def create_targetmd_input(cvs, stringpath, steps_per_point=10000, kappa=10000.0, backwards=False):
    """
    Units info https://plumed.github.io/doc-v2.4/user-doc/html/_u_n_i_t_s.html
    default kj/mol for energy, A for distance.
    Spring constant has unit Energy units per Units of the CV
    :param args:
    :param steps_per_point:
    :param offset:
    :param kappa:
    :param backwards: useful when you equilibrate in the opposite direction of the string
    :return:
    """
    plumed_cvs = ",".join(["cv" + str(idx) for idx in range(len(cvs))])
    kappas = ",".join([str(kappa) for i in range(len(cvs))])
    restraint = "restraint: ...\n\tMOVINGRESTRAINT\n\tARG=" + plumed_cvs + "\n"
    # AT0=0.0988892250591,0.123611531324 STEP0=0     KAPPA0=10000,10000
    template = "\tAT%s=%s STEP%s=%s KAPPA%s=" + kappas + "\n"
    if backwards:
        stringpath = np.flip(stringpath, 0)
        logger.debug("Flipped stringpath to shape %s", stringpath.shape)
    for i, point in enumerate(stringpath):
        rescaled_point = colvars.rescale_points(cvs, point)  # actual non-normalized distance
        restraint += template % (i, ",".join([str(p) for p in rescaled_point]), i, steps_per_point * i, i)
    restraint += "..."
    logger.info("For targeted MD, add this to your plumed file:\n\n%s\n", restraint)
    return restraint


def create_3po0_3sn6_string():
    cvs = load_object(cvs_len5path + "cvs")
    holo_3p0g = md.load("../gpcr/reference_structures/3p0g-ligand/equilibrated.gro")
    holo_3sn6 = md.load("../gpcr/reference_structures/3sn6-ligand/equilibrated.gro")
    active_straight_path = create_straight_path_between_structures(cvs, holo_3p0g, holo_3sn6, 4,
                                                                   savepath=cvs_len5path + "string-paths-3p0g-3sn6/straight4points.txt")


def _create_ash79_initial_string(savepath=None, plot=True):
    cvs = load_object(cvs_len5path + "cvs")
    asp79_path = np.loadtxt("/home/oliver/slask/holo5-optimized-avg-strings/stringaverage150-175.txt")
    rescale = True
    asp79_path = colvars.scale_evals(asp79_path, cvs) if rescale else asp79_path
    ash79_path = asp79_path[::2]
    ash79_path[-1] = asp79_path[-1]
    logger.info("Changed length from %s to %s", asp79_path.shape, ash79_path.shape)
    logger.debug("The endpoints differ by %s and %s", np.linalg.norm(ash79_path[0] - asp79_path[0]),
                 np.linalg.norm(ash79_path[-1] - asp79_path[-1]))
    # print(asp79_path)
    ash79_path = utils.reparametrize_path_iter(ash79_path)
    if savepath is not None:
        np.savetxt(savepath, ash79_path)
        logger.info("Saved stringpath of %s points to %s", len(ash79_path), savepath)
    if plot:
        utils.plot_path(asp79_path, twoD=False, label="asp79_path")
        utils.plot_path(ash79_path, twoD=False, label="ash79_path")
        plt.show()


def _create_beta1_targetedmd():
    cvs_dir = "../../gpcr/cvs/beta1-cvs/"
    beta1_cvs = load_object(cvs_dir + "cvs")
    create_targetmd_input(beta1_cvs, np.loadtxt(cvs_dir + "string-paths/string0_beta1_inactive_endpoint.txt"),
                          steps_per_point=utils.rint(3000000 / 22), backwards=True)


def _create_beta2_targetedmd():
    time_ns = 10
    kappa = 10000.0
    topology = md.load(
        utils.project_dir + "gpcr/reference_structures/3p0g/asp79-apo/equilibrated.gro").topology
    cvs = colvars.cvs_definition_reader.load_cvs(cvs_len5path + "cvs.json")
    plumed_file_content = colvars.plumed_tools.convert_to_plumed_restraints(cvs, topology, kappa)
    logger.info("plumed_file_content:\n\n%s\n", plumed_file_content)
    # stringpath = np.loadtxt(cvs_len5path + "string-paths-drorpath/dror_path_fixedep.txt")
    # stringpath = np.loadtxt(cvs_len5path + "string-paths-3p0g-3sn6/straight4points.txt")
    create_targetmd_input(cvs, stringpath,
                          steps_per_point=utils.rint(time_ns * 1e6 / len(stringpath)),
                          backwards=False,
                          kappa=kappa)


def change_endpoints(stringpath, new_startpoint, new_endpoint):
    stringpath = stringpath.copy()
    logger.debug("Changing start point from %s to %s", new_startpoint, stringpath[0])
    stringpath[0] = new_startpoint
    logger.debug("Changing start point from %s to %s", new_endpoint, stringpath[-1])
    stringpath[-1] = new_endpoint
    new_stringpath = utils.reparametrize_path_iter(stringpath, arclength_weight=None)
    return new_stringpath


def create_endpoint_from_equilibration(cvs, traj, plot=True):
    """
    :param cvs:
    :param traj: an equilibration trajectory
    :param plot
    :return: np.array with coordinates of the endpoints
    """
    evals = colvars.eval_cvs(traj, cvs)
    point = evals.mean(axis=0)
    if plot:
        utils.plot_path(evals)
        utils.plot_path(point, scatter=True, text="Average")
        plt.show()
    return point


def _create_beta2_endpoints(plot=True):
    start_simudir = "/data/oliver/pnas2011b-Dror-gpcr/equilibration/"
    cvs = colvars.cvs_definition_reader.load_cvs(cvs_len5path + "cvs.json")
    starttraj = md.load(
        start_simudir + "aug/fixed_loop/asp79-holo-protonated-ligand/production/3p0g-protonated-ligand-full.xtc",
        top=start_simudir + "aug/fixed_loop/asp79-holo-protonated-ligand/production/3p0g-protonated-ligand.gro"
    )
    startpoint = create_endpoint_from_equilibration(cvs, starttraj, plot=plot)
    endtraj = md.load(
        "/data/oliver/pnas2011b-Dror-gpcr/equilibration/noligand-jan24/2rh1-charmm-gui/gromacs/step7.xtc",
        top="/data/oliver/pnas2011b-Dror-gpcr/equilibration/noligand-jan24/2rh1-charmm-gui/gromacs/step6.6_equilibration.gro"
    )
    endpoint = create_endpoint_from_equilibration(cvs, endtraj, plot=plot)
    return startpoint, endpoint


def _update_beta2_endpoints(outpath, plot=True):
    startpoint, endpoint = _create_beta2_endpoints(plot=plot)
    freeendpoint_stringpath = np.loadtxt(
        utils.project_dir + "gpcr/cvs/len5/asp79/string-paths/dror_path_looseep.txt")
    new_stringpath = change_endpoints(freeendpoint_stringpath, startpoint, endpoint)
    np.savetxt(outpath, new_stringpath)
    if plot:
        utils.plot_path(freeendpoint_stringpath, label="old")
        utils.plot_path(new_stringpath, label="new")
        plt.show()


if __name__ == "__main__":
    setup_beta2_string = False
    logger.info("Starting")
    # create_3po0_3sn6_string()
    # _create_ash79_initial_string(savepath="/home/oliver/slask/holo5-optimized-avg-strings/ash79-in.txt")
    _create_beta2_targetedmd()
    if setup_beta2_string:
        _update_beta2_endpoints(
            utils.project_dir + "gpcr/cvs/len5/asp79/string-paths/dror_path_equilibratedep.txt",
            plot=True)

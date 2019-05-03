from __future__ import absolute_import, division, print_function

import sys

reload(sys)
sys.setdefaultencoding(
    'utf-8')  # Fixing weird errors according to https://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte

from analysis.FE_analysis.visualization import *

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
from analysis.FE_analysis.control import *
from analysis.FE_analysis.simu_config import *

logger = logging.getLogger("FEAnalysis")


def start(plot=False, show_1D_different_plots=False):
    cvtypes = [
        ####CVS COMMONLY USED FOR ANALYSIS#########
        "5cvs_0and1", # Can enter 'None' as cv for 1d plots. 5cvs_0and1, 5cvs_1and3, 5cvs_1and4, 5cvs_3and4 look ok
        "dror",
        # "nature_2018",
        "loose_coupling_ligand_gprotein",
        "loose_coupling_ligand_connector",
        "loose_coupling_connector_gprotein",
        "water_solubility",
        "DRY-motif-inactive",
        "ionic_lock-YY",
        "rmsd",
        ############LESS COMMON CVS BELOW#########
        # "avg_string_cv",
        # "ioniclock_old",
        # "YY_old",
        #"discrete_classifier",  # "discrete_classifier_old"
        # "nature_2018_FE",
        # "nmrcvs",
        # "cell_2015",
        # "probability_classifier",
        ############OTHER PROTEINS#########
       # "beta1-5cvs_0and1",
       # "beta1-npxxy",
    ]
    full_simulations = [
        # "holo-optimized",
       # "apo-optimized",
        "straight-holo-optimized",
        # "endpoints-holo",
        # "to3sn6-holo",
        # "to3sn6-apo",
        # "holo-curved",
        # "apo-curved",
        # "holo-straight",
        # "beta1-apo",
        # "pierre-ash79",
        # "pierre-asp79_Na"
    ]
    cross_test_simulations = [  # For testing convergence
        # "apo-optimized_part1",
        # "apo-optimized_part2",
        "holo-optimized_part1",
        "holo-optimized_part2",
    ]
    stationary_methods = ["msm"]  # "msm", "db", "msm_bayesian"
    ####RUN THE ANALYZERS
    string_simulations = full_simulations
    analyzers = []
    for cvtype in cvtypes:
        for simu_id in string_simulations:
            logger.info("Starting with simulation %s", simu_id)
            (start_iteration, last_iteration) = simuid_to_iterations(simu_id)
            for stationary_method in stationary_methods:
                try:
                    logger.info("Starting FE computations for cv %s, simu_id %s and stationary_method %s", cvtype,
                                simu_id,
                                stationary_method)
                    feanalysis = FEAnalysisController(get_args_for_simulation(simu_id),
                                                      cvtype,
                                                      start_iteration, last_iteration,
                                                      stationary_method=stationary_method,
                                                      transition_frame_loader=simuid_to_transitions(simu_id, cvtype))
                    feanalysis.compute()
                    analyzers.append(feanalysis)
                except Exception as ex:
                    logger.error("Failed to compute FE for simu %s, cvs %s and stationary method %s", simu_id, cvtype,
                                 stationary_method)
                    logger.exception(ex)
    if plot:
        for i, feanalysis in enumerate(analyzers):
            last_plot = i == len(analyzers) - 1
            plot_immediately = (feanalysis.dependent_cvs is None and len(feanalysis.cv_indices) > 1) \
                               or (feanalysis.dependent_cvs is not None and len(feanalysis.dependent_cvs)) > 1 \
                               or show_1D_different_plots
            feanalysis.visualize(do_plot=plot_immediately, plot_reference_structures=plot_immediately or last_plot)
            if not plot_immediately and last_plot:
                plt.show()
        logger.debug("Done")


if __name__ == "__main__":
    start()

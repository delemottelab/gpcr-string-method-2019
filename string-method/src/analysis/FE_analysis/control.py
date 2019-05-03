from __future__ import absolute_import, division, print_function

from analysis.FE_analysis.visualization import *

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
from analysis.FE_analysis.calculation import *
from system_setup import create_cvs
import run_simulation as rs
from notebooks import trajclassifier

logger = logging.getLogger("FEController")


class FEAnalysisController(object):
    """
    Class for configuration, computing FE and visualizing it all in one.
    """

    def __init__(self, args, cvtype, start_iteration, last_iteration, transition_frame_loader=None,
                 stationary_method="msm", load_transition_count=None):
        object.__init__(self)
        self.runner = rs.StringIterationRunner(args)
        self.cvtype = cvtype
        self.field_visualizer = None
        self.rescale = True
        self.smear_transitions = False
        self.normalize_grid = False
        self.field_cutoff = 10
        self.field_label = None
        self.gridmin, self.gridmax = None, None
        self.stringpath_type = "None"
        self.show_network, self.show_FE = False, True
        self.plot_probability = False
        self.stationary_method = stationary_method
        self.start_iteration, self.last_iteration = start_iteration, last_iteration
        self.transition_frame_loader = transition_frame_loader
        self.save_network_plot = True
        self.cvi, self.cvj = 0, None
        self.cv_indices = None
        self.dependent_cvs = None
        self.load_transition_count = load_transition_count
        """DIFFERENT CVS"""
        if cvtype.startswith("5cvs"):
            color_cvs = utils.load_binary("../gpcr/cvs/old/cvs-len5_good/cvs.pkl")
            self.cvs = np.array(color_cvs)
            indices_string = cvtype.split("_")[1].split("and")
            self.cvi = int(indices_string[0])
            if indices_string[1] == "None":
                self.cvj = None
            else:
                self.cvj = int(indices_string[1])
            self.gridmin, self.gridmax = None, None  # -0.2, 1.2
            self.ngrid = 39
            self.smear_transitions = False
            self.rescale = True
            self.stringpath_type = "avg"
        elif cvtype.startswith("beta1-5cvs"):
            color_cvs = utils.load_binary("../gpcr/cvs/beta1-cvs/cvs.pkl")
            self.cvs = np.array(color_cvs)
            indices_string = cvtype.split("_")[1].split("and")
            self.cvi = int(indices_string[0])
            if indices_string[1] == "None":
                self.cvj = None
            else:
                self.cvj = int(indices_string[1])
            self.ngrid = 30
            self.smear_transitions = False
            self.rescale = False
            self.stringpath_type = "avg"
        elif cvtype.startswith("beta1-npxxy"):
            self.cvs = create_cvs.create_beta1_npxxy_cvs()
            self.cvi, self.cvj = 0, 1
            self.ngrid = 30
            self.smear_transitions = False
            self.rescale = True
            self.normalize_grid = True
        elif cvtype == "ionic_lock-YY":
            self.cvs = [create_cvs.create_ionic_lock_cv(), create_cvs.create_YY_cv()]
            self.cvi, self.cvj = 0, 1
            self.ngrid = 41
            self.rescale = True
        elif cvtype == "ioniclock_old":
            self.stringpath_type = "1D"
            # A CV along a single reaction coordinate
            self.cvs = np.array(utils.load_binary("/home/oliverfl/projects/gpcr/cvs/ionic-lock-COM.pkl"))
            self.gridmin, self.gridmax = None, None  # 0.35, 2.1
            self.ngrid = 61
            # self.smear_transitions = False
            self.rescale = True
        elif cvtype == "YY_old":
            res1, res2 = 219, 326
            self.cvs = [colvars.CADistanceCv("|Y219-Y326|(CA)", res1, res2, periodic=False)]
            self.gridmin, self.gridmax = None, None
            self.ngrid = 101
            # self.smear_transitions = False
            self.rescale = True
            self.plot_probability = False
            self.field_cutoff = 50
        elif cvtype == 'rmsd':
            self.cvi, self.cvj = 0, 1
            self.cvs = utils.load_binary("../gpcr/cvs/rmsd-cvs/cvs.pkl")
            self.gridmin, self.gridmax = None, None  # -0.2, 2.0
            self.ngrid = 50
        elif cvtype.startswith("loose_coupling_"):
            self.cvi, self.cvj = 0, 1
            ligand_cv, connector_cv, gprotein_cv = create_cvs.create_loose_coupling_cvs(normalize=True)
            if cvtype.endswith("ligand_connector"):
                self.cvs = np.array([ligand_cv, connector_cv])
            elif cvtype.endswith("ligand_gprotein"):
                self.cvs = np.array([ligand_cv, gprotein_cv])
            elif cvtype.endswith("connector_gprotein"):
                self.cvs = np.array([connector_cv, gprotein_cv])
            self.rescale = True
            self.smear_transitions = False
            self.ngrid = 29  # 51 smearing, 29 regular
        elif cvtype == "dror":
            self.cvi, self.cvj = 0, 1  # 3, 4
            self.cvs = utils.load_binary("../gpcr/cvs/dror-cvs/cvs.pkl")
            logger.info("Norm scales for dror CVs, %s", ([(cv._norm_scale, cv._norm_offset) for cv in self.cvs]))
            self.gridmin, self.gridmax = -0.6, 1.8
            self.ngrid = 51 #41 for holo
            self.smear_transitions = False
            self.rescale = True
        elif cvtype == "probability_classifier":
            dt, nclusters = 4, 3
            probability_classifier_cvs = utils.load_binary(
                "/home/oliverfl/projects/gpcr/neural_networks/strajs_%s_clusters_dt%s/probability_classifier_cvs.pkl" % (
                    nclusters, dt))
            self.cvi, self.cvj = 0, 1
            self.cvs = np.array([probability_classifier_cvs[self.cvi], probability_classifier_cvs[self.cvj]])
            self.gridmin, self.gridmax = 0, nclusters
            self.ngrid = nclusters + 1
            self.show_network, self.show_FE = True, True
        elif cvtype == "nmrcvs":
            color_cvs = utils.load_binary("../gpcr/cvs/nmr-cvs/cvs.pkl")
            self.cvi, self.cvj = 0, 1  # , 2  # 3, 4
            self.cvs = np.array([color_cvs[self.cvi]])
            self.gridmin, self.gridmax = -0.2, 1.2
            self.ngrid = 40
            self.rescale = True
        elif cvtype.startswith("nature_2018"):
            # From https://www.nature.com/articles/nature22354
            # Distance 266-148
            res1, res2 = 266, 148
            self.cvs = [colvars.CADistanceCv("|TM4-TM6|(CA)", res1, res2, periodic=False)]
            self.gridmin, self.gridmax = 1.5, 6
            self.ngrid = 50
            # self.smear_transitions = False
            self.rescale = True
            self.field_cutoff = 1000
            if not cvtype.endswith("FE"):
                self.plot_probability = True
                self.field_visualizer = FRETConverter.convert_to_fret_efficiency
        elif cvtype == "cell_2015":
            # From https://www.sciencedirect.com/science/article/pii/S0092867415004997
            # Distance 265-148 (but 265 is missing in our system so we use 266 instead)
            res1, res2 = 266, 148
            # qCom = "protein and resSeq {} and element != H"
            # self.cvs = [colvars.CADistanceCv("|TM4-TM6|(CA)", res1, res2, periodic=False)]
            # self.cvs = [colvars.COMDistanceCv("|TM4-TM6|(COM)", qCom.format(res1), qCom.format(res2))]
            self.cvs = [colvars.MaxDistanceCv("Max|TM4-TM6|", res1, res2)]
            self.gridmin, self.gridmax = 1.5, 6
            self.ngrid = 100
            # self.smear_transitions = False
            self.rescale = True
            self.plot_probability = True
            self.field_cutoff = 1000
        elif cvtype.startswith("DRY-motif"):
            self.cvi, self.cvj = 0, None
            active_DRY_rmsd, inactive_DRY_rmsd = create_cvs.create_DRY_cvs(normalize=True)
            self.cvs = np.array([active_DRY_rmsd if cvtype == "DRY-motif-active" else inactive_DRY_rmsd])
            self.ngrid = 40
            # self.smear_transitions = False
            self.rescale = True
        elif cvtype.startswith("discrete_classifier"):
            dt, nclusters = 4, 3
            if cvtype == "discrete_classifier_old":
                discrete_classifier_cv = utils.load_binary(
                    "/home/oliverfl/projects/gpcr/neural_networks/strajs_%s_clusters_dt%s/discrete_classifier_cv.pkl" % (
                        nclusters, dt))
                self.cvs = np.array([discrete_classifier_cv])
                self.cvi, self.cvj = 0, None
            else:
                classifier = utils.load_binary("/home/oliverfl/projects/gpcr/neural_networks/drorAcvs/classifier.pkl")
                scaler = utils.load_binary("/home/oliverfl/projects/gpcr/neural_networks/drorAcvs/scaler.pkl")
                simulation_cvs = utils.load_binary("../gpcr/cvs/cvs-len5_good/cvs.pkl")
                discrete_classifier_cv = trajclassifier.DependentDiscreteClassifierCv(
                    "Cluster",
                    trajclassifier.CvsVectorizer(simulation_cvs),
                    scaler,
                    classifier
                )
                self.dependent_cvs = np.array([discrete_classifier_cv])
                self.cvs = simulation_cvs
                self.cv_indices = [i for i in range(len(simulation_cvs))]
            self.smear_transitions = False
            self.gridmin, self.gridmax = 0, nclusters
            self.ngrid = nclusters + 1
            self.show_network, self.show_FE = True, False
        elif cvtype == "avg_string_cv":
            # A CV along a specific string
            simulation_cvs = utils.load_binary("../gpcr/cvs/cvs-len5_good/cvs.pkl")
            logger.info("Using stringpath from simulations %s-%s", self.start_iteration, self.last_iteration)
            # stringpath = self.runner.stringpath
            avg_string, strings = extra_analysis.compute_average_string(self.runner,
                                                                        start_iteration=self.start_iteration,
                                                                        end_iteration=self.last_iteration,
                                                                        rescale=False,
                                                                        plot=False,
                                                                        save=False)
            stringCv = colvars.StringIndexCv("avg{}-{}".format(self.start_iteration, self.last_iteration), avg_string,
                                             simulation_cvs, interpolate=False)
            stringCv.normalize(scale=1. * len(avg_string), offset=0)
            # self.smear_transitions = False
            self.dependent_cvs = np.array([stringCv])
            self.cvs = simulation_cvs
            # self.gridmin, self.gridmax = -0.1, 1.1  # len(avg_string)
            self.ngrid = 30  # 1 + utils.rint(len(avg_string) / 3.2)
            self.cv_indices = [i for i in range(len(simulation_cvs))]
            # self.ngrid = int(np.rint((self.gridmax - self.gridmin) * len(avg_string)))
        elif cvtype == "water_solubility":
            self.cvs = [create_cvs.WaterSolubilityCV(266, 0.8)]
            self.gridmin = 0
            self.gridmax = 80
            self.ngrid = self.gridmax - self.gridmin + 1
        else:
            raise Exception("No valid cv chose " + cvtype)
        self.cvs = np.array(self.cvs)
        logger.info("Using CVtype %s and CVs %s", cvtype, [cv.id + "," + str(cv) for cv in self.cvs])
        if self.cv_indices is None:
            self.cv_indices = [self.cvi] if self.cvj is None else [self.cvi, self.cvj]
        self.calculator = FECalculator(self.runner, self.cvs,
                                       start_iteration=self.start_iteration,
                                       last_iteration=self.last_iteration,
                                       ngrid=self.ngrid,
                                       normalize_grid=self.normalize_grid,
                                       gridmin=self.gridmin,
                                       gridmax=self.gridmax,
                                       smear_transitions=self.smear_transitions,
                                       plot=False,
                                       ignore_missing_files=False,
                                       cv_indices=self.cv_indices,
                                       dependent_cvs=self.dependent_cvs)

    def compute(self):
        self.fe, self.probability_distribution, self.transition_count = self.calculator.compute(load_swarm_values=None,
                                                                                                load_transition_count=self.load_transition_count,
                                                                                                transition_frame_loader=self.transition_frame_loader,
                                                                                                method=self.stationary_method)
        self.field = self.probability_distribution if self.plot_probability else self.fe

    def visualize(self, do_plot=True, plot_reference_structures=True):
        if self.field_visualizer is not None:
            self.field_visualizer(self.runner, self.fe, self.probability_distribution,
                                  get_axis(self.calculator.gridmin, self.calculator.gridmax,
                                           self.probability_distribution))
            if do_plot:
                plt.show()
            return  # Don't plot anything else
        if self.stringpath_type == "avg" and len(self.cvs) > 1:
            stringpath, strings = extra_analysis.compute_average_string(self.runner,
                                                                        start_iteration=self.start_iteration,
                                                                        end_iteration=self.last_iteration,
                                                                        rescale=self.rescale,
                                                                        plot=False, save=False)
            stringpath = stringpath[:, self.cv_indices]
        elif self.stringpath_type == "file":
            stringpath = np.loadtxt(
                # "/home/oliverfl/projects/gpcr/simulations/strings/jan/5cvs-straight/gpcr/cvs/cvs-len5_good/string-paths/string71.txt"
                # "/home/oliverfl/projects/gpcr/simulations/strings/jan/5cvs-drorpath/gpcr/cvs/cvs-len5_good/string-paths/string84.txt"
                "/home/oliverfl/projects/gpcr/simulations/strings/jan/apo5-drorpath/gpcr/cvs/cvs-len5_good/string-paths/string100.txt"

            )[:, self.cv_indices]
        elif self.stringpath_type == "iteration":
            stringpath = self.runner.stringpath
        else:
            stringpath = None
        if self.field_label is None:
            self.field_label = "probability density" if self.plot_probability else r'$\Delta G$ [kcal/mol]'
        title = "method={},iteration {}-{}{}".format(self.stationary_method,
                                                     self.start_iteration,
                                                     self.last_iteration,
                                                     ", smeared" if self.smear_transitions else "")
        # title = None
        plot_cvs = self.cvs[self.cv_indices] if self.dependent_cvs is None else self.dependent_cvs
        if self.show_FE:
            plot_field(self.runner,
                       self.field,
                       plot_cvs,
                       self.calculator.gridmin,
                       self.calculator.gridmax,
                       rescale=self.rescale,
                       stringpath=stringpath,
                       stringlabel=None,
                       cutoff=self.field_cutoff,
                       title=title,
                       cv_labels=[extra_analysis.get_cv_description(cv.id, use_simpler_names=True) for cv in plot_cvs],
                       field_label=self.field_label,
                       axisdata=None,
                       plot_reference_structures=plot_reference_structures)
            if do_plot:
                plt.show()
        if self.show_network:
            plot_networks(self.probability_distribution, self.field, self.transition_count, plot_cvs,
                          description=self.runner.simu_id,
                          simu_id=self.runner.simu_id,
                          save=self.save_network_plot)
            if do_plot and not self.save_network_plot:
                plt.show()

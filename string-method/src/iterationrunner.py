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
import shutil
import glob
from jobs import *
from stringprocessor import *
from jobs.job_executors import *
from os.path import abspath, isabs

append_startmodes = ["append", "analysis"]


class AbstractIterationRunner(object):
    def __init__(self, args):
        super(AbstractIterationRunner, self).__init__()
        """
        TODO load objects etc. outside this class
        :param args:
        """
        self.args = args
        for a in vars(args):
            setattr(self, a, getattr(args, a))
        self.logger = logging.getLogger(self.__class__.__name__)
        self._fix_directories()
        self.iteration = None
        self.job_executor = None

    def _fix_directories(self):
        if not isabs(self.simulation_dir):
            self.simulation_dir = self.working_dir + self.simulation_dir
        if not isabs(self.structure_dir):
            self.structure_dir = self.working_dir + self.structure_dir
        if not isabs(self.template_dir):
            self.template_dir = self.working_dir + self.template_dir
        if hasattr(self, "cvs_dir") and not isabs(self.cvs_dir):
            self.cvs_dir = self.working_dir + self.cvs_dir
        if hasattr(self, "string_filepath") and not isabs(self.string_filepath):
            self.string_filepath = self.cvs_dir + self.string_filepath

    def run(self):
        if self.start_mode == 'server':
            self.create_files()
        else:
            self.logger.warn(
                "Start mode %s does not yet support the creation of any files. You must be sure all files were already created.",
                self.start_mode)
        self.setup_jobs()
        self.wait_for_completion()
        if self.start_mode == 'append':
            self.logger.info("Changing startmode to 'server' from 'append' for next iteration")
            self.start_mode = 'server'

    def init_iteration(self, iteration):
        """Prepares the runner for the given iteration"""
        pass

    def after_iteration(self):
        """Called after everything else has been finished with this runner"""
        pass


class StringIterationRunner(AbstractIterationRunner):
    def __init__(self, args):
        AbstractIterationRunner.__init__(self, args)
        self._init_cvs()
        self.restraints_template = utils.read_file(self.cvs_dir + "restraints_template_%s.txt" % self.simulator)
        # self.commands = type('', (), {})  # object()
        self.topology = md.load(abspath(self.structure_dir + "equilibrated.gro")).topology
        self.init_iteration(args.iteration)
        self.logger.info("Looking in the following directories for files:\n%s, \n%s, \n%s, \n%s, \n%s, \n%s",
                         self.working_dir, self.simulation_dir, self.structure_dir, self.template_dir, self.cvs_dir,
                         self.string_filepath)

    def _init_cvs(self):
        if self.cvs_filetype is None:
            self.cvs_filetype = "json" if self.version >= 2.2 else "pkl"
        if self.cvs_filetype == "json":
            self.cvs = colvars.cvs_definition_reader.load_cvs(self.cvs_dir + "cvs.json")
        elif self.cvs_filetype == "pkl":
            self.cvs = utils.load_binary(self.cvs_dir + "cvs.pkl")
        else:
            raise Exception("Uknown cvs_filetype {}".format(self.cvs_filetype))
        self.logger.info("Loaded CVs: %s", [cv.id for cv in self.cvs])

    def init_iteration(self, iteration):
        """
        Prepare iteration. Load the output of the previous iteration.
        Note that the field 'iteration' is the most important state variable of this class.
        """
        self.logger.info("New iteration. Previous iteration: %s, new iteration: %s", self.iteration, iteration)
        self.iteration = iteration
        self.stringpath = self._load_stringpath(iteration - 1)
        self.input_coordinate_mapping = self._load_input_coordinate_mapping(iteration - 1)
        self.logger.debug("Loaded string path\n%s", self.stringpath)
        self.swarmProcessor = SwarmProcessor(self)
        if self.job_executor is not None:
            self.job_executor.stop()
        self.job_executor = JobExecutor("JobExecutor", self.job_pool_size,
                                        append_mode=self.start_mode in append_startmodes,
                                        callbacks=[self.swarmProcessor])
        self.swarm_batches_count = np.zeros((len(self.stringpath),), dtype=int)  # Swarm batches count per index
        if self.start_mode in append_startmodes:
            self.compute_number_of_swarms()

    def _load_input_coordinate_mapping(self, iteration):
        filepath = self._get_stringfilepath(str(iteration) + "-mapping")
        if not os.path.exists(filepath):
            return None
        return np.loadtxt(filepath)

    def _load_stringpath(self, dynamic_filename):
        return np.loadtxt(self._get_stringfilepath(dynamic_filename))

    def _get_stringfilepath(self, dynamic_filename):
        if "%s" in self.string_filepath:
            filepath = self.string_filepath % dynamic_filename
        else:
            filepath = self.string_filepath
        return filepath

    def after_iteration(self):
        super(StringIterationRunner, self).after_iteration()

    def _write_restraints(self, point, point_idx):
        rescaled_point = colvars.rescale_points(self.cvs, point)  # actual non-normalized distance
        # logger.debug("Converted point %s, %s", rescaled_point, point)
        if self.simulator == 'gromacs':  # copy the topology so we can add point specific restraints (deprecated)
            shutil.copytree(self.structure_dir + "topology", self.point_path(point_idx) + "topology/")
        restraints = self.restraints_template
        for idx, cv in enumerate(self.cvs):
            # TODO use utils.inject instead
            tol = cv._norm_scale * self.restraint_tolerance  # TODO norm scale can be computed as eval(1) - eval(0) or something
            mark = "$" + "cv" + str(idx)
            restraints = restraints.replace(mark + "_center", str(rescaled_point[idx]))
            restraints = restraints.replace(mark + "_low", str(rescaled_point[idx] - tol))
            restraints = restraints.replace(mark + "_up1", str(rescaled_point[idx] + tol))
            restraints = restraints.replace(mark + "_up2", str(rescaled_point[idx] + 2 * tol))
            restraints = restraints.replace(mark + "_idx", str(idx))
        # append restraints
        with open(self.point_path(point_idx) + self.restraints_out_file, "a") as restraintsfile:
            restraintsfile.write("\n" + restraints)

    def iteration_path(self, iteration=None):
        return self.simulation_dir + str(self.iteration if iteration is None else iteration) + "/"

    def point_path(self, point_idx, iteration=None):
        return self.iteration_path(iteration) + str(point_idx) + "/"

    def point_name(self, point_idx, iteration=None):
        return "i%sp%s" % (self.iteration if iteration is None else iteration, point_idx)

    def swarm_name(self, point_idx, swarm_batch_idx, swarm_idx=None, iteration=None):
        if self.version >= 2:
            suffix = "b"
            if swarm_idx is not None:
                suffix += str(swarm_idx)
        else:
            suffix = ""
        return self.point_name(point_idx, iteration=iteration) + "s" + str(swarm_batch_idx) + suffix

    def to_submission_filename(self, path, name):
        return path + "submit_" + name + ".sh"

    def _create_scripts(self, point_idx):
        self._create_restrained_scripts(point_idx)
        for swarmid in range(self.min_swarm_batches):
            self._create_swarms_scripts(point_idx, swarmid)
            self.swarm_batches_count[point_idx] += 1

    def _create_restrained_scripts(self, point_idx):
        if self.start_mode == "analysis":
            raise Exception("Cannot create scripts in analysis mode")
        point_path = self.point_path(point_idx)
        point_name = self.point_name(point_idx)
        restrained_run_file = point_path + point_name + ".sh"
        # Copy file to run simulation
        utils.copy_and_inject("bash/run_restrained_%s.sh" % self.simulator,
                              restrained_run_file,
                              [point_name, abspath(point_path), abspath(self.structure_dir), abspath(self.template_dir),
                               self.command_gmx],
                              marker="$%s", start_index=1)
        # Copy submission file
        utils.copy_and_inject("bash/submit_restrained_%s_%s.sh" % (self.simulator, self.environment),
                              self.to_submission_filename(point_path, point_name),
                              [point_name + self.simu_id, abspath(restrained_run_file)],
                              marker="$%s", start_index=1)

    def _create_swarms_scripts(self, point_idx, swarmid):
        if self.start_mode == "analysis":
            raise Exception("Cannot create scripts in analysis mode")
        point_path = self.point_path(point_idx)
        point_name = self.point_name(point_idx)
        swarm_name = self.swarm_name(point_idx, swarmid)
        # TODO make this (amount of time etc.) fully automated and as parameters
        swarm_run_file = point_path + swarm_name + ".sh"
        utils.copy_and_inject("bash/run_swarm.sh",
                              swarm_run_file,
                              [point_name, abspath(point_path), abspath(self.structure_dir),
                               abspath(self.template_dir), swarm_name, self.swarm_batch_size, self.command_gmx],
                              marker="$%s", start_index=1)
        # Copy submission file
        utils.copy_and_inject("bash/submit_swarm_%s_%s.sh" % (self.simulator, self.environment),
                              self.to_submission_filename(point_path, swarm_name),
                              [swarm_name + self.simu_id, abspath(swarm_run_file), self.swarm_batch_size],
                              marker="$%s", start_index=1)

    def create_files(self):
        """
        Create folders and write restraints etc.
        """
        self.logger.debug("Creating simu files for iteration %s", self.iteration)
        utils.makedirs(self.simulation_dir, overwrite=False)
        utils.makedirs(self.iteration_path(), overwrite=True, backup=True)
        if self.iteration > 1:
            previous_stringpath = self._load_stringpath(self.iteration - 2)
        else:
            previous_stringpath = self.stringpath
        string_length_diff = len(self.stringpath) - len(previous_stringpath)
        point_ratio = len(previous_stringpath) / len(self.stringpath)
        if string_length_diff > 0:
            self.logger.debug("Number of points changed. Mapping previous points to new ones")
        for point_idx, point in enumerate(self.stringpath):
            if self.fixed_endpoints and (point_idx == 0 or point_idx == len(self.stringpath) - 1):
                continue
            point_path = self.point_path(point_idx)
            utils.makedirs(point_path, overwrite=True, backup=True)
            previous_idx = utils.rint(point_idx * point_ratio)
            in_coordinates = self._get_input_coordinates(point_idx, previous_idx)
            shutil.copy(in_coordinates, point_path + self.point_name(point_idx) + "-in.gro")
            self._write_restraints(point, point_idx)
            self._create_scripts(point_idx)
        self.logger.info("All files created for iteration %s", self.iteration)

    def _get_input_coordinates(self, point_idx, previous_idx):
        """
        Get input coordinates from previous string iteration
        :param point_idx: this point
        :param previous_idx: previous corresponding point on string (they differ if the string changed length)
        :return:
        """
        # Copy previous restrained trajectory to use as input coordinates
        # logger.debug("%s<-%s", previous_idx, idx)
        if self.input_coordinate_mapping is None:
            input_point, input_swarm_batch, input_swarm = previous_idx, -1, -1
        else:
            input_point, input_swarm_batch, input_swarm = self.input_coordinate_mapping[point_idx]
        if input_swarm_batch < 0:
            coordinate_name = self.point_name(utils.rint(input_point), self.iteration - 1) + "-restrained.gro"
        else:
            coordinate_name = self.swarm_name(utils.rint(input_point),
                                              utils.rint(input_swarm_batch),
                                              iteration=self.iteration - 1) + str(utils.rint(input_swarm)) + ".gro"
        in_coordinates = self.point_path(utils.rint(input_point), iteration=self.iteration - 1) + coordinate_name
        return in_coordinates

    def wait_for_completion(self):
        """Start jobs and wait for them to finish"""
        self.job_executor.run()

    def setup_jobs(self):
        """
        Prepare and submit bash jobs
        """

        for point_idx in range(len(self.stringpath)):
            if self.fixed_endpoints and (point_idx == 0 or point_idx == len(self.stringpath) - 1):
                continue
            restrainjob = self._setup_restrained_job(point_idx)
            swarmjobs = []
            number_swarmjobs = self.swarm_batches_count[point_idx]
            for swarmid in range(number_swarmjobs):
                sj = self._setup_swarms_job(point_idx, swarmid)
                swarmjobs.append(sj)
                # print(sj)
            restrainjob.add_child_jobs(swarmjobs)
            self.job_executor.add(restrainjob)
            # break

    def _setup_restrained_job(self, point_idx):
        point_path = self.point_path(point_idx)
        point_name = self.point_name(point_idx)
        restrainjob = BashJob(point_name,
                              [self.command_submit, abspath(self.to_submission_filename(point_path, point_name))],
                              done_file=point_path + point_name + ".done", type="restrained")
        restrainjob.point_idx = point_idx
        return restrainjob

    def _setup_swarms_job(self, point_idx, swarmid):
        point_path = self.point_path(point_idx)
        # point_name = self.point_name(point_idx)
        swarm_name = self.swarm_name(point_idx, swarmid)
        # TODO: run multiple swarms in one simulation to better utilize resources
        # See http://manual.gromacs.org/documentation/5.1/user-guide/mdrun-features.html
        swarmjob = BashJob(swarm_name,
                           [self.command_submit, abspath(self.to_submission_filename(point_path, swarm_name))],
                           done_file=point_path + swarm_name + ".done", type="swarm")
        swarmjob.point_idx = point_idx
        swarmjob.swarmid = swarmid
        return swarmjob

    def run_new_swarmbatch(self, restrainedjob):
        if self.start_mode == "analysis":
            logger.warn("Not running new swarmbatch from %s since you are in startmode analys", restrainedjob.id)
            return
        point_idx = restrainedjob.point_idx
        logger.debug("Creating new swarm files and submitting a job for point %s", point_idx)
        swarmid = self.swarm_batches_count[point_idx]
        self._create_swarms_scripts(point_idx, swarmid)
        swarmjob = self._setup_swarms_job(point_idx, swarmid)
        restrainedjob.add_child_job(swarmjob)
        self.job_executor.add(swarmjob)
        self.swarm_batches_count[point_idx] += 1
        return swarmjob

    def compute_number_of_swarms(self):
        """Looks for the number of number of swarm submission files"""
        for point_idx in range(len(self.stringpath)):
            filepath = self.point_path(point_idx) + self.swarm_name(point_idx, 999).replace("999", "*") + ".sh"
            count = len(glob.glob(filepath))
            self.swarm_batches_count[point_idx] = count

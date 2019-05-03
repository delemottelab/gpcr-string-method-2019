from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("transitionloading")
from analysis import extra_analysis
from analysis.FE_analysis.simu_config import *
import mdtraj as md
import numpy as np
import colvars


class TransitionFrameLoader(object):

    def __init__(self, directory, traj_filename, top_filename, dt, lagtime, outfile="transitions_%s", traj_query=None):
        object.__init__(self)
        self.directory = directory
        self.traj_filename = traj_filename
        self.top_filename = top_filename
        self.outfile = outfile
        self.dt = dt
        self.lagtime = lagtime
        self.traj_query = traj_query
        if not lagtime % dt == 0:
            raise Exception("Lagtime must be divisible by dt (the time between frames)")

    def load_transitions(self, cvs):
        """Load saved transitions from npy file"""
        return np.load(self._to_outfilename(cvs) + ".npy")

    def compute(self, cvs=None, cv_indices=None, save=True, load=None):
        logger.info("Computing transition frames")
        if load is None or load:
            try:
                return self.load_transitions(cvs)
            except Exception as ex:
                logger.exception(ex)
                if load:
                    raise ex  # Throw error up the stack
                logger.debug("npy files not found. Computing instead.")
        traj = self.load_traj()
        nframes = len(traj)
        ncvs = len(cvs)
        # Every two frames form one transition in the trajectory.
        # That means N-1 transitions for N frames
        transitions = np.empty((nframes - 1, 2, ncvs))
        previous_frame_evals = None
        idx = 0
        cv_names = [cv.id for cv in cvs]
        for frame in traj:
            frame_evals = colvars.eval_cvs(frame, cvs)
            if previous_frame_evals is not None:
                transitions[idx, 0, :] = previous_frame_evals
                transitions[idx, 1, :] = frame_evals
                idx += 1
                if idx % 1e4 == 0:
                    logger.debug("Computed transition %s/%s for cvs %s", idx, nframes - 1, cv_names)
            previous_frame_evals = frame_evals
        if save:
            np.save(self._to_outfilename(cvs), transitions)
        return transitions

    def load_traj(self):
        """
        -Load FREE MD simulations
        """
        logger.debug("Loading files from directory %s", self.directory)
        atom_indices = None
        if self.traj_query is not None:
            toptraj = md.load(self.directory + self.top_filename)
            atom_indices = toptraj.top.select(self.traj_query)
            logger.debug("Only using %s atoms of the system based on query '%s'", len(atom_indices), self.traj_query)
        stride = int(np.rint(self.lagtime / self.dt))
        traj = md.load(
            self.directory + self.traj_filename,
            top=self.directory + self.top_filename,
            atom_indices=atom_indices,
            stride=stride)
        logger.debug("Loaded trajectory %s ", traj)
        return traj

    # def __str__(self):
    #     return "TransitionFrameLoader"
    def _to_outfilename(self, cvs):
        swarm_suffix = ""
        for cv in cvs:
            swarm_suffix += "_" + extra_analysis.get_cv_description(cv.id)
        return self.directory + self.outfile % swarm_suffix + "dt_%s_lt%s" % (self.dt, self.lagtime)


class StringSimulationFrameLoader(object):
    def __init__(self, simu_id, cvtype):
        object.__init__(self)
        self.simu_id = simu_id
        self.cvtype = cvtype
        self.start_iteration, self.last_iteration = simuid_to_iterations(simu_id)

    def compute(self, cvs=None, cv_indices=None, save=True, load=None):
        from analysis.FE_analysis.control import FEAnalysisController
        self.feanalysis = FEAnalysisController(get_args_for_simulation(self.simu_id),
                                               self.cvtype,
                                               self.start_iteration, self.last_iteration,
                                               stationary_method=None,
                                               transition_frame_loader=simuid_to_transitions(self.simu_id, self.cvtype))
        # self.feanalysis.compute()
        transitions = self.feanalysis.calculator.compute_swarm_values(load)
        if transitions is None:
            logger.warn("Transitions was None!!!")
        if self.feanalysis.dependent_cvs is not None:
            return colvars.eval_transitions(transitions, self.feanalysis.dependent_cvs, self.feanalysis.cvs)
        if cv_indices is None:
            return transitions
        else:
            return transitions[:, :, cv_indices]

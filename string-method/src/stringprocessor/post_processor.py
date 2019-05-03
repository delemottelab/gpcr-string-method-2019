from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from stringprocessor.processing_utils import *

arcweight_epsilon = 0.1


class SingleIterationPostProcessor(object):
    """
    Process the simulation results from one string iteration and write new coordinates
    """

    def __init__(self, runner, save=True, plot=False):
        super(SingleIterationPostProcessor, self).__init__()
        self.runner = runner
        self.swarmProcessor = runner.swarmProcessor
        self.logger = logging.getLogger("SingleIterationPostProcessor")
        self.iteration = runner.iteration
        # self.converge = sys.float_info.max
        self.convergence = None
        self.new_stringpath = None
        self.plot = plot
        self.save = save
        self.fixed_endpoints = runner.fixed_endpoints
        self.swarm_drift_scale = runner.swarm_drift_scale
        self.old_stringpath = runner.stringpath
        restrainedpath = self.swarmProcessor.restrained_start_coordinates
        if self.fixed_endpoints:
            restrainedpath[0] = self.old_stringpath[0]
            restrainedpath[-1] = self.old_stringpath[-1]
        self.input_old_stringpath = self.old_stringpath
        self.old_stringpath = restrainedpath
        self.input_coordinate_mapping = None

    def run(self):
        self.new_stringpath = self.compute_new_stringpath()
        self.convergence = np.linalg.norm(self.new_stringpath - self.old_stringpath) / np.linalg.norm(
            self.new_stringpath)
        if (self.iteration % self.runner.new_point_frequency) == 0 and \
                len(self.new_stringpath) < self.runner.max_number_points:
            self.new_stringpath = create_stringpath_files_of_different_lengths(self.runner, self.new_stringpath)
        self.input_coordinate_mapping = self.swarmProcessor.find_closest_trajectories(self.new_stringpath)
        if self.save:
            save_string(self.runner.string_filepath, self.runner.iteration, self.new_stringpath)
            save_input_coordinate_mapping(self.runner.string_filepath, self.runner.iteration,
                                          self.input_coordinate_mapping)
        if self.plot:
            if hasattr(self, "input_old_stringpath"):
                utils.plot_path(self.input_old_stringpath, "Input")
                utils.plot_path(self.old_stringpath, "Restrained")
            else:
                utils.plot_path(self.old_stringpath, "Old")
            utils.plot_path(self.new_stringpath, "New")
            plt.legend()
            plt.show()

    def compute_new_stringpath(self):
        drifted_string = self.compute_drifted_string()
        if self.runner.equidistant_points:
            arc_drifts = None
        else:
            arc_drifts = self.compute_weights_from_drift_along_arcs()
        return utils.reparametrize_path_iter(drifted_string, arclength_weight=arc_drifts)

    def compute_weights_from_drift_along_arcs(self):
        """Compute weights based on the average drift from the swarms between two adjacent points along the unitvector connecting two adjacent points"""
        arc_drifts = np.empty((len(self.old_stringpath) - 1,))
        for start_point in range(len(self.old_stringpath) - 1):
            end_point = start_point + 1
            # if self.fixed_endpoints and start_point == 0):
            #     path[start_point, :] = self.old_stringpath[start_point]
            transition_forward = self.swarmProcessor.compute_average_transition_metric(start_point, end_point)
            transition_backward = self.swarmProcessor.compute_average_transition_metric(end_point, start_point)
            arc_drifts[start_point] = max(arcweight_epsilon, transition_forward, transition_backward)
        return arc_drifts

    def compute_drifted_string(self):
        """Return the unparametrized string after the swarms drift"""
        path = np.empty(self.old_stringpath.shape)
        for point_idx in range(len(self.old_stringpath)):
            if self.fixed_endpoints and (point_idx == 0 or point_idx == len(self.old_stringpath) - 1):
                path[point_idx, :] = self.old_stringpath[point_idx]
                continue
            # Compute the average coordinate from all swarms' final locations
            avgdrift = self.swarmProcessor.compute_average_drift(point_idx)
            if np.isnan(avgdrift[0]):
                path[point_idx, :] = self.old_stringpath[point_idx, :]
            else:
                path[point_idx, :] = self.old_stringpath[point_idx, :] + avgdrift * self.swarm_drift_scale
                # TODO save/analyze std etc
        return path

    def merge_swarms(self):
        """Merges all swarms for the current iteration to one trajectory"""
        raise Exception("Implement again! TODO Use glob with regex and utils.sort_alphanumerical")
        # traj = None
        # for i in range(len(self.old_stringpath)):
        #     if self.fixed_endpoints and (i == 0 or i == len(self.old_stringpath) - 1):
        #         continue
        #     for s in range(self.runner.number_swarms):
        #         swarmtraj = load_swarm(self.runner, i, s, ignore_missing_files=self.ignore_missing_files,
        #                                traj_filetypes=self._traj_filetypes)
        #         if swarmtraj is None:
        #             continue
        #         if traj is None:
        #             traj = swarmtraj[-1]
        #         else:
        #             traj += swarmtraj[-1]
        return traj

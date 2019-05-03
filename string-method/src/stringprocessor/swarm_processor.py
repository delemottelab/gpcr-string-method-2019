from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from stringprocessor.processing_utils import *


class SwarmProcessor(object):
    """
    Class to adaptively see when the swarm drift has converged to a sensible value
    """

    def __init__(self, runner, ignore_missing_files=False):
        self.runner = runner
        self.ncvs = len(runner.cvs)
        self.fixed_endpoints = runner.fixed_endpoints
        self.ignore_missing_files = ignore_missing_files
        self.logger = logging.getLogger("swarmRTProcessor")
        self._traj_filetypes = ["xtc", "trr"]
        self.string_length = len(self.runner.stringpath)
        self.restrained_start_coordinates = np.zeros((self.string_length, self.ncvs))
        if self.fixed_endpoints:
            self.restrained_start_coordinates[0] = self.runner.stringpath[0]
            self.restrained_start_coordinates[-1] = self.runner.stringpath[-1]
        self.swarm_coordinates = []
        self.queued_swarms_count = np.zeros((self.string_length,), dtype=int)
        self.last_swarm_convergence_fraction = np.zeros((self.string_length)) + np.nan
        for i in range(self.string_length):
            self.swarm_coordinates.append(np.empty((0, self.ncvs)))

    def on_job_finished(self, job, already_finished=False):
        if job.type != "restrained" and job.type != "swarm":
            return
        point_idx = job.point_idx
        if job.type == "restrained":
            self.restrained_start_coordinates[point_idx] = self.compute_restrained_coordinates(point_idx)
        elif job.type == "swarm":
            self.queued_swarms_count[point_idx] -= 1
            previous_avg_drift = self.compute_average_drift(point_idx)
            new_coordinates = self.compute_swarm_coordinates(point_idx, job.swarmid)
            self.swarm_coordinates[point_idx] = np.append(self.swarm_coordinates[point_idx], new_coordinates, axis=0)
            if len(previous_avg_drift) == 0:
                swarm_convergence_fraction = np.nan
            else:
                new_avg_drift = self.compute_average_drift(point_idx)
                swarm_convergence_fraction = np.linalg.norm(new_avg_drift - previous_avg_drift) / np.linalg.norm(
                    new_avg_drift)
            self.logger.debug(
                "swarm_convergence_fraction for point %s after %s swarms: %s. %s more swarm jobs currently in queue for this point",
                point_idx,
                len(self.swarm_coordinates[point_idx]), swarm_convergence_fraction, self.queued_swarms_count[point_idx])
            self.last_swarm_convergence_fraction[point_idx] = swarm_convergence_fraction
            if (np.isnan(
                    swarm_convergence_fraction) or swarm_convergence_fraction > self.runner.swarm_convergence_fraction) \
                    and self.queued_swarms_count[point_idx] == 0:
                self.runner.run_new_swarmbatch(job.parent_jobs[0])

    def on_job_began(self, job):
        pass

    def on_job_queued(self, job):
        if job.type != "swarm":
            return
        self.queued_swarms_count[job.point_idx] += 1

    def eval_cvs(self, traj):
        return colvars.eval_cvs(traj, self.runner.cvs)

    def compute_restrained_coordinates(self, point_idx):
        traj = load_restrained(self.runner, point_idx)
        return self.eval_cvs(traj).squeeze()

    def compute_swarm_coordinates(self, point_idx, swarmid):
        coords = np.empty((self.runner.swarm_batch_size, self.ncvs))
        for i in range(self.runner.swarm_batch_size):
            swarmtraj = load_swarm(self.runner, point_idx, swarmid, i,
                                   ignore_missing_files=self.ignore_missing_files,
                                   traj_filetypes=self._traj_filetypes)
            if swarmtraj is None:
                raise Exception("Swarm {} not found for point {}".format(swarmid, point_idx))
            coords[i, :] = self.eval_cvs(swarmtraj[-1])
        return coords

    def compute_swarm_drifts(self, point_idx):
        start = self.restrained_start_coordinates[point_idx]
        swarm_coordinates = self.swarm_coordinates[point_idx]
        if len(swarm_coordinates) == 0:
            return np.empty((len(start), 0))
        return swarm_coordinates - start

    def compute_average_drift(self, point_idx):
        swarm_drifts = self.compute_swarm_drifts(point_idx)
        if len(swarm_drifts) == 0:
            return np.empty((1, self.ncvs)) + np.nan
        return np.mean(swarm_drifts, axis=0)

    def compute_average_transition_metric(self, point_i, point_j):
        """
        Computes transition metric i->j defined as the mean of the swarm displacement from point i and j of swarms starting at point i,
        scaled by the sum of the displacement to point i and point j:
        :param point_i:
        :param point_j:
        :return: av value between 0 and 1 or -1 if there are no swarms for this point. A value of 1 means a full transition and 0 no transition
        """
        xi = self.restrained_start_coordinates[point_i]
        xj = self.restrained_start_coordinates[point_j]
        swarm_coordinates = self.swarm_coordinates[point_i]
        if len(swarm_coordinates) == 0:
            return -1
        sum_transitions = 0
        for xs in swarm_coordinates:
            dist_is = np.linalg.norm(xs - xi)
            dist_js = np.linalg.norm(xs - xj)
            sum_transitions += dist_is / (dist_is + dist_js)
        return sum_transitions / len(swarm_coordinates)

    def find_closest_trajectories(self, stringpath):
        """
        Find closest swarm or restrained simulation to the coordinates specified by the parameter stringpath
        :return an array with length of string path where every row contains the point index, swarm batch index and swarm index
        If it maps to a restrained simulation the swarm indices are negative
        """
        frames = np.empty((len(stringpath), 3), dtype=int)
        for point_idx, coordinates in enumerate(stringpath):
            if self.fixed_endpoints and point_idx == 0:  # start point
                frames[point_idx] = np.array([point_idx, -1, -1])
            elif self.fixed_endpoints and point_idx == len(stringpath) - 1:  # end point
                # Note how we handle the case when the string has changed length here
                frames[point_idx] = np.array([self.string_length - 1, -1, -1])
            else:
                frames[point_idx] = self.find_closest_trajectory(coordinates)
        return frames

    def find_closest_trajectory(self, coordinates):
        """
        Find closest swarm or restrained simulation
        :return an array which contains the point index, swarm batch index and swarm index
        If it maps to a restrained simulation the swarm indices are negative
        """
        restrained_traj, restrained_dist = self.find_closest_restrained(coordinates)
        swarm_traj, swarm_dist = self.find_closest_swarm(coordinates)
        if restrained_dist < swarm_dist:
            return restrained_traj
        else:
            return swarm_traj

    def find_closest_restrained(self, coordinates):
        """
        Find the closest restrained start coordinate for the string.
        If the endpoints are fixed, the endpoints will not be returned as closest points
        :param coordinates:
        :return:
        """
        dists = np.linalg.norm(self.restrained_start_coordinates - coordinates, axis=1)
        for k in range(0, self.string_length):
            closest_point = np.argpartition(dists, k)[k]
            is_endpoint = self.fixed_endpoints and (closest_point == 0 or closest_point == (self.string_length - 1))
            # Don't map to the end points, find the second closest
            if not is_endpoint:
                min_dist = np.partition(dists, k)[k]
                break
            else:
                min_dist = sys.maxint
        return np.array([closest_point, -1, -1]), min_dist

    def find_closest_swarm(self, coordinates):
        closest_point, closest_batch, closest_swarm = -1, -1, -1
        min_dist = sys.maxint
        for point_idx in range(self.string_length):
            if len(self.swarm_coordinates[point_idx]) == 0:
                continue
            dists = np.linalg.norm(self.swarm_coordinates[point_idx] - coordinates, axis=1)
            point_min_dist = dists.min()
            if point_min_dist < min_dist:
                min_dist = point_min_dist
                closest_point = point_idx
                min_idx = dists.argmin()
                closest_batch = utils.rint(np.floor(min_idx / self.runner.swarm_batch_size))
                closest_swarm = utils.rint(min_idx - self.runner.swarm_batch_size * closest_batch)
        return np.array([closest_point, closest_batch, closest_swarm]), min_dist

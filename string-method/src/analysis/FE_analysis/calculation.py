from __future__ import absolute_import, division, print_function

from analysis.FE_analysis.visualization import *

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import stringprocessor
import os
from analysis.FE_analysis.experimental_comparison import *
from analysis.FE_analysis.index_converter import *
from msmbuilder import msm, tpt, hmm
from analysis import extra_analysis

logger = logging.getLogger("FECalc")


def cubic_spline(center, point, sigma):
    """http://www.mathcces.rwth-aachen.de/_media/3teaching/0classes/archiv/042_ces_seminar_smoothparticlehydrodynamics.pdf"""
    q = np.linalg.norm(center - point) / sigma
    if q <= 1:
        return 1 - 1.5 * q ** 2 + 0.75 * q ** 3
    elif q <= 2:
        return 0.25 * (2 - q) ** 3
    else:
        return 0


class FECalculator(object):
    """
    Calculate the Free energy from swarm transitions
    """
    kB = 0.0019872041  # kcal/(mol*K)
    T = 310.15

    def __init__(self, runner, cvs, start_iteration, last_iteration=-1, ignore_missing_files=False, ngrid=50,
                 normalize_grid=False,
                 gridmin=-0.2, gridmax=1.2, save=True, smear_transitions=False, plot=False, lagtime=1e-11,
                 cv_indices=None, dependent_cvs=None):
        """
        :param runner:
        :param cvs:
        :param start_iteration:
        :param last_iteration:
        :param ignore_missing_files:
        :param ngrid:
        :param gridmin:
        :param gridmax:
        :param save:
        :param smear_transitions:
        :param plot:
        :param lagtime:
        :param cv_indices: the indices of the cvs parameter to use in transition count. If None, use all
        :param dependent_cvs: cvs which don't process trajectories. Instead they will process the output of the parameter 'cvs', convert it to new values and use it in the final computation of the transition matrix
        """
        self.runner = runner
        if not hasattr(cvs, "__len__"):
            cvs = [cvs]
        self.cvs = cvs
        self.dependent_cvs = dependent_cvs
        if cv_indices is None:
            self.cv_indices = np.arange(0, len(self.cvs))
        else:
            self.cv_indices = cv_indices
        self.start_iteration = start_iteration
        self.last_iteration = last_iteration
        self.ignore_missing_files = ignore_missing_files
        self.ngrid = ngrid  # Note that there are ngrid -1 bins per dimension
        self.gridmin = gridmin
        self.gridmax = gridmax
        if self.gridmax is None or self.gridmin is None:
            self.bin_width = None
        else:
            self.bin_width = FECalculator.compute_bin_width(self.gridmin, self.gridmax, self.ngrid)
        self.normalize_grid = normalize_grid
        self.setup_converter()
        self.save = save
        self.smear_transitions = smear_transitions
        self.plot = plot
        self.lagtime = lagtime

    def setup_converter(self):
        self._ndim = len(self.cv_indices) if self.dependent_cvs is None else len(self.dependent_cvs)
        converter = IndexConverter(self._ndim, self.ngrid)
        self._indexconverter = converter

    def compute_swarm_values(self, load):
        file_suffix, transition_count_suffix = self._get_suffix()
        iteration = self.start_iteration
        swarm_values = None
        path = self.runner.working_dir + "../swarm_transitions/swarms%s/" % file_suffix
        utils.makedirs(path, overwrite=False)
        while self.last_iteration < 0 or iteration <= self.last_iteration:
            filename = path + str(iteration)
            if load is None or load:
                # load is None means compute file if not exists -> TODO Make better parameter type
                try:
                    newvals = np.load(filename + ".npy")
                except IOError as err:
                    newvals = None
                if newvals is None and load:
                    logger.warn("Did not find swarms to load for iteration %s. Skipping this iteration", iteration)
                    iteration += 1
                    continue
            else:
                newvals = None
            if newvals is None:
                newvals = self._compute_iter_swarm_values(iteration)
                if newvals is None:
                    logger.warn("Did not find swarms for iteration %s. Skipping this iteration", iteration)
                    iteration += 1
                    continue
                if self.save:
                    logger.debug("Saving swarms file %s with %s entries", filename, newvals.shape)
                    np.save(filename, newvals)
            if swarm_values is None:
                swarm_values = newvals
            else:
                swarm_values = np.append(swarm_values, newvals, axis=0)
            iteration += 1
        if self.plot:
            self.plot_swarm_values(swarm_values)
        return swarm_values

    def _compute_iter_swarm_values(self, iteration):
        self.runner.init_iteration(iteration)
        startpoint = 1 if self.runner.fixed_endpoints else 0
        endpoint = len(self.runner.stringpath) - 1 if self.runner.fixed_endpoints else len(self.runner.stringpath)
        swarm_values = []
        for i in range(startpoint, endpoint):
            if self.runner.fixed_endpoints and (i == 0 or i == len(self.runner.stringpath) - 1):
                continue
            logger.debug("Computing cv values for swarms for iteration %s and point %s and cvs %s" % (
                self.runner.iteration, i, [cv.id for cv in self.cvs]))
            for s in range(self.runner.max_swarm_batches):
                try:
                    for si in range(self.runner.swarm_batch_size):
                        swarmtraj = stringprocessor.load_swarm(self.runner, i, s, si,
                                                               ignore_missing_files=self.ignore_missing_files,
                                                               fallback_on_restrained_output=True)
                        swarmtraj = swarmtraj[[0, -1]]
                        if swarmtraj is None:
                            logger.debug("Missing swarm traj for swarm %s point %s", s, i)
                            break
                        # TODO in case of many intermediates, get them as well
                        swarmcvs = colvars.eval_cvs(swarmtraj, self.cvs)
                        swarm_values.append(swarmcvs)
                except IOError:
                    if len(swarm_values) > 0:
                        # We probably didn't have this many swarms... Just break
                        break
                    else:  # No data found at all for this string
                        return None
        res = np.empty((len(swarm_values), 2, len(self.cvs)))
        for index, swarmcvs in enumerate(swarm_values):
            res[index, 0] = swarmcvs[0]
            res[index, -1] = swarmcvs[-1]
        return res

    def compute_transition_count(self, swarm_values, extra_transitions):
        transition_count = np.zeros((self._indexconverter.nbins, self._indexconverter.nbins))  # Transition per bin
        swarm_values = self._init_grid_params(swarm_values, extra_transitions)
        kernel_type = "gaussian"
        if self.smear_transitions:
            logger.debug("Using kernel type %s for smearing", kernel_type)
        for swarmcvs in swarm_values:
            if self.smear_transitions:
                self._smear_transitions(swarmcvs, transition_count, kernel_type)
            else:
                self._add_transition(swarmcvs, transition_count)
        return transition_count

    def _init_grid_params(self, swarm_values, extra_transitions):
        if extra_transitions is not None:
            logger.debug("Appending %s extra transition to %s swarm values. ", extra_transitions.shape,
                         swarm_values.shape)
            swarm_values = np.append(swarm_values, extra_transitions, axis=0)
            logger.debug("Done. New shape of swarm transitions: %s ", swarm_values.shape)
        if self.normalize_grid:
            # Normalize swarms along every cv to values between 0 and 1
            for i in range(swarm_values.shape[2]):
                sv = swarm_values[:, :, i]
                swarmmin = sv.min()
                swarmmax = sv.max()
                swarm_values[:, :, i] = (sv - swarmmin) / (swarmmax - swarmmin)
        if len(swarm_values) > 0:
            margin = 0.33 if self.smear_transitions else 0.1
            if self.gridmax is None:
                gridmax = swarm_values.max()
                self.gridmax = (1 - margin) * gridmax if gridmax < 0 else (1 + margin) * gridmax
                logger.debug("Setting gridmax to %s", self.gridmax)
            if self.gridmin is None:
                gridmin = swarm_values.min()
                self.gridmin = (1 - margin) * gridmin if gridmin > 0 else (1 + margin) * gridmin
                logger.debug("Setting gridmin to %s", self.gridmin)
            self.bin_width = FECalculator.compute_bin_width(self.gridmin, self.gridmax, self.ngrid)
        return swarm_values

    def compute_probability_distribution_msm(self, transition_count, bayesian=True):
        """
        Uses a transition matrix which obeys detailed balance and estimate the error with a bayesian approach
        See http://msmbuilder.org/3.8.0/_msm/msmbuilder.msm.MarkovStateModel.html#msmbuilder.msm.MarkovStateModel.fit
        """
        if self.smear_transitions:
            logger.warn("Smeared transitions not recommended when you use 'msm'")
        n_states = len(transition_count)
        sequences = []
        multiplicator = 100 if self.smear_transitions else 1
        for i in range(n_states):
            for j in range(n_states):
                transition = [i, j]
                for t in range(utils.rint(multiplicator * transition_count[i, j])):
                    sequences.append(transition)
        if bayesian:
            n_transition_matrices = 1000 if self._ndim == 1 else 1000
            logger.info("Computing %s transition matrices with a bayesian approach", n_transition_matrices)
            mm = msm.BayesianMarkovStateModel(lag_time=1, n_samples=n_transition_matrices)
            mm.fit(sequences)
            # TODO analyze timescales here
        else:
            n_transition_matrices = 1
            mm = msm.MarkovStateModel(lag_time=1)  # lag time here is the step between frames in "sequences" below
            mm.fit(sequences)
            self._analyze_timescales(mm)
        # wrap things up and compute error
        stationary_solution = np.empty((n_transition_matrices, n_states,))
        for i in range(n_states):
            # Convert from mm internal indices back to our indices
            mm_index = mm.mapping_.get(i, None)
            if mm_index is None:
                stationary_solution[:, i] = np.nan
            else:
                stationary_solution[:, i] = mm.all_populations_[:, mm_index] if bayesian else mm.populations_[mm_index]
        return stationary_solution

    def _analyze_timescales(self, mm):
        mfpts = tpt.mfpts(mm) * self.lagtime  # mean first passage times
        logger.info(
            "Means first passage time from last to first state and vice versa: %s, %s",
            mfpts[0, -1],
            mfpts[-1, 0])
        if len(mfpts) > 0:
            logger.info("Max first passage time: %s", mfpts.max())
        timescales = mm.timescales_
        timescales[np.isnan(timescales)] = -1
        timescales = timescales * self.lagtime
        if len(timescales) > 0:
            logger.info("Max relaxation time for the system %s", timescales.max())

    def compute_probability_distribution_hmm(self, transitions):
        """
        Uses a hmm model to generate a transition matrix which obeys detailed balance.
        see http://msmbuilder.org/3.8.0/_hmm/msmbuilder.hmm.GaussianHMM.html#msmbuilder.hmm.GaussianHMM
        """
        if self.smear_transitions:
            raise Exception("Smeared transitions not supported for hmm (the hmm implicitly does the smearing for you)")
        mm = hmm.GaussianHMM(n_states=self.ngrid)
        mm.fit(transitions)
        self._analyze_timescales(mm)
        return mm.populations_

    def compute_probability_distribution_eigenvector(self, transition_count):
        """
        Find the eigenvector (s) of the transition matrix
        :param transition_count:
        :return:
        """
        transition_probability = np.zeros(transition_count.shape)
        transition_count = self._remove_transitions_to_isolated_bins(transition_count)
        for rowidx, row in enumerate(transition_count):
            # transition_probability[rowidx, rowidx] = 0
            rowsum = np.sum(row)
            if rowsum > 0:
                transition_probability[rowidx] = row / rowsum
        eigenvalues, eigenvectors = np.linalg.eig(transition_probability.T)
        stationary_solution = None
        unit_eigenval = None  # The eigenvalue closest to 1
        for idx, eigenval in enumerate(eigenvalues):
            vec = eigenvectors[:, idx]
            # logger.debug("Eigenvec for eigenvalue %s:\n%s", eigenval, vec)
            if np.isclose(1., eigenval, rtol=1e-2):
                neg_vec, pos_vec = vec[vec < 0], vec[vec > 0]
                if len(pos_vec) == 0:
                    # No positive entries. All must be negative. We can multiply the eigenvector by a factor of -1
                    vec = -1 * vec
                elif len(neg_vec) > 0:
                    logger.warn("Found a vector with eigenvalue ~1(%s) but with negative entries in its eigenvector",
                                eigenval)
                    continue
                if stationary_solution is not None:
                    raise Exception(
                        "Multiple stationary solutions found. Perhaps there were no transitions between states. Eigenvalues:\n%s" % eigenvalues)
                vec = np.real(vec)
                stationary_solution = vec / np.sum(vec)
                unit_eigenval = eigenval
        relaxation_eigenval = None  # corresponds to the largest eigenvalue less than 1
        for idx, eigenval in enumerate(eigenvalues):
            if eigenval < 1 and eigenval != unit_eigenval:
                if relaxation_eigenval is None or eigenval > relaxation_eigenval:
                    relaxation_eigenval = eigenval
        if stationary_solution is None:
            raise Exception("No stationary solution found. Eigenvalues:\n%s", eigenvalues)
        if relaxation_eigenval is not None:
            logger.info("Relaxation time for system: %s (s). Eigenval=%s", -np.log(relaxation_eigenval) / self.lagtime,
                        relaxation_eigenval)
        return stationary_solution

    def compute_probability_distribution_detailed_balance(self, transition_count):
        nbins = transition_count.shape[0]
        transition_probability = np.zeros(transition_count.shape)
        rho = np.zeros((nbins,))
        inaccessible_states, nonstarting_states = 0, 0
        transition_count = self._remove_transitions_to_isolated_bins(transition_count)
        for rowidx, row in enumerate(transition_count):
            # transition_probability[rowidx, rowidx] = 0
            rowsum = np.sum(row)
            if rowsum > 0:
                # transition_probability[rowidx, rowidx] = 0
                transition_probability[rowidx] = row / rowsum
                # transition_probability[rowidx, rowidx] = -rowsum
            else:
                rho[rowidx] = np.nan

        if inaccessible_states > 0 or nonstarting_states > 0:
            logger.warn("Found %s inaccessible states and %s states with no starting points.", inaccessible_states,
                        nonstarting_states)
        # Set up starting guess for distribution: all accessible states equal
        for i, rhoi in enumerate(rho):
            if np.isnan(rhoi):
                rho[i] = 0
            else:
                rho[i] = 1
        rho = rho / np.sum(rho)
        convergences = []
        convergence = 100
        while convergence > 1e-5:
            last = rho  # np.copy(rho)
            for k, rhok in enumerate(rho):
                if rhok == 0:
                    continue
                crossterm = np.dot(rho, transition_probability[:, k]) - np.sum(rhok * transition_probability[k, :])
                # crossterm = 0
                # for l, rhol in enumerate(rho):
                #     if l != k:
                #         crossterm += rhol * transition_probability[l, k] - rhok * transition_probability[k, l]
                if rhok == 0 and crossterm > 0:
                    logger.warn("NOOOO for index %s. Crossterm %s rhok %s", k, crossterm, rhok)
                rho[k] = rhok + crossterm
            rho = rho / np.sum(rho)
            if last is not None:
                convergence = np.linalg.norm(rho - last)
                convergences.append(convergence)
        logger.debug("Converged with master equation after %s iterations", len(convergences))
        # plt.plot(convergences, label="Convergence")
        # plt.show()
        if len(rho[rho == 0]) < inaccessible_states:
            raise Exception("Something went wrong. Number inaccessible states differ %s vs. %s" % (len(rho[rho == 0]),
                                                                                                   inaccessible_states))
        result = np.zeros((1, nbins,))
        result[0] = rho  # minor fix to return correct format. TODO use np function to extend dimensions if necessary
        return result

    def _remove_transitions_to_isolated_bins(self, transition_count):
        """Remove all transitions which moves from a bin with no starting points"""
        last_inaccessible_states, last_nonstarting_states = -1, -1
        inaccessible_states, nonstarting_states = 1, 1
        while inaccessible_states != last_inaccessible_states or nonstarting_states != last_nonstarting_states:
            last_inaccessible_states, last_nonstarting_states = inaccessible_states, nonstarting_states
            inaccessible_states, nonstarting_states, accessible_states = 0, 0, 0
            for rowidx, row in enumerate(transition_count):
                rowsum = np.sum(row)
                if rowsum == 0:
                    # transition_probability[rowidx] = 0
                    if np.sum(transition_count[:, rowidx]) == 0:
                        # logger.warn("Found inaccessible state at index %s ", rowidx)
                        inaccessible_states += 1
                        # rho[rowidx] = np.nan
                    else:
                        # logger.warn("Found non-starting states")
                        nonstarting_states += 1
                    # TODO see if this makes sense: to set all transition into this state to zero to completely isolate it!
                    transition_count[:, rowidx] = 0
                else:
                    accessible_states += 1
        if inaccessible_states > 0 or nonstarting_states > 0:
            logger.warn("Found %s accessible states, %s inaccessible states and %s states with no starting points.",
                        accessible_states,
                        inaccessible_states,
                        nonstarting_states)
        return transition_count

    def compute_probability_distribution_iter(self, transition_count):
        """
        Find stationary distribution by starting from an initial guess of equal distribution until convergence
        This does not guarantee detailed balance. Use the other method for that, but the results should be the same.
        :param transition_count:
        :return:
        """
        nbins = transition_count.shape[0]
        transition_probability = np.zeros(transition_count.shape)
        rho = np.zeros((nbins,))
        inaccessible_states, nonstarting_states = 0, 0
        transition_count = self._remove_transitions_to_isolated_bins(transition_count)
        for rowidx, row in enumerate(transition_count):
            # transition_probability[rowidx, rowidx] = 0
            rowsum = np.sum(row)
            if rowsum > 0:
                # transition_probability[rowidx, rowidx] = 0
                transition_probability[rowidx] = row / rowsum
                # transition_probability[rowidx, rowidx] = -rowsum
            else:
                rho[rowidx] = np.nan

        if inaccessible_states > 0 or nonstarting_states > 0:
            logger.warn("Found %s inaccessible states and %s states with no starting points.", inaccessible_states,
                        nonstarting_states)
        # Set up starting guess for distribution: all accessible states equal
        for i, rhoi in enumerate(rho):
            if np.isnan(rhoi):
                rho[i] = 0
            else:
                rho[i] = 1
        rho = rho / np.sum(rho)
        convergences = []
        convergence = 100
        while convergence > 1e-5:
            last = rho  # np.copy(rho)
            rho = np.matmul(rho.T, transition_probability)
            rho = rho / np.sum(rho)  # probabably not necessary
            if last is not None:
                convergence = np.linalg.norm(rho - last)
                convergences.append(convergence)
        logger.debug("Converged to stationary distribution after %s iterations", len(convergences))
        # plt.plot(convergences, label="Convergence")
        # plt.show()
        if len(rho[rho == 0]) < inaccessible_states:
            raise Exception("Something went wrong. Number inaccessible states differ %s vs. %s" % (len(rho[rho == 0]),
                                                                                                   inaccessible_states))
        return rho

    def compute_probability_distribution(self, transition_count, method, transitions):
        if method == None:
            logger.warn("No method for computing probability distribution. Returning an array of not a numbers (nan)")
            return np.zeros((transition_count.shape[0],)) + np.nan
        elif method.startswith("msm"):
            return self.compute_probability_distribution_msm(transition_count, bayesian=method == "msm_bayesian")
        elif method == "db":  # detailed balance
            return self.compute_probability_distribution_detailed_balance(transition_count)
        elif method == "eig":
            logger.warn("Using deprecated method %s", method)
            return self.compute_probability_distribution_eigenvector(transition_count)
        elif method == "iter":
            logger.warn("Using deprecated method %s", method)
            return self.compute_probability_distribution_iter(transition_count)
        elif method == "hmm":
            logger.warn("Using no so very well tested method %s", method)
            return self.compute_probability_distribution_hmm(transitions)
        else:
            raise NotImplementedError(method)

    def compute(self, load_swarm_values=None, load_transition_count=False, method="db",
                transition_frame_loader=None):
        """
        :param load_swarm_values:
        :param load_transition_count:
        :param method:
        :return: the free energy, the probability density (both as a grid)
        """
        logger.info("Computing swarms with CVs %s and FE for CV indices %s",
                    [extra_analysis.get_cv_description(cv.id) for cv in self.cvs], self.cv_indices)
        swarm_values = self.compute_swarm_values(load_swarm_values)
        logger.debug("Loaded swarm cv values of shape %s", swarm_values.shape)
        swarm_values = swarm_values[:, :, self.cv_indices]  # extract the dimensions we want
        if self.dependent_cvs is not None:
            swarm_values = colvars.eval_transitions(swarm_values, self.dependent_cvs, self.cvs)
        logger.debug("Changes swarm cv values to shape %s", swarm_values.shape)
        if transition_frame_loader is None:
            extra_transitions = None
        else:
            extra_transitions = transition_frame_loader.compute(cvs=self.cvs, cv_indices=self.cv_indices,
                                                                save=self.save, load=load_swarm_values)
        swarm_suffix, transition_count_suffix = self._get_suffix(
            swarm_values)  # TODO pass extra transition as parameter
        transition_count_path = self.runner.working_dir + "../transition_grids/transition" + transition_count_suffix
        transition_count = None
        if method == "hmm":
            transition_count = []
        elif load_transition_count in [True, None]:
            if load_transition_count is True and not os.path.exists(transition_count_path + ".npy"):
                raise Exception("File not found %s" % transition_count_path)
            elif os.path.exists(transition_count_path + ".npy"):
                transition_count = np.load(transition_count_path + ".npy")
                self._init_grid_params(swarm_values, extra_transitions=extra_transitions)
        if transition_count is None:
            transition_count = self.compute_transition_count(swarm_values, extra_transitions)
            if self.save and self.smear_transitions:  # Only save smeared transitions since that is what takes time
                np.save(transition_count_path, transition_count)
        if self.plot:
            self.plot_transition_counts(transition_count)
        probability_distributions = self.compute_probability_distribution(transition_count, method, swarm_values)
        # logger.debug("Probability distribution: %s", probability_distribution)
        fe = np.empty(probability_distributions.shape)
        for dist_idx, probability_distribution in enumerate(probability_distributions):
            for bin_idx, p in enumerate(probability_distribution):
                if p is None or np.isnan(p) or p <= 1e-7:
                    fe[dist_idx, bin_idx] = sys.float_info.max  # np.nan
                else:
                    fe[dist_idx, bin_idx] = -FECalculator.kB * FECalculator.T * np.log(p)
        return self._indexconverter.convert_to_grid(fe), \
               self._indexconverter.convert_to_grid(probability_distributions) / self.bin_width, \
               transition_count

    def _add_transition(self, swarmcvs, transition_count, weight=1):
        start_bin = self._find_bin(swarmcvs[0])
        end_bin = self._find_bin(swarmcvs[-1])
        transition_count[int(start_bin), int(end_bin)] += weight

    def _smear_transitions(self, swarmcvs, transition_count, kernel_type):
        """Smear to nearest neightbours"""
        ncvs = len(self.cv_indices) if self.dependent_cvs is None else len(self.dependent_cvs)
        start, end = swarmcvs[0], swarmcvs[-1]
        sigma = self.bin_width / 4
        if kernel_type == "gaussian":
            bins_to_smear = 4
            gaussian = lambda center, point, sigma: 1 / (2 * np.sqrt(2 * np.pi)) * np.exp(
                -(np.linalg.norm(center - point) / sigma) ** 2 / 2)
            kernel = gaussian
        elif kernel_type == "cubic_spline":
            bins_to_smear = 4
            kernel = cubic_spline
        elif kernel_type == "simple":
            bins_to_smear = 2

            # TODO not a good model in 2D. Use a hypersphere for smoothing or something. See PIC
            def simple_kernel(center, point, sigma):
                # print(center, point)
                dist = np.linalg.norm(point - center) / (self.bin_width)
                return np.max(1 - dist, 0)

            kernel = simple_kernel
        else:
            raise Exception("TODO")
        startpoints, endpoints = self._to_smeared_grid(start, bins_to_smear, ncvs), self._to_smeared_grid(end,
                                                                                                          bins_to_smear,
                                                                                                          ncvs)
        # logger.debug("Smearing using %s kernel", kernel_type)
        for startval in startpoints:
            if (startval > self.gridmax).any() or (startval < self.gridmin).any():
                continue
            wstart = kernel(start, startval, sigma)
            for endval in endpoints:
                if (endval > self.gridmax).any() or (endval < self.gridmin).any():
                    continue
                wend = kernel(end, endval, sigma)
                cv_vals = np.empty((2, ncvs))
                cv_vals[0] = startval
                cv_vals[1] = endval
                # print(startval, endval, self.gridmax, self.gridmin, self.ngrid)
                self._add_transition(cv_vals, transition_count, weight=wstart * wend)

    def _to_smeared_grid(self, center, bins_to_smear, ncvs):
        smearing_bins = np.empty((ncvs, bins_to_smear + 1))
        for i in range(ncvs):
            smearing_bins[i] = np.array(
                [center[i] - j * self.bin_width for j in
                 range(-int((bins_to_smear / 2)), int(bins_to_smear / 2 + 1))])
        # Avoiding iterations as much as possible below
        points = np.stack(np.meshgrid(*smearing_bins), -1).reshape(-1, ncvs)
        return points

    def _find_bin(self, evals):
        grid_coord = np.empty(evals.shape)
        for idx, value in enumerate(evals):
            grid_coord[idx] = (value - self.gridmin) / self.bin_width
        return self._indexconverter.convert_to_bin_idx(grid_coord.astype(int))

    def plot_transition_counts(self, transition_count):
        plt.subplot(*(3, 1, 1))
        start_states = self._indexconverter.convert_to_grid(transition_count.sum(axis=1))
        if len(start_states.shape) > 1:
            plt.contourf(start_states)
            plt.contour(start_states)
            plt.title("Start states")
            plt.colorbar()
            plt.grid()
            plt.subplot(*(3, 1, 2))
            end_states = self._indexconverter.convert_to_grid(transition_count.sum(axis=0))
            plt.contourf(end_states)
            plt.contour(end_states)
            plt.title("End states")
            plt.colorbar()
            plt.grid()
            plt.subplot(*(3, 1, 3))
            fraction = end_states / (start_states + 1e-10)
            fraction[fraction > 1e2] = np.nan
            plt.contourf(fraction)
            plt.contour(fraction)
            plt.title("fraction")
            plt.colorbar()
            plt.grid()
            plt.show()

    def plot_swarm_values(self, swarm_values):
        start_states = swarm_values[:, 0, :].T
        end_states = swarm_values[:, 1, :].T
        idx1, idx2 = 0, 1
        twoD = start_states.shape[0] > 1
        if twoD:
            plt.scatter(start_states[idx1], start_states[idx2], label="Start states", alpha=0.1)
            plt.scatter(end_states[idx1], end_states[idx2], label="End states", alpha=0.1)
            plt.ylabel(extra_analysis.get_cv_description(self.cvs[idx2].id))
        else:
            start, end = start_states[idx1], end_states[idx1]
            plt.hist([start, end],
                     label=["Start states", "end states"], alpha=0.3,
                     bins=self.ngrid - 1)
            # plt.hist(end_states[idx1], label="End states", alpha=0.3)
            plt.title(self.runner.simu_id)
            plt.ylabel("Count")
        plt.xlabel(extra_analysis.get_cv_description(self.cvs[idx1].id))
        plt.grid()
        plt.legend()
        plt.show()

    @classmethod
    def compute_bin_width(cls, gridmin, gridmax, ngrid):
        return (gridmax - gridmin) / (ngrid - 1)

    def _get_suffix(self, swarm_values=None):
        swarm_suffix, transition_count_suffix = "", "_ngrid%s%s" % (
            self.ngrid, "_smeared" if self.smear_transitions else "")
        for idx, cv in enumerate(self.cvs):
            swarm_suffix += "_" + extra_analysis.get_cv_description(cv.id)
            if idx in self.cv_indices:
                transition_count_suffix += extra_analysis.get_cv_description(cv.id)
        if self.gridmin is not None:
            transition_count_suffix += "_gridmin%s" % self.gridmin
        if self.gridmax is not None:
            transition_count_suffix += "_gridmax%s" % self.gridmax
        if swarm_values is not None:
            transition_count_suffix += "_" + str(hash(swarm_values.tostring()))
        return swarm_suffix, transition_count_suffix

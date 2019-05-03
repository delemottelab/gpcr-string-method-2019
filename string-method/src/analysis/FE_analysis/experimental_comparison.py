from __future__ import absolute_import, division, print_function

from analysis.FE_analysis.visualization import *

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
import scipy
import sklearn.mixture as mixture

logger = logging.getLogger("expcomp")


class FRETConverter(object):
    @staticmethod
    def convert_to_fret_efficiency(runner, free_energies, probability_density, bin_distances,
                                   help_plots=False):
        """
        Assuming 1D
        Plots extracted from paper to datasets with https://automeris.io/WebPlotDigitizer/
        :param free_energy:
        :param probability_density:
        :param bin_distances: distance in [nm]
        :param distance_offset a fixed offset to handly dye distance. TODO: load from function
        :return:
        """
        probability_density = np.mean(probability_density, axis=0)
        free_energies = np.mean(free_energies, axis=0)
        simu_id = runner.simu_id
        data = load_numpy_csv('Cy3B-Cy7_fret_efficiency.csv')
        # Distance is in Angstrom, conver to nm
        data[:, 0] = data[:, 0] * 1e-1
        ligand_type = "apo" if "apo" in runner.simu_id else "ISO"
        paper_fret = load_numpy_csv('paper_{}_Cy3B-Cy7_fret_efficiency.csv'.format(ligand_type))
        # Compute fluorophore prob density as a function of fluorophore distance.
        bin_distances, probability_density = convert_to_fluorophore_distances(bin_distances, probability_density,
                                                                              help_plots=help_plots)
        bins_to_use = []
        for idx, prob in enumerate(probability_density):
            dist = bin_distances[idx]
            # remove irrelevant probability values and
            # make sure we only use values within the interval of fret efficiency for which we have data
            if not np.isnan(prob) and prob > 0 and dist >= data[:, 0].min() and dist <= data[:, 0].max():
                bins_to_use.append(idx)
        # Create function and interpolate FRET efficiency
        fret_efficiency = scipy.interpolate.interp1d(data[:, 0], data[:, 1], kind="quadratic")
        simu_fret = fret_efficiency(bin_distances[bins_to_use])
        prob_func = scipy.interpolate.interp1d(simu_fret, probability_density[bins_to_use], kind="quadratic")
        # Smoothen the curve a bit and make it equidistant
        equidistant_range = np.arange(simu_fret.min(), simu_fret.max(), (simu_fret.max() - simu_fret.min()) / 100)
        # Plotting
        # Offset data to be zero at minimum
        # paper_fret[:, 1] -= paper_fret[:, 1].min()
        paper_prob = paper_fret[:, 1]
        paper_prob[paper_prob <= 0] = 0
        paper_prob /= (paper_fret[1, 0] - paper_fret[0, 0]) * np.sum(paper_prob)  # normalize
        plt.plot(paper_fret[:, 0], paper_prob, '--',
                 label="Experimental {}".format(simuid_to_label.get(simu_id, simu_id)), linewidth=linewidth,
                 color=simuid_to_color.get(simu_id), alpha=0.5)
        plt.plot(equidistant_range, prob_func(equidistant_range),
                 label=simuid_to_label.get(simu_id, simu_id), linewidth=linewidth, color=simuid_to_color.get(simu_id))
        plt.xlabel("FRET efficiency", fontsize=label_fontsize)
        plt.ylabel("Probability density", fontsize=label_fontsize)
        plt.legend(fontsize=label_fontsize)
        plt.grid()
        # plt.title("FRET comparison {}".format(ligand_type))
        plt.tick_params(labelsize=ticks_labelsize)
        if help_plots:
            generated_samples = []
            for i, p in enumerate(prob_func(equidistant_range)):
                f = equidistant_range[i]
                for j in range(utils.rint(p * 1000)):
                    generated_samples.append(f)
            plt.plot(paper_fret[:, 0], paper_fret[:, 1], label="Experimental", alpha=0.5)
            plt.hist(generated_samples, bins=int(len(equidistant_range) / 4), normed=True, label="Simulation",
                     linewidth=linewidth, color=simuid_to_color.get(simu_id))
            plt.grid()
            plt.legend()
            plt.xlabel("FRET efficiency", fontsize=label_fontsize)
            plt.ylabel("Count %", fontsize=label_fontsize)
            plt.tick_params(labelsize=ticks_labelsize)
            plt.show()
        return simu_fret


def convert_to_fluorophore_distances(CA_distances, CA_probability_density, help_plots=False, previous_gmm=[],
                                     offset_type="avg"):
    if offset_type == "fixed":
        return CA_distances + 1.2, CA_probability_density  # (OLD) FIXED offset which gives decent results
    # Load distances i angstrom for fluorophores and CAlphas as a function of time in ns
    time_CA = load_numpy_csv("CA_distance_Angstrom_ns.csv")
    time_fluorophore = load_numpy_csv("fluorophore_distance_Angstrom_ns.csv")
    # conver to nm
    time_CA[:, 1] /= 10 - 0.05
    time_fluorophore[:, 1] /= 10
    # Extrapolate the functions so that they have the same time values
    min_t = max(time_CA[:, 0].min(), time_fluorophore[:, 0].min())
    max_t = min(time_CA[:, 0].max(), time_fluorophore[:, 0].max())
    all_times = np.arange(min_t, max_t, (max_t - min_t) / (20 * (len(time_CA) + len(time_fluorophore))))
    interpolation_kind = "quadratic"
    time_CA_func = scipy.interpolate.interp1d(time_CA[:, 0], time_CA[:, 1], kind=interpolation_kind)
    time_fluorophore_func = scipy.interpolate.interp1d(time_fluorophore[:, 0], time_fluorophore[:, 1],
                                                       kind=interpolation_kind)
    X = np.empty((len(all_times), 2))
    X[:, 0] = time_CA_func(all_times)
    X[:, 1] = time_fluorophore_func(all_times)
    dist_mean = (X[:, 1] - X[:, 0]).mean()
    dist_std = (X[:, 1] - X[:, 0]).std()
    if offset_type == "avg":
        # return CA_distances + dist_mean, CA_probability_density  # (OLD) offset set to avg value of time series
        return CA_distances + (time_fluorophore[:, 1].mean() - time_CA[:, 1].mean()), CA_probability_density
    # if help_plots:
    #     # Show distribution of data
    #     plt.scatter(X[:, 0], X[:, 1])
    #     plt.show()
    # Create gaussian mixture model. Since we don't know the number of GMM components we use a variational method
    ##kSee http://scikit-learn.org/stable/modules/mixture.html#bgmm and Variational Bayesian Gaussian Mixture
    # n_components is an upper bound for the number of components. Not all will necessarily be used
    if len(previous_gmm) == 0:
        gmm = mixture.BayesianGaussianMixture(n_components=1, max_iter=500)
        gmm.fit(X)
        previous_gmm.append(gmm)
    else:
        gmm = previous_gmm[0]
        logger.info("GMM previously computed")
    # Iterate over the x in CA_distances. Every x maps to a probability distribution of fluorophore distances f(x).
    # Now we need to map the probability density, p, of x to a new probability of fluorophore distance. this is done by integrating p' =p(x)f(x)
    # Every x add to p'
    fluo_dist_min, fluo_dist_max = X[:, 1].min(), X[:, 1].max()
    fluorophore_distances = np.arange(fluo_dist_min, fluo_dist_max,
                                      (fluo_dist_max - fluo_dist_min) / len(CA_distances))
    fluorophore_probability_density = np.zeros(fluorophore_distances.shape)
    for idx_CA, x in enumerate(CA_distances):
        CA_prob = CA_probability_density[idx_CA]
        if np.isnan(CA_prob):
            continue
        for idx_fluo, y in enumerate(fluorophore_distances):
            y_prob = np.exp(gmm.score_samples(np.matrix([x, y])))
            # Just us a normal distribution below:
            # y_prob = np.exp(-(y - x - dist_mean) ** 2 / (2 * dist_std)) / (np.sqrt(2 * np.pi * dist_std ** 2))
            fluorophore_probability_density[idx_fluo] += CA_prob * y_prob
    fluorophore_probability_density /= sum(
        fluorophore_probability_density * (fluorophore_distances[1] - fluorophore_distances[0]))
    if help_plots:
        plt.plot(fluorophore_distances, fluorophore_probability_density, label="fluorophores")
        plt.plot(CA_distances, CA_probability_density, '--', label="CA")
        plt.ylabel("Probability density")
        plt.xlabel("Distance")
        plt.title("#Components={}, interpolation method={}".format(gmm.n_components, interpolation_kind))
        plt.legend()
        plt.grid()
        plt.show()

    return fluorophore_distances, fluorophore_probability_density


def load_numpy_csv(file, dir="../gpcr/experimental_data/nature22354_2018/"):
    return np.genfromtxt(dir + file, delimiter=',')

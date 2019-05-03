from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.helpfunc import *

logger = logging.getLogger(__name__)


def eval_cvs(traj, cvs, rescale=False):
    res = np.empty((len(traj), len(cvs)))
    for i, cv in enumerate(cvs):
        res[:, i] = np.squeeze(cv.eval(traj))
    return rescale_evals(res, cvs) if rescale else res


def rescale_evals(evals, cvs):
    if len(evals.shape) == 1:
        return cvs[0].rescale(evals)
    res = np.empty(evals.shape)
    for i, cv in enumerate(cvs):
        res[:, i] = cv.rescale(evals[:, i])
    return res


def scale_evals(evals, cvs):
    """The opposite of rescale_evals"""
    if len(evals.shape) == 1:
        return cvs[0].scale(evals)
    res = np.empty(evals.shape)
    for i, cv in enumerate(cvs):
        res[:, i] = cv.scale(evals[:, i])
    return res


def eval_transitions(transitions, dependent_cvs, original_cvs):
    result = np.empty((len(transitions), 2, len(dependent_cvs)))
    result[:, 0, :] = eval_cvs(transitions[:, 0, :], dependent_cvs)
    result[:, 1, :] = eval_cvs(transitions[:, 1, :], dependent_cvs)
    return result


def normalize_cvs(cvs, simulations=None, trajs=None):
    if simulations is not None and trajs is None:
        trajs = [s.traj for s in simulations]
    if trajs is not None:
        for cv in cvs:
            cv.normalize(trajs)
    return cvs


def rescale_points(cvs, points):
    if len(points.shape) == 1:
        return np.array([cv.rescale(p) for cv, p in zip(cvs, points)])
    else:
        res = np.empty(points.shape)
        for i, point in enumerate(points):
            for j, cv in enumerate(cvs):
                res[i, j] = cv.rescale(point[j])
        return res


def scale_points(cvs, points):
    """THe opposite of rescale_points"""
    if len(points.shape) == 1:
        return np.array([cv.scale(p) for cv, p in zip(cvs, points)])
    else:
        res = np.empty(points.shape)
        for i, point in enumerate(points):
            for j, cv in enumerate(cvs):
                res[i, j] = cv.scale(point[j])
        return res


def crosscorrelate_cvs(simu, cvs1, cvs2, plot_limit=0, output=True, number_simus=1):
    def compute_correlation(eval1, eval2):
        return np.corrcoef(eval1, eval2)[0, 1]

    def correlation_sort(cvs, correlations):
        return sorted(
            zip(cvs, correlations),
            cmp=lambda (cv1, corr1), (cv2, corr2): int(1000 * (corr1 - corr2))
        )

    all_corr = []
    max_correlations_cvs2 = np.zeros(
        (len(cvs2),), )
    evals1 = eval_cvs(simu.traj, cvs1)
    evals2 = eval_cvs(simu.traj, cvs2)
    index = 1
    for i, cvi in enumerate(cvs1):
        val1 = evals1[:, i]
        max_corr = None
        for j, cvj in enumerate(cvs2):
            val2 = evals2[:, j]
            corr = compute_correlation(val1, val2)
            label = cvi.id + "-" + cvj.id
            index += 1
            all_corr.append((cvi, cvj, corr))
            if abs(corr) > max_correlations_cvs2[j]:
                max_correlations_cvs2[j] = abs(corr)
    if output:
        table = "cv1\tcv2\t\t\tcorrelation\n"
        # Find those with correlation close to zero, these are uncorrelated!
        for cvi, cvj, corr in sorted(all_corr,
                                     cmp=lambda (cv11, cv12, e1), (cv21, cv22, e2): int(10000 * (abs(e1) - abs(e2)))):
            table += "%s\t%s\t%s\n" % (cvi.id, cvj.id, corr)
        # logger.info("Correlation table for all correlations\n%s", table)
        # take the second set of CVs with the lowest correlation to all the first set of CVs
        table = "cv\t\t\tmax corr\tDesc.\n"
        # iterate through max_corr and change the correlation if they are internally correlated
        for i in range(len(max_correlations_cvs2)):
            val1 = evals2[:, i]
            for j in range(i + 1, len(max_correlations_cvs2)):
                corri, corrj = max_correlations_cvs2[[i, j]]
                val2 = evals2[:, j]
                cross_corr = abs(compute_correlation(val1, val2))
                if corri < corrj:
                    if cross_corr > corrj:
                        # this means that these two are internally more correlated
                        # since corri is more unique, we update corrj to a higher correlation value
                        max_correlations_cvs2[j] = cross_corr
                elif cross_corr > corri:
                    max_correlations_cvs2[i] = cross_corr
        max_corr = correlation_sort(cvs2, max_correlations_cvs2)
        ticks = np.arange(0, len(evals2), len(evals2) / number_simus)
        for index, (cv, corr) in enumerate(max_corr):
            # TODO check if the discovered DOF correlate with eachother!
            table += "%s\t%s\t%s\n" % (cv.id, corr, cv)
            if index < plot_limit:
                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(plot_limit, 1, index + 1)
                ax.set_xticks(ticks)
                plt.plot(evals1, '--', alpha=0.3)
                plt.title("Max correlation=%s" % corr)
                cv_vals = eval_cvs(simu.traj, [cv])
                plt.plot(cv_vals, color="black", alpha=0.5)
                plt.legend([c.id for c in cvs1] + [cv.id])
                # plt.ylabel("Normalized value")
                plt.grid()
        logger.info("Correlation table for max correlation per cv\n%s", table)
        if plot_limit > 0:
            plt.show()
    return max_corr, np.array(all_corr)

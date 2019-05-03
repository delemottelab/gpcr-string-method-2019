import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import (
    _init_centroids,
    _labels_inertia,
    _tolerance,
    _validate_center_shape,
)
from sklearn.utils import (
    check_array,
    check_random_state,
    as_float_array,
)
from sklearn.cluster import _k_means
from sklearn.preprocessing import normalize
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.extmath import row_norms, squared_norm


def _spherical_kmeans_single_lloyd(X, n_clusters, max_iter=300,
                                   init='k-means++', verbose=False,
                                   x_squared_norms=None,
                                   sample_weights=None,
                                   random_state=None, tol=1e-4,
                                   precompute_distances=True):
    '''
    Modified from sklearn.cluster.k_means_.k_means_single_lloyd.
    '''
    random_state = check_random_state(random_state)

    best_labels, best_inertia, best_centers = None, None, None

    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()

        # labels assignment
        # TODO: _labels_inertia should be done with cosine distance
        #       since ||a - b|| = 2(1 - cos(a,b)) when a,b are unit normalized
        #       this doesn't really matter.
        labels, inertia = \
            _labels_inertia(X, sample_weights, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=distances)
    
        # computation of the means
        if sp.issparse(X):
            centers = _k_means._centers_sparse(X, sample_weights, labels, n_clusters,
                                               distances)
        else:
            centers = _k_means._centers_dense(X, sample_weights, labels, n_clusters, distances)

        # l2-normalize centers (this is the main contibution here)
        centers = normalize(centers)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, x_squared_norms, best_centers,
                            precompute_distances=precompute_distances,
                            distances=distances)

    return best_labels, best_inertia, best_centers, i + 1


def spherical_k_means(X, n_clusters, init='k-means++', n_init=10,
            max_iter=300, verbose=False, tol=1e-4, random_state=None,
            copy_x=True, n_jobs=1, algorithm="auto", return_n_iter=False):
    """Modified from sklearn.cluster.k_means_.k_means.
    """
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    best_inertia = np.infty
    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = _spherical_kmeans_single_lloyd(
                X, n_clusters, max_iter=max_iter, init=init, verbose=verbose,
                tol=tol, x_squared_norms=x_squared_norms,
                random_state=random_state)

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_spherical_kmeans_single_lloyd)(X, n_clusters,
                                   max_iter=max_iter, init=init,
                                   verbose=verbose, tol=tol,
                                   x_squared_norms=x_squared_norms,
                                   # Change seed to ensure variety
                                   random_state=seed)
            for seed in seeds)

        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


class SphericalKMeans(KMeans):
    def __init__(self,  **kwargs):
        KMeans.__init__(self, **kwargs)


    def fit(self, X, y=None, **kwargs):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        random_state = check_random_state(self.random_state)
        #X = self._check_fit_data(X) #Removed from sklearn later versions
        X = KMeans.fit(self, X, y=y, sample_weight=None).transform(X)
        

        # TODO: add check that all data is unit-normalized

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            spherical_k_means(
                X, n_clusters=self.n_clusters, init=self.init,
                n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs,
                return_n_iter=True)

    	return self

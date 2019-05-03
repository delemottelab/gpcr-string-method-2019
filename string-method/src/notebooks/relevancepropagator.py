from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# imports
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

sys.path.append('MD_common/')
sys.path.append('heatmapping/')
# sys.path.append('/home/oliverfl/git
from utils.helpfunc import *
import modules
from notebooks.trajclassifier import transform_to_vector

"""
* **relevance propagation method** described at http://heatmapping.org/tutorial/

* **Some info on MLP** (from https://www.hiit.fi/u/ahonkela/dippa/node41.html):

The computations performed by such a feedforward network with a single hidden layer with nonlinear activation functions and a linear output layer can be written mathematically as

 $\displaystyle \mathbf{x}= \mathbf{f}(\mathbf{s}) = \mathbf{B}\boldsymbol{\varphi}( \mathbf{A}\mathbf{s}+ \mathbf{a} ) + \mathbf{b}$	(4.15)

where  $ \mathbf{s}$ is a vector of inputs and  $ \mathbf{x}$ a vector of outputs.  $ \mathbf{A}$ is the matrix of weights of the first layer,  $ \mathbf{a}$ is the bias vector of the first layer.  $ \mathbf{B}$ and  $ \mathbf{b}$ are, respectively, the weight matrix and the bias vector of the second layer. The function  $ \boldsymbol{\varphi}$ denotes an elementwise nonlinearity. The generalisation of the model to more hidden layers is obvious.

* **About the MLP implementation we use**:

If you do want to extract the MLP weights and biases after training your model, you use its public attributes coefs_ and intercepts_.
- coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1.
- intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.

"""


class Network(modules.Network):
    def relprop(self, R):
        for l in self.layers[::-1]:
            R = l.relprop(R)
        return R


class ReLU(modules.ReLU):
    def relprop(self, R):
        return R


class Linear(modules.Linear):
    def __init__(self, weight, bias):
        self.W = weight
        self.B = bias


class FirstLinear(Linear):
    """For z-beta rule"""

    def relprop(self, R):
        W, V, U = self.W, np.maximum(0, self.W), np.minimum(0, self.W)
        X, L, H = self.X, self.X * 0 + utils.lowest, self.X * 0 + utils.highest

        Z = np.dot(X, W) - np.dot(L, V) - np.dot(H, U) + 1e-9
        S = R / Z
        R = X * np.dot(S, W.T) - L * np.dot(S, V.T) - H * np.dot(S, U.T)
        return R


class NextLinear(Linear):
    """For z+ rule"""

    def relprop(self, R):
        #         logger.debug("NextLinear relprop for layer")
        V = np.maximum(0, self.W)
        Z = np.dot(self.X, V) + 1e-9
        S = R / Z
        C = np.dot(S, V.T)
        R = self.X * C
        return R


def create_layers(weights, biases, use_first_linear=True):
    layers = []
    for idx, weight in enumerate(weights):
        if idx == 0 and use_first_linear:
            l = FirstLinear(weight, biases[idx])
        else:
            l = NextLinear(weight, biases[idx])
        layers.append(l)
        if idx < len(weights) - 1:  # add to every layer except last
            layers.append(ReLU())
    return layers


def sensitivity_analysis(weights, biases, X, T):
    # TODO get rid of for loop and do all at once
    network = modules.Network(create_layers(weights, biases))
    Y = network.forward(X)
    gradprop = network.gradprop(T)
    S = gradprop ** 2
    return S


def relevance_propagation(weights, biases, X, T):
    # TODO get rid of for loop and do all at once
    # network = Network([ReLU() for coef in weights])
    network = Network(create_layers(weights, biases))
    Y = network.forward(X)
    D = network.relprop(Y * T)
    return D


"""Custom code made for finding relevance per clusters"""


def analyze_relevance(relevance,
                      sensitivity,
                      target_values,
                      plot=True,
                      max_scale=True):
    """
    Compute scaled average of relevance and sensitivity.
    If max_scale=True:
        -Find the average per cluster.
        -normalize to values 0 to 1
        -For every degree of freedom, take max value of all clusters
    """
    target_values = transform_to_vector(target_values)

    def max_scaled_average(values, target_values):
        res = []
        (clusters, cluster_count) = np.unique(
            target_values, return_counts=True)
        for cluster in clusters:
            cluster_values = np.array([
                frame for idx, frame in enumerate(values)
                if target_values[idx] == cluster
            ])
            avg_cluster = np.mean(cluster_values, axis=0)
            norm_cluster = normalize(avg_cluster)
            res.append(norm_cluster)
        res = np.matrix(res)
        return np.amax(res, axis=0).T

    if not max_scale:
        avg_relevance = np.mean(relevance, axis=0)
        avg_sensitivity = np.mean(sensitivity, axis=0)
    else:
        avg_relevance = max_scaled_average(relevance, target_values)
        avg_sensitivity = max_scaled_average(sensitivity, target_values)
    if plot:
        plt.hist(avg_sensitivity, 50, label="Sensitivity", alpha=1)
        plt.hist(avg_relevance, 50, label="Relevance propagation", alpha=0.5)
        plt.legend()
        plt.show()
    return avg_relevance, avg_sensitivity


def analyze_relevance_per_cluster(relevance, sensitivity, target_values):
    """
    Find the most relevant atoms, per cluster
    TODO maybe we don't need this method. It should be refactored
    """
    relevance_per_cluster, sensitivity_per_cluster = {}, {}
    (clusters, cluster_count) = np.unique(target_values, return_counts=True)
    for cluster in clusters:
        cluster_rel = np.array([
            frame for idx, frame in enumerate(relevance)
            if target_values[idx] == cluster
        ])
        cluster_sens = np.array([
            frame for idx, frame in enumerate(sensitivity)
            if target_values[idx] == cluster
        ])
        cluster_target = np.array(
            [t for idx, t in enumerate(target_values) if t == cluster])
        avg_rel, avg_sens = analyze_relevance(
            cluster_rel, cluster_sens, cluster_target, plot=False)

        relevance_per_cluster[cluster] = normalize(avg_rel)
        sensitivity_per_cluster[cluster] = normalize(avg_sens)
    return relevance_per_cluster, sensitivity_per_cluster

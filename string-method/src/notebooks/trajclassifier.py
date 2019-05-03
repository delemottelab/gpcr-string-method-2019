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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append('heatmapping/')
from colvars import *
from notebooks.MD_common import MD_fun

logger = logging.getLogger("trajClass")


def _inverse(frame_distances):
    """In MDFun.getAllCalphaDistances we use the inverse distance to get rid of residues. This takes the inverse again..."""
    for i in range(0, len(frame_distances)):
        for j in range(i + 1, len(frame_distances)):
            res = 1 / frame_distances[::, i, j]
            frame_distances[::, i, j] = res
            frame_distances[::, j, i] = res


def number_unique_elements(rowcount):
    return int(rowcount * (rowcount - 1) / 2)


def vectorize_matrices(frame_distances):
    # TODO look for faster numpy implementation
    # only bother with upper diagonal elements
    if len(frame_distances.shape) == 2:
        logger.debug("Already a vector %s", frame_distances.shape)
        return frame_distances.copy()
    rowcount = frame_distances.shape[1]
    number_elements = number_unique_elements(rowcount)
    vector_to_matrix_mapping = [
        to_matrix_indices(i, rowcount) for i in range(0, number_elements)
    ]
    res = np.empty((len(frame_distances),
                    number_unique_elements(len(frame_distances[0]))))
    for i in range(0, len(frame_distances)):
        if i % 500 == 0:
            logger.debug("Vectorizing frame %s/%s", i, len(frame_distances))
        vec = np.empty((number_elements,))
        res[i] = fill_vector(vec, frame_distances[i], vector_to_matrix_mapping)
    return res


def to_matrix_indices(vector_idx, rowcount):
    """
    We assume that we read the elements of the matrix in sequential order one row at the time only using off diagonal elements in the upper half
    TODO faster O(1) implementation possible if we use some summations etc.
    """
    lastcount = 0
    for i in range(0, rowcount):
        items_for_row = rowcount - i - 1
        newcount = lastcount + items_for_row
        if newcount > vector_idx:
            clmn = i + 1 + (vector_idx - lastcount)
            return i, clmn
        lastcount = newcount


#     counter = 0
#     for i in range(0, rowcount):
#         for j in range(i+1, rowcount):
#             if counter == vector_idx:
#                 return i,j
#             counter +=1


def fill_vector(vec, distance_matrix, vector_to_matrix_mapping):
    for i in range(0, len(vec)):
        vec[i] = distance_matrix[vector_to_matrix_mapping[i]]
    return vec


def transform_to_matrix(target_values):
    """
    Transform a vector of cluster indices to matrix format where a 1 on the ij element means that the ith frame was in cluster state j+1
    """
    number_colors = len(set([t for t in target_values]))
    T = np.zeros((len(target_values), number_colors), dtype=int)
    for i in range(0, len(target_values)):
        T[i, target_values[i] - 1] = 1
    return T


def transform_to_vector(target_values):
    """
    Transform a matrix format where a 1 on the ij element means that the ith frame was in cluster state j+1 to a vector with the cluster indices
    """
    number_colors = target_values.shape[1]
    T = np.empty((len(target_values),), dtype=int)
    for i in range(0, len(target_values)):
        T[i] = target_values[i].argmax() + 1
    return T


def transform_and_scale(frame_distances):
    """
    Vectorizes and scales the input
    """
    training_samples = vectorize_matrices(frame_distances)
    # Very important to scale the samples. Standardscaler seems to work fine
    scaler = StandardScaler()
    scaler.fit(training_samples)
    training_samples = scaler.transform(training_samples)
    return training_samples, scaler


def transform_and_train(frame_distances,
                        cluster_simu,
                        trainingstep=1,
                        randomize=False):
    """
    Vectorizes, scales and trains a classifier
    If randomize is set to False the classifier will not use a constant random seed.
    Therefore you can expect slightly different between instance calls if randomize=True
    """
    training_samples, scaler = transform_and_scale(frame_distances)
    # Note that we can train an MLPClassifier with an array instead of a vector, but that leads to problems with relevance propagation for binary classification
    # See https://stackoverflow.com/questions/46525574/dimension-of-weights-and-bias-for-binary-classification-in-scikit-learn
    target_values = transform_to_matrix(cluster_simu.cluster_indices)
    logger.debug("Starting training with %s samples and %s values",
                 len(training_samples), len(target_values))
    classifier = MLPClassifier(
        solver='lbfgs',
        random_state=(None if randomize else 89274),
        activation='relu',
        max_iter=10000)
    # solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    classifier.fit(training_samples[::trainingstep],
                   target_values[::trainingstep])
    return training_samples, target_values, scaler, classifier


def check_predictions(predictions, training_samples, target_values):
    """Checks that the predicition of the training values are the same as the target values"""
    predictions = transform_to_vector(predictions)
    target_values = transform_to_vector(target_values)
    plt.figure(figsize=(16, 3))
    plt.plot(target_values, label="Targets")
    plt.plot(predictions, '--', label="Predictions")
    plt.ylabel(r'Clustered state')
    plt.yticks(range(0, max(target_values)))
    plt.xlabel("Frame#")
    plt.legend()
    plt.show()
    errorcount = 0.0
    for i, target in enumerate(target_values):
        prediction = predictions[i]
        if prediction != target:
            errorcount += 1

    #             logger.debug("Wrong prediction %s for trained target %s at index %s",prediction, target, i)
    logger.info("Prediction error rate=%s percent",
                100 * (errorcount / len(predictions)))


"""CVs and help classes"""


def create_classifier_cvs(clustering_id, training_samples, target_values, scaler, classifier, trainingstep,
                          query="name CA", cvs=None):
    preprocessor = CvsVectorizer(cvs) if query is None else CAContactsVectorizer(query=query)
    discrete_classifier_cv = DiscreteClassifierCv(clustering_id, preprocessor,
                                                  scaler,
                                                  classifier)
    probaility_classifier_cvs = []
    for i in range(target_values.shape[1]):
        cv = ProbabilityClassifierCv(clustering_id + str(i), preprocessor,
                                     scaler,
                                     classifier, i)
        probaility_classifier_cvs.append(cv)
    return discrete_classifier_cv, probaility_classifier_cvs


class CvsVectorizer(object):

    def __init__(self, cvs):
        self.cvs = cvs

    def process(self, traj):
        dists = eval_cvs(traj, self.cvs)
        return vectorize_matrices(dists)


class CAContactsVectorizer(object):

    def __init__(self, query="name CA"):
        self.query = query

    def process(self, traj):
        dists = MD_fun.MD_functions().getAllCalphaDistances(traj, query=self.query if hasattr(self, "query") else "name CA")
        return vectorize_matrices(dists)


class ClassifierCv(CV):
    def __init__(self, id, preprocessor, scaler, classifier):
        CV.__init__(self, id, lambda traj: self.classify(traj))
        self.preprocessor = preprocessor
        self.scaler = scaler
        self.classifier = classifier

    def classify(self, traj):
        raise NotImplementedError

    def __str__(self):
        return "ClassifierCv(id=%s)" % (self.id)


class DiscreteClassifierCv(ClassifierCv):
    def __init__(self, id, preprocessor, scaler, classifier):
        ClassifierCv.__init__(self, id, preprocessor, scaler, classifier)

    def classify(self, traj):
        samples = self.preprocessor.process(traj)
        samples = self.scaler.transform(samples)
        predictions = self.classifier.predict(samples)
        # convert to one value per frame
        res = np.empty((predictions.shape[0],))
        for i, p in enumerate(predictions):
            res[i] = p.argmax()
        return res


class DependentDiscreteClassifierCv(DependentCV):

    def __init__(self, id, simulation_cvs, scaler, classifier):
        DependentCV.__init__(self,
                             id,
                             lambda cv_evals: self.classify(cv_evals),
                             simulation_cvs)
        self.classifier = classifier
        self.scaler = scaler

    def classify(self, cv_evals):
        samples = self.scaler.transform(cv_evals)
        predictions = self.classifier.predict(samples)
        res = np.empty((predictions.shape[0],))
        for i, p in enumerate(predictions):
            res[i] = p.argmax()
        return res


class ProbabilityClassifierCv(ClassifierCv):
    def __init__(self, id, preprocessor, scaler, classifier, output_index):
        ClassifierCv.__init__(self, id, preprocessor, scaler, classifier)
        self.output_index = output_index

    def classify(self, traj):
        samples = self.preprocessor.process(traj)
        samples = self.scaler.transform(samples)
        prob_predictions = self.classifier.predict_proba(samples)[:, self.output_index]
        return prob_predictions

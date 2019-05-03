from __future__ import absolute_import, division, print_function

import json

from colvars.generic_cvs import *

logger = logging.getLogger(__name__)


def load_cvs_definition(filepath):
    """
    :param filepath:
    :return json objects
    """
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def load_cvs(filepath):
    """
    Loads a file and returns CV objects
    :param filepath:
    :return:
    """
    defs = load_cvs_definition(filepath)
    return create_cvs(defs)


def create_cvs(cvs_definition):
    """
    Create cv objects from definitions of standard types such as float, strings and ints defined as json objects
    :param cvs_definition: JSONs as loaded by function load_cvs_definition
    :return: an numpy array of cvs
    """
    cvs = []
    for i, cv_def in enumerate(cvs_definition["cvs"]):
        clazz = cv_def["@class"]
        if clazz == CADistanceCv.__name__:
            cv = _parse_CADistanceCv(cv_def)
        else:
            logger.warn("Class %s cannot be parsed right now (index %s, json %s). Please implement.", clazz, i, cv_def)
            continue
        cvs.append(cv)

    return np.array(cvs)


def _parse_CADistanceCv(cv_def):
    cv = CADistanceCv(cv_def["id"], cv_def["res1"], cv_def["res2"], periodic=cv_def["periodic"])
    cv.normalize(scale=cv_def.get("scale", 1.), offset=cv_def.get("offset", 0))
    return cv


if __name__ == "__main__":
    logger.info("Started")
    filepath = "..."
    cvs_definition = load_cvs_definition(filepath)
    cvs = create_cvs(cvs_definition)
    logger.debug("Done. Created CVs %s", cvs)

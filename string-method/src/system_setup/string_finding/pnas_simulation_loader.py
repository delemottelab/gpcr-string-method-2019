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
from utils.helpfunc import *

logger = logging.getLogger("pnas_loader")

# plt.style.use('ggplot')
refstruct_dir = "/home/oliverfl/git/string-method/gpcr/reference_structures/"
active_traj = md.load(refstruct_dir + "active_CA.pdb")
logger.info("loaded active structure, %s", active_traj)
# protein database file, inactive structure with g-protein
inactive_traj = md.load(refstruct_dir + "inactive_CA.pdb")
logger.info("loaded inactive active structure, %s", inactive_traj)


def load_simulations(simulation_conditions, stride=1, timestep_size=timestep_size, simulation_dir="simulations"):
    """
    -Load all simulations
    """
    result = []
    for condition, number in simulation_conditions:
        simulation = Simulation({
            "condition": condition,
            "number": number,
            "name": "all",
            "timestep": timestep_size * stride
        })
        traj = md.load(
            simulation_dir + simulation.path + simulation.name + ".dcd",
            top=simulation_dir + simulation.path + simulation.name + ".pdb",
            stride=stride)
        #         traj = traj[::timestep]
        simulation.traj = traj
        result.append(simulation)
    return result


def load_freemd(directory,
                traj_filename,
                top_filename,
                timestep_size=1,
                stride=1):
    """
    -Load FREE MD simulations
    """
    simulation = Simulation({
        "condition": "FREE",
        "number": "MD",
        "name": "",
        "timestep": timestep_size * stride
    })
    traj = md.load(
        directory + traj_filename,
        top=directory + top_filename,
        stride=stride)
    simulation.traj = traj
    return [simulation]

def evaluate_simulations(simulations, cvs):
    """
    -evaluate CVs of the trajectories
    -combine results with simulations into tuples
    """
    result = []
    for simulation in simulations:
        cv_values = np.empty((len(cvs), len(simulation.traj)))
        for i, cv in enumerate(cvs):
            val = cv.eval(simulation.traj)
            if len(val.shape) > 1:
                val = val[:,0]
            cv_values[i, :] = val
        result.append((simulation, cv_values))
    return result

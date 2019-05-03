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
import argparse
import matplotlib as mpl

mpl.use('Agg')  # TO DISABLE GUI. USEFUL WHEN RUNNING ON CLUSTER WITHOUT X SERVER
from iterationrunner import *

logger = logging.getLogger("StringSimuServer")


def create_argparser():
    parser = argparse.ArgumentParser(
        epilog='String with swarms of tra1jectories code. Intended to dispatch bash jobs and analyze the resulting trajectories.\nBy Oliver Fleetwood 2017-2018.')
    """IMPORTANT CONFIG FOR METHOD"""
    parser.add_argument('-wd', '--working_dir', type=str, help='working directory', default="../gpcr/")
    parser.add_argument('-i', '--iteration', type=int, help='String Iteration', required=True)
    parser.add_argument('-mi', '--max_iteration', type=int, help='Maximum iteration number or the job will finish',
                        required=False,
                        default=15)  # Fairly low so that you make sure to check that everything is working as expected
    parser.add_argument('--min_swarm_batches', type=int,
                        help='Min. number of swarm batches per string point',
                        default=20)
    parser.add_argument('--max_swarm_batches', type=int,
                        help='Max. number of swarm batches per string point.',
                        default=20)
    parser.add_argument('--swarm_batch_size', type=int,
                        help='Number of swarms per submitted batch job.',
                        default=1)
    parser.add_argument('--swarm_convergence_fraction', type=float,
                        help='Maximum fraction of the distance between previous and new average drift vector from the swarms relative to the new drift vector.',
                        default=0.2)
    parser.add_argument("--fixed_endpoints", help="Hold string endpoints fixed",
                        type=lambda x: (str(x).lower() == 'true'),
                        default=True)
    parser.add_argument("-npf", "--new_point_frequency", type=int,
                        help="Frequency at which to add a new point to the string",
                        default=sys.maxint)
    parser.add_argument("--max_number_points", type=int,
                        help="Max number of points on the string",
                        default=60)
    parser.add_argument("--swarm_drift_scale", help="Float to multiply the swarms total drift by",
                        type=float,
                        default=1.)
    parser.add_argument("--equidistant_points", help="Keep points equidistant along string (or dynamic distances will be computed from the swarms)",
                        type=lambda x: (str(x).lower() == 'true'),
                        default=False)
    """ENVIRONMENT CONFIG"""
    parser.add_argument("-env", "--environment", type=str, help="Environment type (local/cluster...)", required=True)
    parser.add_argument("-sm", "--start_mode", help="Start mode (server etc.)",
                        choices=['server', 'setup', 'postprocess', 'analysis', 'append'],
                        required=True)
    parser.add_argument("--simulator", help="Simulation package used",
                        type=str,
                        default="plumed")
    parser.add_argument("--command_gmx", help="Command to run gromacs",
                        # nargs='+',
                        type=str,
                        default="gmx")
    parser.add_argument("--command_submit", help="Command to submit jobs",
                        type=str,
                        default="bash")
    parser.add_argument("--version", help="Code version. ",
                        type=float,
                        default=2.2)
    """DIRECTORIES AND FILES"""
    parser.add_argument('-sp', '--string_filepath', type=str,
                        help='string textfile path with which should be formatted with the current iteration number. Relative to cvs_dir if not absolute.',
                        default="string-paths/string%s.txt")
    parser.add_argument('-cd', '--cvs_dir', type=str, help='CV directory. Relative to working_dir if not absolute.',
                        default="cvs/cvs-len5_good/")
    parser.add_argument('--cvs_filetype', type=str, help='CVs filetype (json or pkl).',
                        default=None)
    parser.add_argument("-sdir", "--structure_dir", type=str,
                        help="Top directory for structures (topology, equilibrated structured and index) for simu system. Relative to working_dir if not absolute.",
                        required=True)
    parser.add_argument('-sd', '--simulation_dir', type=str,
                        help='Simulation directory. Relative to working_dir if not absolute.',
                        default=".string_simu/")
    parser.add_argument('--template_dir', type=str,
                        help='Path to mdp templates etc. Relative to working_dir if not absolute.',
                        default="simulation_config/")
    # parser.add_argument("-top", "--topology" type=str, help="Topology file for simu system", required=True)# default="confout.gro")
    parser.add_argument('-ro', '--restraints_out_file', type=str,
                        help='Restrains output file (relative to CV directory). Restraints will be appended to this file',
                        default="restraints.dat")  # default="topology/restraints/PROA_rest.itp")
    """OTHER"""
    parser.add_argument("-restol", "--restraint_tolerance", type=float, help="Tolerance for restraint force",
                        default=1.2e-3)
    parser.add_argument("--convergence_limit", type=float,
                        help="Limit for when the string has converged between two iterations (we recommend you to confirm convergence in other ways)",
                        default=1e-5)
    parser.add_argument("--simu_id", help="Id to distinguish string simulations from eachother",
                        type=str,
                        default="")
    parser.add_argument("--job_pool_size", help="Maximum number of simultaneous jobs",
                        type=int,
                        default=200)

    return parser


def run_server(runner):
    logger.info("Running Server")
    while runner.iteration <= args.max_iteration:
        runner.run()
        processor = postprocess(runner)
        runner.after_iteration()
        if processor.convergence < args.convergence_limit:  # Will in practice never happen
            logger.info("String converged with %s after iteration %s. Stopping", processor.convergence,
                        runner.iteration)
            break
        else:
            logger.debug("Convergence after iteration %s:%s", runner.iteration, processor.convergence)
        runner.init_iteration(runner.iteration + 1)


def postprocess(runner):
    logger.info("Postprocessing")
    processor = SingleIterationPostProcessor(runner, save=True, plot=False)
    processor.run()
    logger.info("String converge after iteration %s: %s", runner.iteration, processor.convergence)
    return processor


def setup(runner):
    logger.info("Setup")
    runner.create_files()
    return runner


if __name__ == "__main__":
    logger.info("----------------Starting string of swarms simulator by Oliver Fleetwood 2017-2018------------")
    parser = create_argparser()
    args = parser.parse_args()
    # args.command_gmx = '"' + ' '.join(args.command_gmx).replace('"', '') + '"'
    logger.info("Starting simulation with arguments: %s", args)
    runner = StringIterationRunner(args)
    if args.start_mode == 'server' or args.start_mode == 'append':
        run_server(runner)
    elif args.start_mode == 'postprocess':
        postprocess(runner)
    elif args.start_mode == 'setup':
        setup(runner)
    else:
        logger.error("Startmode %s not supported", args.start_mode)
    logger.info("Finished.")

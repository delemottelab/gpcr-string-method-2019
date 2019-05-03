from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import run_simulation as rs


def simuid_to_iterations(simu_id):
    return {
        "holo-optimized": (140, 248),
        "apo-optimized": (200, 282),
        "apo-curved": (50, 150),
        "apo-optimized_part1": (90, 122),
        "apo-optimized_part2": (120, 140),
        "endpoints-holo": (1, 10),
        "endpoints-apo": (1, 10),
        "holo-curved": (2, 99),  # (66,70,99)
        "straight-holo-optimized": (350, 400),
        "holo-straight": (100, 155),
        "holo-optimized_part1": (60, 130),
        "holo-optimized_part2": (131, 200),
        "to3sn6-holo": (150, 200),
        "to3sn6-apo": (150, 250),
        "beta1-apo": (130, 189),
        "pierre-ash79": (160, 200),
        "pierre-asp79_Na": (160, 200)
    }.get(simu_id)


def simuid_to_transitions(simu_id, cvtype):
    from analysis.FE_analysis.frameloading import StringSimulationFrameLoader, TransitionFrameLoader
    holo_3p0g_3sn6 = StringSimulationFrameLoader(
        "to3sn6-holo",
        # "endpoints-holo",
        cvtype
    )
    apo_3p0g_3sn6 = StringSimulationFrameLoader(
        "to3sn6-apo",
        cvtype
    )
    holo_freemd_transitions = TransitionFrameLoader(
        # "/data/oliver/pnas2011b-Dror-gpcr/freemd/freemd-dec2017/",
        "/data/oliver/pnas2011b-Dror-gpcr/freemd/3p0g-inactive-endpoint/",
        # "3p0g_prod4.part0003.trr",
        # "all.xtc",
        "freemd.xtc",
        "freemd.gro",
        # "confout.gro",
        dt=500,
        lagtime=5000,
        traj_query="protein"
        # "/data/oliver/pnas2011b-Dror-gpcr/hsampling/easter/hsampling/.simu/151/in/",
        # "swarm.xtc", "swarm.gro", 500, 5000
    )
    return {
        "apo-curved": None,
        "apo-optimized": apo_3p0g_3sn6,
        "holo-curved": None,  # holo_freemd_transitions,  # None,
        "straight-holo-optimized": None,  # holo_3p0g_3sn6,
        "holo-straight": None,
        "holo-optimized": holo_3p0g_3sn6,  # holo_freemd_transitions
        "to3sn6-holo": None,  # StringSimulationFrameLoader("endpoints-holo", cvtype),
        "to3sn6-apo": None,  # StringSimulationFrameLoader("endpoints-apo", cvtype)
    }.get(simu_id, None)


def get_args_for_simulation(simu_id, simulation_directory="/home/oliverfl/projects/gpcr/simulations/strings/"):
    # TODO load from gitignored file
    if simu_id.startswith("apo-optimized"):
        args = "-i \
                34\
                -env\
                local\
                --simu_id=" + simu_id + " " + \
               "--start_mode\
               analysis\
               --simulator\
               plumed\
               -cd=cvs/cvs-len5_good/\
               -wd={}april/apo5-drorpath/gpcr/\
               --fixed_endpoints=True\
               --structure_dir=reference_structures/3p0g-noligand/\
               --swarm_batch_size=4\
               --min_swarm_batches=4\
               --max_swarm_batches=8\
                --cvs_filetype=pkl\
               --version=2"
    elif simu_id.startswith("holo-optimized"):
        args = "-i\
                141 \
                -env\
                local\
                --simu_id=" + simu_id + " " + \
               "--start_mode\
               analysis\
               --simulator\
               plumed\
               -cd=cvs/cvs-len5_good/\
               -wd={}april/holo5-drorpath/gpcr/\
               --fixed_endpoints=True\
               --structure_dir=reference_structures/3p0g-ligand/\
               --swarm_batch_size=4\
               --min_swarm_batches=4\
               --max_swarm_batches=8\
                --cvs_filetype=pkl\
                --version=2"
    elif simu_id.startswith("holo-curved"):
        args = "-i \
                99\
                -env\
                local\
                --simu_id=holo-curved\
                --start_mode\
                analysis\
                --simulator\
                plumed\
                -cd=cvs/cvs-len5_good/\
                -wd={}jan/5cvs-drorpath/gpcr/\
                --fixed_endpoints=False\
                --structure_dir=reference_structures/3p0g-ligand/\
                --swarm_batch_size=1\
                --min_swarm_batches=20\
                --max_swarm_batches=30\
                --cvs_filetype=pkl\
                 --version=1"
    elif simu_id.startswith("apo-curved"):
        args = "-i=151 \
            -env=local\
            --simu_id=apo-curved\
            --start_mode=analysis\
            --simulator=plumed\
            -cd=cvs/cvs-len5_good/\
            -wd={}jan/apo5-drorpath/gpcr/\
            --fixed_endpoints=False\
            --structure_dir=reference_structures/3p0g-noligand/\
            --swarm_batch_size=1\
            --min_swarm_batches=20\
            --max_swarm_batches=30\
            --cvs_filetype=pkl\
            --version=1"
    elif simu_id.startswith("holo-straight"):
        args = "-i=155\
                -env=\
                --simu_id=holo-straight\
                --start_mode=analysis\
                --simulator=plumed\
                -cd=cvs/cvs-len5_good/\
                -wd={}jan/5cvs-straight/gpcr/\
                --fixed_endpoints=True\
                --structure_dir=reference_structures/3p0g-ligand/\
                --swarm_batch_size=1\
                --min_swarm_batches=20\
                --max_swarm_batches=30\
                --cvs_filetype=pkl\
                --version=1"
    elif simu_id.startswith("endpoints-holo"):
        args = "-i\
                11 \
                -env\
                local\
                --simu_id=" + simu_id + " " + \
               "--start_mode\
               analysis\
               --simulator\
               plumed\
               -cd=cvs/cvs-len5_good/\
               -wd={}april/endpoints-holo5/gpcr/\
               --fixed_endpoints=False\
               --structure_dir=reference_structures/3p0g-ligand/\
               --swarm_batch_size=4\
               --min_swarm_batches=1\
               --max_swarm_batches=200\
               --cvs_filetype=pkl\
                --version=2"
    elif simu_id.startswith("endpoints-apo"):
        args = "-i\
                11 \
                -env\
                local\
                --simu_id=" + simu_id + " " + \
               "--start_mode\
               analysis\
               --simulator\
               plumed\
               -cd=cvs/cvs-len5_good/\
               -wd={}april/endpoints-apo5/gpcr/\
               --fixed_endpoints=False\
               --structure_dir=reference_structures/3p0g-noligand/\
               --swarm_batch_size=4\
               --min_swarm_batches=1\
               --max_swarm_batches=200\
                --cvs_filetype=pkl\
                --version=2"
    elif simu_id.startswith("straight-holo-optimized"):
        args = "-i\
                18 \
                -env\
                local\
                --simu_id=" + simu_id + " " + \
               "--start_mode\
               analysis\
               --simulator\
               plumed\
               -cd=cvs/cvs-len5_good/\
               -wd={}jun/holo5-optimized-straight/gpcr/\
               --fixed_endpoints=True\
               --structure_dir=reference_structures/3p0g-ligand/\
               --swarm_batch_size=4\
               --min_swarm_batches=1\
               --max_swarm_batches=200\
               --cvs_filetype=pkl\
                --version=2.1"
    elif simu_id.startswith("to3sn6-holo"):
        args = "-i\
                28 \
                -env\
                local\
                --simu_id=" + simu_id + " " + \
               "--start_mode\
               analysis\
               --simulator\
               plumed\
               -cd=cvs/cvs-len5_good/\
               -wd={}jun/to3sn6-holo5/gpcr/\
               --fixed_endpoints=True\
               --structure_dir=reference_structures/3p0g-ligand/\
               --swarm_batch_size=4\
               --min_swarm_batches=1\
               --max_swarm_batches=200\
                --cvs_filetype=pkl\
                --version=2.1"
    elif simu_id.startswith("to3sn6-apo"):
        args = "-i\
                7 \
                -env\
                local\
                --simu_id=" + simu_id + " " + \
               "--start_mode\
               analysis\
               --simulator\
               plumed\
               -cd=cvs/cvs-len5_good/\
               -wd={}jun/to3sn6-apo5/gpcr/\
               --fixed_endpoints=True\
               --structure_dir=reference_structures/3p0g-noligand/\
               --swarm_batch_size=4\
               --min_swarm_batches=1\
               --max_swarm_batches=200\
               --cvs_filetype=pkl\
               --version=2.1"

    elif simu_id == "beta1-apo":
        args = "-i \
                1\
                -env\
                local\
                -wd={}july/beta1-apo/gpcr/\
                --structure_dir\
                reference_structures/beta1-apo/\
                --start_mode=analysis\
                --simulator\
                plumed\
                -cd\
                cvs/beta1-cvs/\
                --swarm_batch_size=4\
                --min_swarm_batches=20\
                --max_swarm_batches=30\
               --cvs_filetype=pkl\
                --version=2.1\
                --simu_id=" + simu_id
    elif simu_id == "pierre-ash79":
        args = "-i \
                200\
                -env\
                local\
                -wd={}pierre/ash79/gpcr/\
                --structure_dir\
                reference_structures/3p0g-ash79/\
                --start_mode=analysis\
                -cd=cvs/ash79-cvs/\
                --simulator\
                plumed\
                --swarm_batch_size=4\
                --min_swarm_batches=20\
                --max_swarm_batches=30\
                --cvs_filetype=pkl\
                --version=2.1\
                --simu_id=" + simu_id
    elif simu_id == "pierre-asp79_Na":
        args = "-i \
                200\
                -env\
                local\
                -wd={}pierre/asp79_Na/gpcr/\
                --structure_dir\
                reference_structures/3p0g-asp79_Na/\
                --start_mode=analysis\
                -cd=cvs/asp79_Na-cvs/\
                --simulator=plumed\
                --swarm_batch_size=4\
                --min_swarm_batches=20\
                --max_swarm_batches=30\
                --cvs_filetype=pkl\
                --version=2.1\
                --simu_id=" + simu_id
    else:
        raise Exception("Unknown simu_id{}".format(simu_id))
    args = args.format(simulation_directory)
    parser = rs.create_argparser()
    return parser.parse_args(args.split())

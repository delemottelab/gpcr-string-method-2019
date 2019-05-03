#!/bin/bash
echo "This script is an example that helps you run the string method for an apo state beta2 receptor."
echo "Optional parameters in order: iteration startmode environment simulator (see run_simulation.py for details)"
echo "Small modifications this script and the files in the bash/ directory are required to get it up and running."
echo "A working python environment is required."
echo ""
echo "For questions contact oliver.fleetwood@gmail.com"

if [ $# -lt 1 ]
  then
    echo "No iteration number or start mode provided. At least iteration number is required as command line arguments. Try again!"
    exit 1
fi
iteration=$1
if [ $# -lt 2 ]
  then
    echo "No startmode provided. Using default"
    startmode="server"
else
    startmode=$2
fi
if [ $# -lt 3 ]
  then
    echo "No environment provided. Using default"
    #Change to 'local' to run in a local environment without a submission queue.
    env="slurm"
else
    env=$3
fi
if [ $# -lt 4 ]
  then
    echo "No simulator provided. Using default"
    simulator="plumed"
else
    simulator=$4
fi
##Change anacodna below
export LOCAL_ANACONDA=/cfs/klemming/nobackup/o/oliverfl/py2
source $LOCAL_ANACONDA/bin/activate $LOCAL_ANACONDA
cmd="python run_simulation.py \
--iteration=$iteration \
-env=$env \
--start_mode=$startmode \
--simulator=$simulator \
--simu_id=apoAsp79 \
--structure_dir=reference_structures/3p0g/asp79-apo/ \
--cvs_dir=cvs/len5/asp79/ \
--swarm_batch_size=4 \
--min_swarm_batches=4 \
--max_swarm_batches=8 \
--fixed_endpoints=True \
--command_gmx=gmx_mpi \
--command_submit=sbatch \
--job_pool_size=200 \
--max_number_points=60 \
--new_point_frequency=8 \
--max_iteration=200"
echo $cmd
$cmd  >> server.log 2>&1 &
echo "Appending to server.log"
source $LOCAL_ANACONDA/bin/deactivate
#Here I have activated and deactivated anaconda here since we might get weird errors for a batch job once in a while
#if the batch job uses our virtual python environment to load some stuff (the virtual env might not be available)
#I am not completely sure that is the reason that jobs sometimes fails, but it won't hurt to deactive the conda environment afterwards

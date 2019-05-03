#!/bin/bash -l
# see https://www.pdc.kth.se/software/software/GROMACS/beskow/5.0.6/index.html 

# Include your allocation number
##SBATCH -A  YOUR_ALLOCATION

# Name your job
#SBATCH -J $1

# Total number of MPI tasks #number of nodes times 32 (?)
#SBATCH --ntasks 32

# Total number of nodes
#SBATCH --nodes 1

# length in hours
#SBATCH -t 00:05:00

# Receive e-mails when your job starts and ends
#SBATCH --mail-user=YOUR@EMAIL.COM --mail-type=END

# Output file names for stdout and stderr
#SBATCH -e slurm-%j.err -o slurm-%j.out

export OMP_NUM_THREADS=1
export GMX_MAXBACKUP=-1
#APRUN_OPTIONS="-n 8 -d 1 -cc none"
export RUN_OPTIONS="aprun -n 32 -d 1 -cc none " #n should be same as number of MPI above

module swap PrgEnv-cray PrgEnv-gnu
module add gromacs/2016.5-plumed_2.3.5

bash $2


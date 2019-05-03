#!/bin/bash
##Replace with your own plumed kernel below
source /usr/local/gromacs/bin/GMXRC
export PLUMED_KERNEL=/usr/local/lib/libplumedKernel.so
echo "Submitting $1"
export RUN_OPTIONS=""
bash $2


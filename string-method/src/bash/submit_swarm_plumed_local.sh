#!/bin/bash
##Replace with your own kernel below
source /usr/local/gromacs/bin/GMXRC
echo "Submitting $1"
export RUN_OPTIONS=""
bash $2


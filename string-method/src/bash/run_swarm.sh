#!/bin/bash
pointname=$1
simudir=$2
struct_dir=$3
template_dir=$4
swarmname=$5
batchsize=$6
gmx=${RUN_OPTIONS}$7
#cpt=$8
topology="$struct_dir/topology/topol.top"
if [ $batchsize -gt 1 ]; then
    multioption="-multi $batchsize"
    deffname=$swarmname
else
    multioption=""
    deffname=${swarmname}0
fi
echo "Starting swarm simulation $swarmname in directory $simudir."
cd $simudir
if [ ! -f $pointname.done ]; then
    echo "File $pointname.done not found. exiting" >&2
    exit 1
fi
date >> $swarmname.progress
touch ${swarmname}.resubmitted
trap oninterrupt 1 9 15
#--------------------Create done file and see if the other simus are done
function resubmit {
    if [ -f $swarmname.xtc ] && [ -f $swarmname.cpt ] && [[ $(wc -l < ${swarmname}.xtc) -ge 0 ]] && [[ $(wc -l < ${swarmname}.resubmitted) -le 3 ]]; then
            date >> swarmname.resubmitted
            sbatch submit_$swarmname.sh
            exit 89
    fi
    exit 79
}
function oninterrupt {
    echo "Interrupted script $0"
    if [ ! -f $swarmname.done ]; then
        echo "No done file found."
        #Resubmit if resubmission count is not too high
        resubmit
    fi
    exit 99
}

if [ ! -f ${swarmname}*.cpt ]; then
    #---------------------------
    #echo "Run from x:th output of previous simulation"
    #cmd="$gmx convert-tpr -s $pointname-restrained.tpr -f $pointname-restrained.trr -e $pointname-restrained.edr -n $struct_dir/index.ndx -o $swarmname-in.tpr" # -time $offsettime"
    #echo $cmd
    #$cmd
    #cmd="$gmx grompp -f $template_dir/template-swarm.mdp -c $swarmname-in.tpr -o $swarmname.tpr -p $topology -po $swarmname.mdp -n $struct_dir/index.ndx"
    ######################################################3
    #----START FROM OUTPUT COORDINATES OF RESTRAINED SIMU-----------
    #cmd="$gmx grompp -f $template_dir/template-swarm.mdp -c $pointname-restrained.gro -o $swarmname.tpr -p $topology -po $swarmname.mdp -n $struct_dir/index.ndx"
    ########################################
    #-----------------START FROM RESTRAINED CHECKPOINT--------------------------
#    cmd="$gmx grompp -f $template_dir/template-swarm.mdp -c $pointname-restrained.tpr -o $swarmname.tpr -p $topology -po $swarmname.mdp -n $struct_dir/index.ndx -t $pointname-restrained.cpt"
    ##############RUN##############################
#    echo "$cmd"
#    $cmd
    for i in $(eval echo {0..$batchsize}); do cp ${pointname}s.tpr $deffname$i.tpr; done;
    cmd="$gmx mdrun $multioption -s $deffname.tpr -deffnm $deffname -x  -cpt 2 -c"
    echo "$cmd"
    $cmd
else
    cmd="$gmx mdrun $multioption -s $deffname.tpr -cpi $deffname.cpt -deffnm $deffname -c -cpt 3"
	echo $cmd
	$cmd
fi
#--------------------Create done file and see if the other simus are done
if [ ! -f ${swarmname}*.gro ]; then
    echo "No output coordinates found"
    resubmit
fi
echo "Finished simu $swarmname"
date >> $swarmname.done
echo "Removing files.."
rm ${deffname}*.cpt
#rm ${deffname}*.gro
rm ${deffname}*_prev*.cpt
rm ${deffname}*.tpr
rm -f $swarmname.progress


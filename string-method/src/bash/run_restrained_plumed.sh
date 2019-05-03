#!/bin/bash
#iteration="iter$1"
#point="point$2"
name=$1
simudir=$2
struct_dir=$3
template_dir=$4
gmx=${RUN_OPTIONS}$5
topology="$struct_dir/topology/topol.top"
swarmname=${name}s
restrainedname=${name}-restrained
echo "Starting restrained simulation $name in directory $simudir. templates in  $struct_dir"
#---------------------------------------------
cd $simudir
date >> $name.progress
touch ${name}.resubmitted
trap oninterrupt 1 9 15
#--------------------Create done file and see if the other simus are done
function resubmit {
    if [ -f $restrainedname.trr ] && [[ $(wc -l < ${name}-restrained.trr) -ge 0 ]] && [[ $(wc -l < ${name}.resubmitted) -le 3 ]]; then
            date >> $name.resubmitted
            sbatch submit_$name.sh
            exit 89
    fi
    exit 79
}
function oninterrupt {
    echo "Interrupted script $0"
    if [ ! -f $restrainedname.gro ]; then
        echo "No output coordinates found."
        #Resubmit if resubmission count is not too high
        resubmit
    fi
    exit 99
}
#----------------------------------------------
#see http://www.gromacs.org/Documentation/How-tos/Extending_Simulations
if [ ! -f $restrainedname.cpt ]; then
    #echo "------------Minimization------------"
    #cmd="$gmx grompp -f $template_dir/template-minimization_plumed.mdp -c $name-in.gro -o $name-minimization.tpr -p $topology -po $name-minimization.mdp -n $struct_dir/index.ndx -maxwarn -1"
    #echo "$cmd"
    #$cmd
    #cmd="$gmx mdrun -pin on -plumed restraints.dat -s $name-minimization.tpr -deffnm $name-minimization -c"
    #echo "#$cmd"
    #$cmd
    #echo "----------Thermalization----------------"
    #cmd="$gmx grompp -f $template_dir/template-thermalization_plumed.mdp -c $name-minimization.gro -o $name-thermalization.tpr -p $topology -po $name-thermalization.mdp -n $struct_dir/index.ndx -maxwarn -1"
    #echo "$cmd"
    #$cmd
    #cmd="$gmx mdrun -pin on -plumed restraints.dat -s $name-thermalization.tpr -deffnm $name-thermalization -c"
    #echo "$cmd"
    #$cmd
    echo "-------------Equilibration--------------"
    cmd="$gmx grompp -f $template_dir/template-restrained_plumed.mdp -c $name-in.gro -o $restrainedname.tpr -p $topology -po $restrainedname.mdp -n $struct_dir/index.ndx -maxwarn -1"
    echo "$cmd"
    $cmd
    cmd="$gmx mdrun -plumed restraints.dat -s $restrainedname.tpr -deffnm $restrainedname -c -cpt 5"
    echo "$cmd"
    $cmd
else
    cmd="$gmx mdrun -s $restrainedname.tpr -cpi $restrainedname.cpt -deffnm $restrainedname -plumed restraints.dat -c"
	echo $cmd
	$cmd
fi
if [ ! -f $restrainedname.gro ]; then
    echo "No output coordinates found."
    resubmit
else
    echo "Prepare tpr file for swarm files"
    cmd="$gmx grompp -f $template_dir/template-swarm.mdp -c $restrainedname.tpr -o $swarmname.tpr -p $topology -po $swarmname.mdp -n $struct_dir/index.ndx -t $restrainedname.cpt"
    echo "$cmd"
    $cmd
fi
echo "Finished simu $name"
date >> $name.done
echo "Removing files.."
rm $name-*_prev.cpt
rm $name.progress
#rm $name-minimization.tpr
#rm $name-thermalization.tpr

#FOR CHECKPOINTS going from thermalization->restrained:

#gmx grompp -f $template_dir/template-restrained.mdp -c $name-thermalization.tpr -o $restrainedname.tpr -p $topology -po $name-thermalization.mdp -n $struct_dir/index.ndx -t  -cpi $name-thermalization.cpt
#gmx mdrun -plumed restraints.dat -v -s $restrainedname.tpr -deffnm $restrainedname -c -noappend

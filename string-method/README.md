# Welcome to the string-method by Oliver Fleetwood 2017-2019
Code to for running and analyzing the string method with swarms of trajectories

# Setting up a new simulation
## What you need first of all
* A complete simulation system, for example set up with [CHARMM-GUI](http://charmm-gui.org/)
* Some collective variables (CVs). Implement your CVs to extend the *colvars.CV* class, save them to pkl or json format and make sure you can implement them in plumed. There are several examples in the code. Contact us for details. 
* An initial guess of the string between two endpoints in CV space. A straight path is the naive guess.

## Make sure you have all the necessary files in the correct directories 
You can read more on a different page, but in short the following directories are required and should be passed as parameters to the python script __run_simulation.py__:
* parameter _--working_dir_: top directory containing systems specific files
* parameter _--simulation_dir_: The actual coordinates files, gromacs input and output files, submission files and plumed restraints etc. The simulations follow a tree structure of the type {simulation_dir}/{iteration number}/{point index on string}.
* parameter _--cvs_dir_: Top directory to the CVs, containing your file cvs.pkl and the plumed restraints file.
* parameter _--string_filepath_: The path to a numpy txt file with the normalized CV coordinates per point on every row.
* parameter _--structure_dir_: Path to your equilibrated gro file, the index and topology etc. 

In all cases, see the examples in the directory *gpcr* and it should be easier to understand. 

If a path is a relative path, i.e. does not start with a forward slash, '/', then it will be relative to the working directory and not relative to the directory in which you start the simulation (except for string_filepath which is relative to cvs_dir). If any path does start with a slash, it will be an absolute path. 

## Generating input string
* Run targetedmd along your initial string between the endpoints. There is a function with generates this input in plumed for you, see _create_targetmd_input()_ gives some CVs and a string.
* Call _find_initial_points_ to extract frames from the targetedmd trajectory to get good starting points for every point on the string. The output is saved into a directory which can be copied directly to your .string_simu directory

# Running the string simulation
The actual python file for running the simulation is called __run_simulation.py__ but it is often more convenient to create a bash script which loads your python environment, sets the script parameters and redirects the output to a log file. There are examples of such scripts called for example *start_simulation_example.sh*.

The script will start a python process which submits bash jobs and waits for them to finish. Depending on your environment the submitted jobs will look slightly different so you need to make sure these files work for your environment. Look at *src/bash/submit_{restrained/swarm}_{environment}.sh* -> These scripts submit restrained simulations and swarms. There are examples available and you are recommended to start with one of those and make your own changes. For example change the requested allocation time, number of nodes and your allocation ID. 

These submission scripts start a regular bash scripts in which you use gromacs commands such as *mdrun* and *grompp*. These regular bash scripts should be indpendent of environment and will have the necessary simulation parameters injected into them. 

## Equilibrating the string
You probably want to equilibrate your intial frames restrained to the points on the string for some extra time before you start for real. For this to work you need to do two things:
* in *src/bash/submit_restrained_{environment}.sh* change the allocation time and probably the number of nodes
* in the file *template_restrained_plumed.mdp* increase the number of timesteps, *nsteps*.
* Sometimes you also don't want any swarms when you do your equilibration, for example to check that the system looks good before starting your production run. Then you set the parameter *max_swarm_batches* to zero and *max_iteration* to 1. 

## The production run
And finally, to start the production run, you first make sure to reset the script submission time and the number of steps in the restrained simulation, call *bash start_simulation.sh 1*, and sit back and relax.   

To track the progress, you should check the submission queue, tail the server log and look at the created files. 

## Tips:
* If it crashes you can start it with *start_mode=append*. Remember to kill python process when you restart. 
* To test that your submission and gromacs files work, start it with *start_mode=setup* to create a files and then manually submit a job. 


# Analysis
Two types of files are important for analysis:
-the string files called something like *stringX.txt* in the cvs directory.
-npy files which contain the CV values at the start and end of a short trajectory per iteration. These files will be generated by *run_FE_analysis.py* and stored in the directory *swarm_transitions* in your root directory next to *src* and *gpcr*. This script can either run in a cluster or locally on a computer, but it may take a few hours. Visualizations are best done on a desktop. 

First, edit *run_FE_analysis* and select which simulations and CVs you want to compute free energy for. 
Create a file in *src/.local_config/simu_args/{simu_id}* with the parameters for the string simulation. Important is to set the working directory, the CVs and the topology. 
 

To analyze the data, just sync the entire simulation directory from your cluster to a local hard drive. You only need a python environment with the required dependencies installed, you won't need gromacs or plumed for postprocessing. 

In __extra_analysis.py__ there are methods to plot the strings. See how they converge. To check for convergence, compare average strings.

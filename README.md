# gpcr-string-method-2019

This code has been deposited publicly on github. It was originally used to set up and perform string method with swarms of trajectories simulations of the beta2 adrenergic receptor, a G protein coupled receptor (GPCR), as well as to analyze the resulting trajectories. 

In short, the repository be divided into two subtrees:

## String method with swarms of trajectories and free energy analysis
To run and setup string method simulations, see the instructions in the *string-method* directory.

An updated version of the string-method with new powerful features is on its way. Stay tuned!

## Clustering and Deep Taylor decomposition
To perform clustering and Deep Taylor decomposition, see the directory *clustering-analysis*. 
This code is however outdated. If you want to try these methods, we refer you to the following pages:
 * Learning important features from biomolecular simulations in general [Demystifying](https://github.com/mkasimova/Neural.Network.Relevance.Propagation)
 * For awesome clustering methods: [Annie Westerlund's github page](https://github.com/anniewesterlund?tab=repositories)
 * For implementing Deep Taylor decomposition/LRP: [Heatmapping.org](http://www.heatmapping.org/tutorial/)

# Citing this code
Please cite: {TO BE UPDATED}. 

# Contributors
 * Oliver Fleetwood (string method and clustering analysis)
 * Annie Westerlund (clustering)

# Contact information
For questions/issues with the code, open an issue here on github or contact oliver.fleetwood (at) gmail.com. 
For questions about the results of this study and biological insight of the paper you may also contact the corresponding author of the paper. 


# Code dependencies to install via e.g. anaconda
 * Python 2.7
 * msmsbuilder (requires mdtraj and numpy)
 * matplotlib
 * scikit-learn
 * joblib
 * dill

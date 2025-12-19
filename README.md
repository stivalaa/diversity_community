Are Community and Diversity Incompatible? Modeling Social Network Formation and Cultural Dissemination in Heterogeneous Populations
# Diversity and Community Can Co-exist

## Software

Imported from https://sites.google.com/site/alexdstivala/home/communitydiversity

Also available from Zenodo with DOI:

[![DOI](https://zenodo.org/badge/132078819.svg)](https://doi.org/10.5281/zenodo.17993440)



This software is free under the terms of the GNU General Public License.
It is derived from code developed for an earlier publication
[Ultrametric distribution of culture vectors in an extended Axelrod model of cultural dissemination](http://munk.cis.unimelb.edu.au/~stivalaa/ultrametric_axelrod/).
It uses Python
and parallelization using MPI (with [mpi4py](http://mpi4py.scipy.org/)). It also requires the Python libraries [NumPy](http://www.numpy.org/) (part of the [SciPy package](http://www.scipy.org/)) and 
[igraph](http://igraph.sourceforge.net/).

The Python code was run with NumPy version 1.6.2, SciPy version 0.7.2, igraph version 0.6 and mpi4py version 1.3.1 under Python version 2.6.6 on an IBM iDataplex x86 cluster (1120 Intel Sandybridge cores running at 2.7GHz) running Linux (RHEL 6) with Open MPI version 1.6.5.
The C++ code was compiled with gcc version 4.9.1. 
### Running the models

The paper describes two models:

- Model 1: Extended Schelling model with mean similarity. This model can be run with a command line such as: `mpirun --mca mpi_warn_on_fork 0 python ./physicaa2012-python-mpi/src/axelrod/geo/schelling/main.py F:1 ./lattice-schelling-meanoverlap-network/model end 500`

- Model 2: Extended Axelrod-Schelling model. This model can be run with a command line such as: `mpirun --mca mpi_warn_on_fork 0 python ./physicaa2012-python-mpi/src/axelrod/geo/expphysicstimeline/multiruninitmain.py F:5 ./lattice-axelrod-schelling-nealnetwork-cpp-end/model end 500`

## Reference

If you use our software, data, or results in your research, please cite:

- Stivala, A., Robins, G., Kashima, Y., and Kirley, M., 2016,
[Diversity and Community Can Coexist](http://onlinelibrary.wiley.com/doi/10.1002/ajcp.12021/abstract) Am. J. Commun. Psychol. 57:243-254


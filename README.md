# BayesBay phase velocity inversion tools

This repo contains scripts for performing Bayesian trans-dimensional inversions of Love or Rayleigh wave phase velocity with example slurm submission scripts.

This set of scripts uses the python package bayesbay to perform the inversion and uses the slurm array submission to parallelise many inversions at once.

The data is stored in a the data directory of TT or ZZ as seen in the example directory. As default the output is saved to chains and stats of either TT or ZZ. The directory chains will contain the full Markov chain of the inversion saved as a python pickle object and the stats will save the depth, mean, median and std in a text file.

## Usage
To run an inversion set up the input data according to the example directory then edit the paths in the two example submission scripts and tune the settings in invert_1d_phase.py. To change the number of chains per inversion eddit the number of processes in the submission script (#SBATCH --ntasks=8). 

I would recomend runing the inversion for a few selected dispersion curves first as some tuning will be required to get the acceptance rate to between 20-40%. 

## Requirements
This package uses bayesbay ([https://pypi.org/project/bayesbay/](https://pypi.org/project/bayesbay/)) and disba ([https://pypi.org/project/disba/](https://pypi.org/project/disba/)) as the primary engines for doing the inverse and the forward problem respectively. This set of scripts was developed in a python 3.8.2 environment with the versions of the relevant packages listed in requirements.txt, however it is quite a simple set of scripts meaning that so as long as no major changes are made to these libraries a number of different version combinations are likely to work.
import bayesbay as bb
from bayesbay.discretization import Voronoi1D
import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
import argparse
import pickle
import os

import bayesbay as bb
from bayesbay.discretization import Voronoi1D
import numpy as np
from disba import PhaseDispersion
from disba._exception import DispersionError
import argparse
import pickle
import os

###############################################################################################
#                                          SETTINGS                                           #
###############################################################################################

# Global value of Vp/Vs ratio (used to determine Vp and density in forward problem)
VP_VS = 1.78

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Observations ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# All observation information passed as arguments in argparse see ARG-PARSER section

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Inversion Settings ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
n_itterations = 3_000
burnin_iterations = 1_000
save_every = 100
# n_chains = 20

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Tuning ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Velocity and anisotropy perturbation standard deviation
vel_perturb_std = 0.42
ani_perturb_std = 0.04

# Voronoi cells
# Depth extent of inversion
dmin = 0 # min depth km
dmax = 15 # max depth km


move_cell_perturb_std = 4 # km


n_layers = None # None for fixed number of layers that can be purturbed
min_nlayers = 2
max_nlayers = 20
birth_from = "neighbour" # "prior" or "neighbour"
# birth_from = "prior"

# Data error
# Specify data error using a heirarchical scaling term
use_hierarchical_error = False
std_min = 0.001   # Min hierarchical error
std_max = 0.1     # Max hierarchical error
std_perturb_std = 0.002  # Hierarchical error proposal width
# Specify data error from file
use_cov_matrix = True


# Velocity Prior
# prior_type can be "gaussian" or "uniform"
vel_prior_type = "uniform"

# uniform
vel_position = [0,16] # list of depths in km
vel_vmin = [1.5,2.0] # list of coresponding minimum velocities for uniform prior
vel_vmax = [3.7,4.8] # list of coresponding maximum velocities for uniform prior

# gaussian
vel_position = [0,16] # list of depths in km
vel_vmean = [2.8,3.5] # list of coresponding mean velocities for gaussian prior
vel_vstd = [1.0,1.0] # list of coresponding standard deviations for gaussian prior

# Anisotropy prior
# prior_type can be "gaussian" or "uniform"
ani_prior_type = "gaussian"

# uniform
ani_position = [0,16]
ani_vmin = [-0.5,-0.5]
ani_vmax = [0.5,0.5]

# gaussian
ani_position = [0,16]
ani_vmean = [0,0]
ani_vstd = [0.4,0.4]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Postprocessing Settings ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
numd = 89 # number of bins in depth for mean/median/std etc

###############################################################################################
#                                          ARG-PARSER                                         #
###############################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('-Z', '--ZZ_inputfile', type = str, help = "input ZZ dat file")
parser.add_argument('-T', '--TT_inputfile', type = str, help = "input TT dat file")
parser.add_argument('-O', '--output', type = str, help = "output directory")

parser.add_argument('-N', '--inversion_number', type = int, help = "number of inversion to save results to, e.g. the number of the array in a slurm job")
parser.add_argument('-C', '--n_chains', type = int, default = 8, help = "number of chains in inversion, should be the same as ntasks")

# parser.add_argument('-W', '--wave_type', type = str, default = "rayleigh", help = "wave type, rayleigh or love")

args = parser.parse_args()

ZZ_inputfile = args.ZZ_inputfile
TT_inputfile = args.TT_inputfile
output = args.output
inversion_number = args.inversion_number
n_chains = args.n_chains
# wave_type = args.wave_type

# Read from input file

ZZ_PERIODS, ZZ_d_obs, ZZ_d_std = np.loadtxt(ZZ_inputfile,unpack=True)
TT_PERIODS, TT_d_obs, TT_d_std = np.loadtxt(TT_inputfile,unpack=True)

# Make output dirs
os.makedirs(f"{output}/chains",exist_ok=True)
os.makedirs(f"{output}/stats",exist_ok=True)

###############################################################################################
#                                            PRIOR                                            #
###############################################################################################

print("Initialising Prior")

def initialize_vs_uniform(param, positions=None):
    unstable = True
    while unstable:
        vmin, vmax = param.get_vmin_vmax(positions)
        thickness = Voronoi1D.compute_cell_extents(positions)
        sorted_vals = np.sort(np.random.uniform(vmin, vmax, positions.size))
        vs = sorted_vals
        vp = vs * VP_VS
        rho = 0.32 * vp + 0.77
        try:
            pd = PhaseDispersion(thickness, vp, vs, rho)
            d_pred = pd(TT_PERIODS, mode=0, wave="rayleigh").velocity
            d_pred = pd(ZZ_PERIODS, mode=0, wave="love").velocity
            unstable = False
            print("Found stable initialisation")
        except DispersionError:
            unstable = True
    return sorted_vals

def initialize_vs_gaussian(param, positions=None):
    vmean = param.get_mean(positions)
    vstd = param.get_std(positions)
    sorted_vals = np.sort(np.random.normal(vmean, vstd, positions.size))
    return sorted_vals

def initialize_ani(param, positions=None):
    vals = np.zeros(positions.size)
    return vals

# ~~~~~~~~~~~~~~~ Velocity ~~~~~~~~~~~~~~~ #

if vel_prior_type == "uniform":
    vs = bb.prior.UniformPrior(name="vs", 
                                        vmin=vel_vmin, 
                                        vmax=vel_vmax, 
                                        position=vel_position,
                                        perturb_std=vel_perturb_std)
    vs.set_custom_initialize(initialize_vs_uniform)
elif vel_prior_type == "gaussian":
    vs = bb.prior.GaussianPrior(name="vs",
                                        mean=vel_vmean,
                                        std=vel_vstd,
                                        position=vel_position,
                                        perturb_std=vel_perturb_std)
    vs.set_custom_initialize(initialize_vs_gaussian)
else:
    raise ValueError(f"Prior type {vel_prior_type} not recognised for vel use 'uniform' or 'gaussian'")

# ~~~~~~~~~~~~~~~ Anisotropy ~~~~~~~~~~~~~~~ #

if ani_prior_type == "uniform":
    ani = bb.prior.UniformPrior(name="ani", 
                                        vmin=ani_vmin, 
                                        vmax=ani_vmax, 
                                        position=ani_position,
                                        perturb_std=ani_perturb_std)
    ani.set_custom_initialize(initialize_ani)
elif ani_prior_type == "gaussian":
    ani = bb.prior.GaussianPrior(name="ani",
                                        mean=ani_vmean,
                                        std=ani_vstd,
                                        position=ani_position,
                                        perturb_std=ani_perturb_std)
    ani.set_custom_initialize(initialize_ani)
else:
    raise ValueError(f"Prior type {ani_prior_type} not recognised use 'uniform' or 'gaussian'")

voronoi = Voronoi1D(
    name="voronoi",
    vmin=dmin,                           # Min Depth (km)
    vmax=dmax,                           # Max Depth (km)
    perturb_std=move_cell_perturb_std,   # Cell change position purtubation std (km)
    n_dimensions=n_layers,               # None for flexible layers
    n_dimensions_min=min_nlayers,        # Min number of layers
    n_dimensions_max=max_nlayers,        # Max number of layers
    parameters=[vs,ani],                 # Prior vs and anisotropy
    birth_from=birth_from                # birth from 'prior' or 'neighbour'
)
parameterization = bb.parameterization.Parameterization(voronoi)

###############################################################################################
#                                Setup target and run inversion                               #
###############################################################################################

print("Initialising Target")
len_ZZ = len(ZZ_PERIODS)

d_obs = np.concatenate((ZZ_d_obs,TT_d_obs))
if use_cov_matrix:
    d_std = np.concatenate((ZZ_d_std,TT_d_std))
    inv_cov_mat = np.diag(1/np.power(d_std,2))

def forward_sw(state):
    voronoi = state["voronoi"]
    voronoi_sites = voronoi["discretization"]
    thickness = Voronoi1D.compute_cell_extents(voronoi_sites)
    vsv = voronoi["vs"]
    ani = voronoi["ani"]
    vsh = vsv+ani
    # ZZ
    vp = vsv * VP_VS
    rho = 0.32 * vp + 0.77
    pd = PhaseDispersion(thickness, vp, vsv, rho)
    ZZ_d_pred = pd(ZZ_PERIODS, mode=0, wave="rayleigh").velocity
    if len(ZZ_d_pred) != len(ZZ_PERIODS):
        raise DispersionError("Length of ZZ_d_pred != length of ZZ_PERIODS")
    # TT
    vp = vp = vsh * VP_VS
    rho = 0.32 * vp + 0.77
    pd = PhaseDispersion(thickness, vp, vsh, rho)
    TT_d_pred = pd(TT_PERIODS, mode=0, wave="love").velocity
    if len(TT_d_pred) != len(TT_PERIODS):
        raise DispersionError("Length of TT_d_pred != length of TT_PERIODS")
    #
    d_pred = np.concatenate((ZZ_d_pred,TT_d_pred))
    return d_pred

if use_hierarchical_error:
    target = bb.Target("data_vector", 
                d_obs, 
                std_min=std_min, 
                std_max=std_max, 
                std_perturb_std=std_perturb_std
                )
else:
    if use_cov_matrix:
        target = bb.Target("data_vector", 
                d_obs, 
                covariance_mat_inv=inv_cov_mat
                )
    else:
        raise ValueError("If no hierarcical scaling must use cov-matrix")

log_likelihood = bb.LogLikelihood(targets=target, fwd_functions=forward_sw)

print("Initialising Inversion")
inversion = bb.BayesianInversion(
    parameterization=parameterization, 
    log_likelihood=log_likelihood,
    n_chains=n_chains
)
print(f"Running inversion over {n_chains} chains for {n_itterations} itterations")
inversion.run(
    sampler=None, 
    n_iterations=n_itterations, 
    burnin_iterations=burnin_iterations, 
    save_every=save_every,
    verbose=False
)
for chain in inversion.chains:
    chain.print_statistics()

###############################################################################################
# 
###############################################################################################

interp_depths = np.linspace(dmin, dmax, numd)

results = inversion.get_results(concatenate_chains=True)

# Save results as pickle file
with open(f"{output}/chains/results_{inversion_number}.pkl","wb") as f:
    pickle.dump(results,f)

sampled_voronoi_nuclei = results['voronoi.discretization']
sampled_thickness = [Voronoi1D.compute_cell_extents(n) for n in sampled_voronoi_nuclei]
sampled_vs = results['voronoi.vs']
sampled_ani = results['voronoi.ani']

statistics_vs = Voronoi1D.get_tessellation_statistics(
    sampled_thickness, sampled_vs, interp_depths, input_type='extents'
    )
statistics_ani = Voronoi1D.get_tessellation_statistics(
    sampled_thickness, sampled_ani, interp_depths, input_type='extents'
    )

with open(f"{output}/stats/depth_meanvs_medianvs_stdvs_meanani_medianani_stdani_{inversion_number}.txt","w") as f:
    for depth, mean_vs, median_vs, std_vs, mean_ani, median_ani, std_ani in zip(interp_depths,statistics_vs["mean"],statistics_vs["median"],statistics_vs["std"],
                                        statistics_ani["mean"],statistics_ani["median"],statistics_ani["std"]):
        f.write(f"{depth} {mean_vs} {median_vs} {std_vs} {mean_ani} {median_ani} {std_ani}\n")
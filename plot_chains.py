from bayesbay.discretization import Voronoi1D
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

# Global value of Vp/Vs ratio (used to determine Vp and density in forward problem)
VP_VS = 1.78

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Settings ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Directory containing TT or ZZ
basepath = "./"
# Component suffix (eg directory could be TT_v1 making "_v1" the component suffix)
comp_suffix = ""

#Tesselation density min and max for plot
tp_vmin = 0
tp_vmax = 4000

#Prior
plot_prior_width = False
pr_position = [0,15]
pr_vmin = [1.5,2.0]
pr_vmax = [3.3,4.20]

n_chains = 8

brunin = 175_000
n_itterations = 800_000

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Main ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--inversion_number', type = int, help = "number of inversion to save results to, e.g. the number of the array in a slurm job")
parser.add_argument('-W', '--wavetype', type = str, default = "rayleigh", help = "wave type, rayleigh or love")
parser.add_argument('-D', '--max_dims', type = int, help = "Maximum number of dimensions")
args = parser.parse_args()

# Define component 
if args.wavetype == "rayleigh":
    comp = "ZZ"
elif args.wavetype == "love":
    comp = "TT"
else:
    raise ValueError(f"wavetype {args.wavetype} is not valid, use rayleigh or love")

#Read in inversion results
print("Reading in chains and stats")
with open(f"{basepath}/{comp}{comp_suffix}/chains/results_{args.inversion_number}.pkl","rb") as f:
    results = pickle.load(f)
interp_depths, mean, median, std = np.loadtxt(f"{basepath}/{comp}{comp_suffix}/stats/depth_mean_median_std_{args.inversion_number}.txt",unpack=True)
PERIODS, phasevel, phasevel_error = np.loadtxt(f"{basepath}/{comp}{comp_suffix}/data/data_num_{args.inversion_number}.dat",unpack=True)

# Reproduce stats for plot
print("Producing stats from all chains")
sampled_voronoi_nuclei = results['voronoi.discretization']
sampled_thickness = [Voronoi1D.compute_cell_extents(n) for n in sampled_voronoi_nuclei]
sampled_vs = results['voronoi.vs']
statistics_vs = Voronoi1D.get_tessellation_statistics(
    sampled_thickness, sampled_vs, interp_depths, input_type='extents'
    )


for key in results:
    print(key)

print(phasevel.shape)
print(phasevel_error.shape)
dpred = results["rayleigh.dpred"]
print(len(dpred))
n_dim = results["voronoi.n_dimensions"]
chlen = int(len(n_dim)/n_chains)

fig = plt.figure(figsize=(12,5))
plt.grid(True)
for i in range(n_chains-1):
    itters = brunin + 100*np.arange(0,len(n_dim[i*chlen:i*chlen+chlen]),1)
    plt.plot(itters,n_dim[i*chlen:i*chlen+chlen],linewidth=0.5,alpha=0.2)
plt.ylim(ymin=0,ymax=15)
plt.ylabel("N Dim")
plt.xlabel("Itteration")
fig.savefig(f"./{args.inversion_number}_ndim.png")
plt.close()


cov = (1/(phasevel_error**2))*np.ones(len(phasevel_error))
cov = np.diag(cov)

fig = plt.figure(figsize=(12,5))
plt.grid(True)
for i in range(n_chains-1):
    chain_dpred = dpred[i*chlen:i*chlen+chlen]
    likelihood = []
    for j in range(len(chain_dpred)):
        j_dpred = chain_dpred[j]
        res = j_dpred - phasevel
        neg_log_likelihood = np.dot(res.T,cov)
        neg_log_likelihood = np.dot(neg_log_likelihood,res)
        likelihood.append(neg_log_likelihood)
    likelihood = np.array(likelihood)
    plt.plot(itters,likelihood,linewidth=0.7)
plt.ylabel("Negative Log Likelihood")
plt.xlabel("Itteration")
fig.savefig(f"./{args.inversion_number}_likelihood.png")
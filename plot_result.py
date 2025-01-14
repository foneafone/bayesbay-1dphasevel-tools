import bayesbay as bb
from bayesbay.discretization import Voronoi1D
import numpy as np
import matplotlib.pyplot as plt
from disba import PhaseDispersion
import argparse
import pickle
import multiprocessing
import os
from tqdm import tqdm

VP_VS = 1.78
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Settings ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

basepath = "/home/jwf39/rds/rds-nr441-fone/askja_1d/bayesbay1d"
comp_suffix = "_fmst_v3"

#Tesselation density
tp_vmin = 0
tp_vmax = 4000

#Prior
plot_prior_width = False
pr_position = [0,15]
pr_vmin = [1.5,2.0]
pr_vmax = [3.3,4.20]

best_N = 0

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
print("Reading in cains and stats")
with open(f"{basepath}/{comp}{comp_suffix}/chains/results_{args.inversion_number}.pkl","rb") as f:
    results = pickle.load(f)
interp_depths, mean, median, std = np.loadtxt(f"{basepath}/{comp}{comp_suffix}/stats/depth_mean_median_std_{args.inversion_number}.txt",unpack=True)
PERIODS, phasevel, phasevel_error = np.loadtxt(f"{basepath}/{comp}{comp_suffix}/data/data_num_{args.inversion_number}.dat",unpack=True)

# Reproduce stats for plot
print("Producin stats from all chains")
sampled_voronoi_nuclei = results['voronoi.discretization']
sampled_thickness = [Voronoi1D.compute_cell_extents(n) for n in sampled_voronoi_nuclei]
sampled_vs = results['voronoi.vs']
statistics_vs = Voronoi1D.get_tessellation_statistics(
    sampled_thickness, sampled_vs, interp_depths, input_type='extents'
    )

# Produce stats from best_N results
if best_N > 0:
    print(f"Producing stats from best {best_N} chains")
    def worker(i,thickness,vs,d_obvs):
        vp = vs * VP_VS
        rho = 0.32 * vp + 0.77
        pd = PhaseDispersion(thickness, vp, vs, rho)
        d_pred = pd(PERIODS, mode=0, wave=args.wavetype).velocity
        rms = np.sqrt(np.mean((d_pred-d_obvs)*(d_pred-d_obvs)))
        return i, rms
    #
    if __name__ == "__main__":
        with multiprocessing.Pool(20) as pool:
            procs = []
            rms_list = np.zeros((len(sampled_thickness)))
            for i, (thickness,vs) in enumerate(zip(sampled_thickness,sampled_vs)):
                p = pool.apply_async(worker,args=(i,thickness,vs,phasevel))
                procs.append(p)
            for p in tqdm(procs):
                i, rms = p.get()
                rms_list[i] = rms
        sorted_inds = np.argsort(rms_list)
        inds = np.array(sorted_inds[:best_N],dtype=int)
        best_sampled_thickness = [sampled_thickness[i] for i in inds]
        best_sampled_vs = [sampled_vs[i] for i in inds]
    #
    best_statistics_vs = Voronoi1D.get_tessellation_statistics(
        best_sampled_thickness, best_sampled_vs, interp_depths, input_type='extents'
        )

#Plot model and posterior
print("Plottig model and posterior")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 8), gridspec_kw={'width_ratios': [2.5, 1]})
#Axes 1 - model and posterior
ax1, cbar = Voronoi1D.plot_tessellation_density(sampled_thickness, 
                                                sampled_vs, 
                                                input_type='extents', 
                                                ax=ax1, 
                                                cmap='binary', 
                                                vmin=tp_vmin, 
                                                vmax=tp_vmax)
ax1.plot(statistics_vs['mean'], interp_depths, 'b', lw=2, label='Vs Ensemble Mean')
ax1.plot(statistics_vs['median'], interp_depths, 'r', lw=2, label='Vs Ensemble Median')
ax1.plot(mean+std, interp_depths, "r", alpha=0.5, lw=1.5, label="Uncertainty")
ax1.plot(mean-std, interp_depths, "r", alpha=0.5, lw=1.5)
if best_N > 0:
    ax1.plot(best_statistics_vs['mean'], interp_depths, 'c--', lw=2, label='Vs Best Ensemble Mean')
    ax1.plot(best_statistics_vs['median'], interp_depths, 'm--', lw=2, label='Vs Best Ensemble Median')
    ax1.plot(best_statistics_vs['mean']+statistics_vs["std"], interp_depths, "r--", alpha=0.5, lw=1.5, label="Best Uncertainty")
    ax1.plot(best_statistics_vs['mean']-statistics_vs["std"], interp_depths, "r--", alpha=0.5, lw=1.5)
if plot_prior_width:
    ax1.plot(pr_vmin,pr_position,color="blue")
    ax1.plot(pr_vmax,pr_position,color="blue")
ax1.set_ylim(ymax=np.min(interp_depths),ymin=np.max(interp_depths))
ax1.set_xlabel("Vs [km/s]")
ax1.set_ylabel("Depth [km]")
ax1.set_title(f"Posterior for model {args.inversion_number}")
ax1.legend()
#Axes 2 - interface histogram
Voronoi1D.plot_interface_hist(sampled_voronoi_nuclei, bins=75, ec='w', ax=ax2)
ax2.tick_params(labelleft=False)
ax2.set_ylabel('')
ax2.set_ylim(*ax1.get_ylim())
plt.tight_layout()
plt.show()
fig.savefig("./askja_model_result.png")
plt.close()

# Data plot
print("Plotting data fit")
thick = (np.max(interp_depths)-np.min(interp_depths))/len(interp_depths)
# vs_mean = statistics_vs['mean'] - 0.2
vp = statistics_vs['mean'] * VP_VS
rho = 0.32 * vp + 0.77
pd = PhaseDispersion(thick*np.ones_like(interp_depths), vp, statistics_vs['mean'], rho)
# pd = PhaseDispersion(thick*np.ones_like(interp_depths), vp, vs_mean, rho)
mean_phase_vel = pd(PERIODS, mode=0, wave=args.wavetype).velocity

vp = statistics_vs['median'] * VP_VS
rho = 0.32 * vp + 0.77
pd = PhaseDispersion(thick*np.ones_like(interp_depths), vp, statistics_vs['median'], rho)
median_phase_vel = pd(PERIODS, mode=0, wave=args.wavetype).velocity

if best_N > 0:
    vp = best_statistics_vs['mean'] * VP_VS
    rho = 0.32 * vp + 0.77
    pd = PhaseDispersion(thick*np.ones_like(interp_depths), vp, best_statistics_vs['mean'], rho)
    best_mean_phase_vel = pd(PERIODS, mode=0, wave=args.wavetype).velocity
    #
    vp = best_statistics_vs['median'] * VP_VS
    rho = 0.32 * vp + 0.77
    pd = PhaseDispersion(thick*np.ones_like(interp_depths), vp, best_statistics_vs['median'], rho)
    best_median_phase_vel = pd(PERIODS, mode=0, wave=args.wavetype).velocity

fig = plt.figure(figsize=(8,6))
plt.plot(PERIODS, median_phase_vel, 'g--', label='From Median', lw=0.5)
plt.plot(PERIODS, mean_phase_vel, 'b--', label='From Mean', lw=0.5)
if best_N > 0:
    plt.plot(PERIODS, best_median_phase_vel, 'c--', label='From Best Median', lw=0.5)
    plt.plot(PERIODS, best_mean_phase_vel, 'm--', label='From Best Mean', lw=0.5)
plt.errorbar(PERIODS, phasevel, yerr=phasevel_error, fmt="o", color="r", label=f'{args.wavetype} obs.')
plt.xlabel("Periods [s]")
plt.ylabel("Phase Velocity [km/s]")
plt.title(f"Data fit for model {args.inversion_number}")
plt.tight_layout()
plt.legend()
fig.savefig("./askja_avg_fit.png")
plt.close()

# Number of dimensions
print("Plotting dimension histogram")
ndim = results["voronoi.n_dimensions"]
fig = plt.figure(figsize=(8,6))
bins = np.arange(0,args.max_dims+1,1)
plt.hist(ndim,bins=bins,rwidth=0.8)
plt.xlabel("Number of dimensions")
plt.ylabel("Count")
plt.title(f"Number of dimensions for model {args.inversion_number}")
plt.tight_layout()
fig.savefig("./askja_num_dim.png")
plt.close()
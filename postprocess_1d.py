from bayesbay.discretization import Voronoi1D
import numpy as np
import pickle
import xarray as xr
import json
from tqdm import tqdm
from multiprocessing import Pool

# Script for re-constructing 3D grid of shear wave velocity using inversion output
# - no example provided but this is not important to the inversion and is just how
#   I have merged things.

###############################################################################################
#                                          SETTINGS                                           #
###############################################################################################

comp = "TT"
# Directory containing TT or ZZ and where output grids will apear
basepath = "./"
# Component suffix (eg directory could be TT_v1 making "_v1" the component suffix)
comp_suffix = ""

# Json of the form:
# {
#   "periods" : (2,4,6 ... etc), # (period values in seconds)
#   0 : {"lat": 64.81, "lon": -17.18, "i": 0, "j": 0,                   # Location (lat/lon) and index in grid (i=lat index, j=lon index)
#           "mean" : (2.3, 2.4 ... etc), "std" : (0.1, 0.12 ... etc)},  # phase velocity values and their standard deviation in km/s 
#   1 : .....
# }
# where 0, 1 etc represents the number of the pseudo dispersion curve being inverted for
input_json = f"./{comp}{comp_suffix}/data/pseudo_curves_phase_{comp}.json"

# Netcdf4 grid with latidude and longitude of all the grid points with dimensions ("depth","lat","lon")
base_grid = f"./{comp}{comp_suffix}/data/base_vsgrid_{comp}.nc"

# Number of points in the depth axis
num_depth = 89

# Use the output of the chains to calculate rather than the stats
use_results = False

threads = 10

###############################################################################################
#                                            MAIN                                             #
###############################################################################################

base_grid = xr.load_dataarray(base_grid)

with open(input_json,"r") as f:
        data_dict = json.load(f)

periods = data_dict.pop("periods",False)

out_grid_mean = np.zeros((num_depth,base_grid.data.shape[0],base_grid.data.shape[1]))
out_grid_median = np.zeros((num_depth,base_grid.data.shape[0],base_grid.data.shape[1]))
out_grid_std = np.zeros((num_depth,base_grid.data.shape[0],base_grid.data.shape[1]))

def worker(num):
    try:
        if use_results:
            depth, mean, median, std = np.loadtxt(f"{basedir}/{comp}{comp_suffix}/stats/depth_mean_median_std_{num}.txt",unpack=True)
            depth = np.linspace(depth.min(),depth.max(),num_depth)
            with open(f"{basedir}/{comp}{comp_suffix}/chains/results_{num}.pkl","rb") as f:
                results = pickle.load(f)
            sampled_voronoi_nuclei = results['voronoi.discretization']
            sampled_thickness = [Voronoi1D.compute_cell_extents(n) for n in sampled_voronoi_nuclei]
            sampled_vs = results['voronoi.vs']
            statistics_vs = Voronoi1D.get_tessellation_statistics(
                sampled_thickness, sampled_vs, depth, input_type='extents'
            )
            mean, median, std = statistics_vs["mean"], statistics_vs["median"], statistics_vs["std"]
        else:
            depth, mean, median, std = np.loadtxt(f"{basedir}/{comp}{comp_suffix}/stats/depth_mean_median_std_{num}.txt",unpack=True)
    except FileNotFoundError:
        print(f"Inversion number {num} result not found filling with nans")
        depth = np.nan*np.ones((num_depth))
        mean = np.nan*np.ones((num_depth))
        median = np.nan*np.ones((num_depth))
        std = np.nan*np.ones((num_depth))
    return num, depth, mean, median, std

with Pool(threads) as pool:
    procs = []
    for num in data_dict:
        p = pool.apply_async(worker,args=(num,))
        procs.append(p)
    #
    for p in tqdm(procs):
        num, depth, mean, median, std = p.get()
        i = data_dict[num]["i"]
        j = data_dict[num]["j"]
        #
        out_grid_mean[:,i,j] = mean
        out_grid_median[:,i,j] = median
        out_grid_std[:,i,j] = std
#
out_grid_mean = xr.DataArray(
    data=out_grid_mean,
    dims=("depth","lat","lon"),
    coords=dict(
        depth=depth,
        lat=base_grid.coords["lat"].data,
        lon=base_grid.coords["lon"].data
    )
)
out_grid_mean.to_netcdf(f"{basedir}/{comp}{comp_suffix}/stats/mean_grid_{comp}.nc")
out_grid_median = xr.DataArray(
    data=out_grid_median,
    dims=("depth","lat","lon"),
    coords=dict(
        depth=depth,
        lat=base_grid.coords["lat"].data,
        lon=base_grid.coords["lon"].data
    )
)
out_grid_median.to_netcdf(f"{basedir}/{comp}{comp_suffix}/stats/median_grid_{comp}.nc")
out_grid_std = xr.DataArray(
    data=out_grid_std,
    dims=("depth","lat","lon"),
    coords=dict(
        depth=depth,
        lat=base_grid.coords["lat"].data,
        lon=base_grid.coords["lon"].data
    )
)
out_grid_std.to_netcdf(f"{basedir}/{comp}{comp_suffix}/stats/std_grid_{comp}.nc")
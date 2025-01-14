import json
import argparse
import multiprocessing
from tqdm import tqdm

def write_out(key,Ts,vs,ss,path):
    with open(path,"w") as f:
        for T,v,s in zip(Ts,vs,ss):
            f.write(f"{T} {v} {s}\n")
    return 0

# Optional script to read in a json file containing pseudo dispersion curves and save it to an input data
# directory for invert_1d_phase.py to use.
# Json of the form:
# {
#   "periods" : (2,4,6 ... etc), # (period values in seconds)
#   0 : {"lat": 64.81, "lon": -17.18, "i": 0, "j": 0,                   # Location (lat/lon) and index in grid (i=lat index, j=lon index)
#           "mean" : (2.3, 2.4 ... etc), "std" : (0.1, 0.12 ... etc)},  # phase velocity values and their standard deviation in km/s 
#   1 : .....
# }
# where 0, 1 etc represents the number of the pseudo dispersion curve being inverted for


parser = argparse.ArgumentParser()
parser.add_argument('-I', '--input', type = str, help = "input json file")
parser.add_argument('-O', '--output', type = str, help = "output directory")
parser.add_argument('-T', '--threads', type = int, default = 10, help = "number of threads")
args = parser.parse_args()



with open(args.input,"r") as f:
    data_dict = json.load(f)

periods = data_dict.pop("periods",False)
if periods == False:
    raise ValueError("periods does not exist in data json")

with multiprocessing.Pool(args.threads) as pool:
    procs = []
    for key in data_dict:
        psudo_curve_dict = data_dict[key]
        path = f"{args.output}/data_num_{int(key)}.dat"
        vs = psudo_curve_dict["mean"]
        ss = psudo_curve_dict["std"]
        p = pool.apply_async(write_out,args=(key,periods,vs,ss,path))
        procs.append(p)
    for p in tqdm(procs):
        out = p.get()
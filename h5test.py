import h5py
import numpy as np
import pdb
import matplotlib.pyplot as plt

filename = '/work/bmaier/CASE/run2_mixed_MC_sig_bkg/train/mixed_sig_bkg_1.h5'

print(f'Opening {filename} to look what\'s inside ...')

with h5py.File(filename, "r") as f:
    for key in f.keys():
        print(f'Key: {key} with shape: {f[key][()].shape}')
    
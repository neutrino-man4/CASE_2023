import h5py
import numpy as np
import pdb
import matplotlib.pyplot as plt

dir = '/work/abal/CASE/VAE_results/model_analysis/run_98765/auc/'
filename = 'qcdSigMCOrigReco_QstarToQW_M_3000_mW_80Reco.h5'

print(f'Opening {filename} to look what\'s inside ...')

with h5py.File(dir+filename, "r") as f:
    for key in f.keys():
        print(f'Key: {key} with shape: {f[key][()].shape}')
    
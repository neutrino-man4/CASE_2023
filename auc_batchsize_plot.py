# Minimal plotting script for AUC vs batch size

import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess

model_dir = '/work/abal/CASE/VAE_results/model_analysis/auc_vs_batch'
plot_dir=os.path.join(model_dir,"plots/")

files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
extraticks=[128,256,512,1024,2048,4096]
for f in files:
    sig_sample=os.path.splitext(f)[0].replace('Reco','')
    print(f'Plotting for {sig_sample}')
    aucs=pd.read_csv(os.path.join(model_dir,f),names=['auc','batch_sz','seed'])
    aucs=aucs.sort_values('batch_sz')
    plt.plot(aucs['batch_sz'],aucs['auc'],'-ko',label=sig_sample)
    plt.ylim(0.5,1.) # Anything below 0.5 is useless
    plt.xlim(2**6.,2**13)
    plt.xlabel("Batch size")
    plt.ylabel("AUC")
    plt.title(sig_sample)
    plt.minorticks_on()
    plt.xticks(extraticks,fontsize='small')
    plt.xscale('log', base=2)
    plt.grid(which='major',color='b',alpha=0.6)
    #plt.grid(which='minor',color='k',linestyle='--',alpha=0.6)
    plt.savefig(f'{plot_dir}/{sig_sample}.png')
    plt.close()

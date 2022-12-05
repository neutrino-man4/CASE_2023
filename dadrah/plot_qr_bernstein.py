import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import pandas as pd
import mplhep as hep
import glob

plt.style.use(hep.style.CMS)
jet= plt.get_cmap('gnuplot')
colors = jet(np.linspace(0,1,6))

df = pd.read_csv("/data/t3home000/bmaier/CASE/QR_results/analysis/run_50000/sig_WkkToWRadionToWWW_M3000_Mr170Reco/xsec_0/loss_rk5_05/qr_cuts/mjj_vs_loss.txt")
#print(df)

mjj = df['mjj']
loss = df['loss']

x_min = np.min(mjj)
x_max = np.percentile(mjj, 1e2*(1-1e-4))
#x_max = np.max(mjj)

fig,ax = plt.subplots()

plt.hist2d(mjj, loss, range=((x_min , x_max), (np.min(loss), np.percentile(loss, 1e2*(1-1e-3)))), norm=LogNorm(), bins=200, cmap=cm.get_cmap('Blues'), cmin=0.01)    

xs = np.arange(x_min, x_max, 0.001*(x_max-x_min))


ccounter = 0
for i in [10,30,50,70,90,99]:
    files = glob.glob('/data/t3home000/bmaier/CASE/QR_results/analysis/run_50000/sig_WkkToWRadionToWWW_M3000_Mr170Reco/xsec_0/loss_rk5_05/qr_cuts/lambdaBS_*x0Q*%s*.txt'%(str(i)))
    #print('/data/t3home000/bmaier/CASE/QR_results/analysis/run_50000/sig_WkkToWRadionToWWW_M3000_Mr170Reco/xsec_0/loss_rk5_05/qr_cuts/lambdaV4_*x0Q*%s*.txt'%(str(i)))
    
    #print(files)

    for f in files:
        print("Reading",f)
        df_pred = pd.read_csv(f)
        if f == files[0]:
            plt.plot(df_pred['x'], df_pred['y'], '-', lw=2.5, label='Q %s'%str(i), color=colors[ccounter])
        else:
            plt.plot(df_pred['x'], df_pred['y'], '-', lw=2.5, color=colors[ccounter])

    ccounter += 1

plt.legend()
plt.ylabel('min(L1,L2)')
plt.xlabel('$M_{jj}$ (GeV)')
plt.text(0.06,0.89,'4th order pol.',transform=ax.transAxes)
#plt.title('quantile cuts' + title_suffix)                                                              
plt.colorbar()

plt.savefig("/home/tier3/bmaier/public_html/figs/case/kvae/run_50000/qr/plots_aggregate/qr_aggregate_lambdaBS.pdf", bbox_inches='tight')
plt.savefig("/home/tier3/bmaier/public_html/figs/case/kvae/run_50000/qr/plots_aggregate/qr_aggregate_lambdaBS.png", bbox_inches='tight', dpi=300)


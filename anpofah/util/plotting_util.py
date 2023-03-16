import matplotlib.pyplot as plt
from matplotlib import colors
import os
import numpy as np
import pandas as pd

import anpofah.util.data_preprocessing as dpr


def subplots_rows_cols(n):
    ''' get number of subplot rows and columns needed to plot n histograms in one figure '''
    return int(np.round(np.sqrt(n))), int(np.ceil(np.sqrt(n)))


def plot_hist(data, bins=100, xlabel='x', ylabel='num frac', title='histogram', plot_name='plot', fig_dir=None, legend=[], ylogscale=True, normed=True, ylim=None, legend_loc='best', xlim=None, clip_outlier=False, fig_format='.png'):
    fig = plt.figure()
    fig.set_size_inches(18,12)
    if clip_outlier:
        data = [dpr.clip_outlier(dat) for dat in data]
    counts, edges = plot_hist_on_axis(plt.gca(), data, bins=bins, xlabel=xlabel, ylabel=ylabel, title=title, legend=legend, ylogscale=ylogscale, normed=normed, ylim=ylim, xlim=xlim)
    if legend:
        plt.legend(loc=legend_loc,fontsize=9)
    plt.tight_layout()
    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, plot_name + fig_format))
    else:
        plt.show()
    plt.close(fig)
    return counts, edges


def plot_multihist(data, bins=100, suptitle='histograms', titles=[], clip_outlier=False, plot_name='histograms', fig_dir=None, fig_format='.pdf'):
    ''' plot len(data) histograms on same figure 
        data = list of features to plot (each element is flattened before plotting)
    '''
    rows_n, cols_n = subplots_rows_cols(len(data))
    fig, axs = plt.subplots(nrows=rows_n,ncols=cols_n, figsize=(9,9))
    for ax, dat, title in zip(axs.flat, data, titles):
        if clip_outlier:
            dat = dpr.clip_outlier(dat.flatten())
        plot_hist_on_axis(ax, dat.flatten(), bins=bins, title=title)
    [a.axis('off') for a in axs.flat[len(data):]] # turn off unused subplots
    plt.suptitle(suptitle)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    if fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, plot_name + fig_format))
    else:
        plt.show();
    plt.close(fig)


def plot_hist_on_axis(ax, data, bins, xlabel='', ylabel='', title='histogram', legend=[], ylogscale=True, normed=True, ylim=None, xlim=None):
    if ylogscale:
        ax.set_yscale('log', nonposy='clip')
    counts, edges, _ = ax.hist(data, bins=bins, normed=normed, histtype='step', label=legend)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    return counts, edges


def plot_hist_2d(x, y, xlabel='x', ylabel='num frac', title='histogram', plot_name='hist2d', fig_dir=None, legend=[], ylogscale=True, normed=True, ylim=None, legend_loc='best', xlim=None, clip_outlier=False):
    
    if clip_outlier:
        idx = dpr.is_outlier_percentile(x) | dpr.is_outlier_percentile(y)
        x = x[~idx]
        y = y[~idx]

    fig = plt.figure()
    ax = plt.gca()
    im = plot_hist_2d_on_axis( ax, x, y, xlabel, ylabel, title )
    fig.colorbar(im[3])
    plt.tight_layout()
    if fig_dir:
        plt.savefig(os.path.join(fig_dir,plot_name+'.png'))
    plt.show()
    plt.close(fig)
    return ax
    
    
def plot_hist_2d_on_axis(ax, x, y, xlabel, ylabel, title):
    im = ax.hist2d(x, y, bins=100, norm=colors.LogNorm())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im


def plot_bg_vs_sig(data, bins=100, xlabel='x', ylabel='num frac', title='histogram', plot_name='plot', fig_dir=None, legend=['bg','sig'], ylogscale=True, normed=True, legend_loc=1, clip_outlier=False, xlim=None, fig_format='.pdf'):
    '''
    plots feature distribution treating first data-array as backround and rest of arrays as signal
    :param data: list/array of N elements where first element is assumed to be background and elements 2..N-1 assumed to be signal. all elements = array of length M
    '''

    fig = plt.figure()
    fig.set_size_inches(10.,7.5)
    alpha = 0.4
    histtype = 'stepfilled'
    if ylogscale:
        plt.yscale('log')

    for i, dat in enumerate(data):
        if i > 0:
            histtype = 'step'
            alpha = 1.0
        if clip_outlier:
            idx = dpr.is_outlier_percentile(dat)
            dat = dat[~idx]
        plt.hist(dat, bins=bins, density=normed, alpha=alpha, histtype=histtype, label=legend[i])
        print(dat)
        print(legend[i])
        if legend[i] == 'WpToBpT_Wp3000_Bp400_Top170_ZbtReco' or legend[i] == 'qcdSigMCOrigReco' or legend[i] == 'RSGravitonToGluonGluon_kMpl01_M_3000Reco' or legend[i] == 'QstarToQW_M_2000_mW_400Reco' or legend[i] == 'WkkToWRadionToWWW_M3000_Mr170Reco':
            print("--------------------")
            print("--------------------")
            tmp_dict = {'loss':dat.tolist()}
            df = pd.DataFrame.from_dict(tmp_dict)
            csvname = "%s_%s.csv"%(title,legend[i])
            csvname = csvname.replace(" ","")
            df.to_csv("/work/abal/CASE/csv/%s"%csvname)

    if xlim:
        plt.xlim(xlim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.draw()
    if fig_dir:
        fig.savefig(os.path.join(fig_dir, plot_name + fig_format))
    plt.close(fig)


def plot_bg_vs_sig_multihist(data_bg, data_sig, subtitles, bins=100, suptitle='histograms', clip_outlier=False, normed=True, ylogscale=True, plot_name='multihist', fig_dir='fig', fig_format='.png'):
    '''
    plot background versus signal for multiple features as 1D histograms in one figure
    param data_bg: list of K features with each N background values
    param data_sig: list of K features with each N signal values
    '''
    rows_n, cols_n = subplots_rows_cols(len(data_bg))
    fig, axs = plt.subplots(nrows=rows_n, ncols=cols_n)

    for ax, d_bg, d_sig, title in zip(axs.flat, data_bg, data_sig, subtitles):
        if clip_outlier:
            d_bg = dpr.clip_outlier(d_bg.flatten())
            d_sig = dpr.clip_outlier(d_sig.flatten())
        ax.hist(d_bg, bins=bins, density=normed, alpha=0.6, histtype='stepfilled', label='BG')
        ax.hist(d_sig, bins=bins, density=normed, alpha=1.0, histtype='step', linewidth=1.3, label='SIG')
        if ylogscale:
            ax.set_yscale('log', nonpositive='clip')
        ax.set_title(title)

    for a in axs[:, 0]: a.set_ylabel('frac num events')
    [a.axis('off') for a in axs.flat[len(data_bg):]] # turn off unused subplots
    plt.suptitle(suptitle)
    # plt.legend(loc='best')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(fig_dir, plot_name + fig_format))
    plt.close(fig)

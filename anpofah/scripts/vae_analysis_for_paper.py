import pofah.util.experiment as ex
import pofah.path_constants.sample_dict_file_parts_reco as sdfr 
import pofah.util.sample_factory as sf
import anpofah.model_analysis.roc_analysis as ra
import anpofah.sample_analysis.sample_analysis as saan
import dadrah.selection.loss_strategy as lost
import anpofah.util.sample_names as samp

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import sklearn.metrics as skl
import os


def plot_roc(neg_class_losses, pos_class_losses_na, pos_class_losses_br, legend_str, title='ROC', legend_loc='best', plot_name='ROC', fig_dir=None, x_lim=None, log_x=True, fig_format='.png'):

    class_labels_na, losses_na = ra.get_label_and_score_arrays(neg_class_losses, pos_class_losses_na) # stack losses and create according labels per strategy
    class_labels_br, losses_br = ra.get_label_and_score_arrays(neg_class_losses, pos_class_losses_br) # stack losses and create according labels per strategy
    class_labels, losses = class_labels_na + class_labels_br, losses_na + losses_br

    palette = ['#3E96A1', '#EC4E20', '#FF9505', '#713E5A']
    colors = [palette[0]]*len(neg_class_losses) + [palette[1]]*len(neg_class_losses) # pick line color for narrow & broad resonances, times number of loss strategies to plot
    styles = ['dashed', 'dotted', 'solid'][:len(neg_class_losses)]*2 # take as many styles as there are loss strategies to plot, times two for narrow & broad

    aucs = []
    fig = plt.figure() # figsize=(5, 5)

    for y_true, loss, color, style in zip(class_labels, losses, colors, styles):
        fpr, tpr, threshold = skl.roc_curve(y_true, loss)
        aucs.append(skl.roc_auc_score(y_true, loss))
        if log_x:
            plt.loglog(tpr, 1./fpr, linestyle=style, color=color) # label=label + " (auc " + "{0:.3f}".format(aucs[-1]) + ")",
        else:
            plt.semilogy(tpr, 1./fpr, linestyle=style, color=color)

    dummy_res_lines = [Line2D([0,1],[0,1],linestyle='-', color=c) for c in palette[:2]]
    plt.semilogy(np.linspace(0, 1, num=100), 1./np.linspace(0, 1, num=100), linewidth=1.2, linestyle='solid', color='silver')
    
    # add 2 legends (vae score types and resonance types)
    lines = plt.gca().get_lines()
    legend1 = plt.legend(dummy_res_lines, ['Narrow', 'Broad'], loc='upper right', frameon=False, title=title, handlelength=1.5)
    legend2 = plt.legend([lines[i] for i in range(len(legend_str))], legend_str, loc='center right', frameon=False, title='AD score')
    for leg in legend1.legendHandles:
        leg.set_linewidth(2.2)
    for leg in legend2.legendHandles: 
        leg.set_color('black')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    plt.grid()
    if x_lim:
        plt.xlim(left=x_lim)
    plt.xlabel('True positive rate')
    plt.ylabel('1 / False positive rate')
    plt.tight_layout()
    if fig_dir:
        print('writing ROC plot to {}'.format(fig_dir))
        fig.savefig(os.path.join(fig_dir, plot_name + fig_format), bbox_inches='tight')
    plt.close(fig)
    return aucs



def plot_mass_center_ROC(bg_sample, sig_sample_na, sig_sample_br, mass_center, plot_name_suffix=None, fig_dir='fig', fig_format='.png'):
    ''' 
        plot ROC for narrow and broad signal (color)
        reco, kl and total combined loss (marker)
    ''' 

    _, bg_center_bin_sample, _ = ra.get_mjj_binned_sample(bg_sample, mass_center)
    _, sig_center_bin_sample_na, _ = ra.get_mjj_binned_sample(sig_sample_na, mass_center)
    _, sig_center_bin_sample_br, _ = ra.get_mjj_binned_sample(sig_sample_br, mass_center)

    # strategy_ids = ['r5', 'kl5', 'rk5']
    strategy_ids = ['r5', 'kl5']
    title_strategy_suffix = 'loss J1 && loss J2 > LT'
    # legend_str = ['Reco', r'$D_{KL}$', 'Combined']
    legend_str = ['Reco', r'$D_{KL}$']
    plot_name = '_'.join(filter(None, ['ROC', sig_sample_na.name.replace('Reco', 'br'), plot_name_suffix]))
    title = r'$G_{{RS}} \to WW \, m_{{G}} = {} TeV$'.format(mass_center/1000)
    log_x = False
    x_lim = None

    neg_class_losses = [lost.loss_strategy_dict[s_id](bg_center_bin_sample) for s_id in strategy_ids]
    pos_class_losses_na = [lost.loss_strategy_dict[s_id](sig_center_bin_sample_na) for s_id in strategy_ids]
    pos_class_losses_br = [lost.loss_strategy_dict[s_id](sig_center_bin_sample_br) for s_id in strategy_ids]

    plot_roc(neg_class_losses, pos_class_losses_na, pos_class_losses_br, legend_str=legend_str, title=title, plot_name=plot_name, fig_dir=fig_dir, x_lim=x_lim, log_x=log_x, fig_format=fig_format)



if __name__ == '__main__':    

    # setup analysis inputs
    run_n = 113
    # set background sample to use (sideband or signalregion)
    BG_sample = samp.BG_SR_sample
    mass_centers = [1500, 2500, 3500, 4500]
    plot_name_suffix = None

    # set up analysis outputs 
    experiment = ex.Experiment(run_n).setup(model_analysis_dir=True)
    paths = sf.SamplePathDirFactory(sdfr.path_dict).update_base_path({'$run$': experiment.run_dir})
    print('Running analysis on experiment {}, plotting results to {}'.format(run_n, experiment.model_analysis_dir))
    
    # read in data
    data = sf.read_inputs_to_jet_sample_dict_from_dir(samp.all_samples, paths, read_n=None)

    # Load CMS style sheet
    plt.style.use(hep.style.CMS)

    # *****************************************
    #                   ROC
    # *****************************************
    # for each mass center
    for SIG_sample_na, SIG_sample_br, mass_center in zip(samp.SIG_samples_na, samp.SIG_samples_br, mass_centers):
        plot_mass_center_ROC(data[BG_sample], data[SIG_sample_na], data[SIG_sample_br], mass_center, plot_name_suffix=plot_name_suffix, fig_dir=experiment.model_analysis_dir_roc)

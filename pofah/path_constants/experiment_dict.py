
path_dict = {

    # VAE directory patterns (TODO: fix run str)
	
	'model_dir' : '/data/t3home000/bmaier/CASE/VAE_models/$run$',
	'analysis_base_dir_bin_count' : '/data/t3home000/bmaier/CASE/VAE_results/bump_hunt_results/$run$/bin_count/$strategy$/$quantile$',
	'model_analysis_base_dir': '/data/t3home000/bmaier/CASE/VAE_results/model_analysis/$run$',
	'model_comparison_dir': '/data/t3home000/bmaier/CASE/VAE_results/model_analysis/$run1$_vs_$run2$',
	
    # bump hunt results pattern: run_x/fig/loss_u

    'analysis_base_dir_fig' : '/data/t3home000/bmaier/CASE/VAE_results/bump_hunt_results/$run$/fig/loss_$loss_strat$',

    # QR directory pattern: run_x/sig_y/xsec_z/loss_u

    'model_dir_qr': '/data/t3home000/bmaier/CASE/QR_models/run_$run$',
    'analysis_base_dir_qr': '/data/t3home000/bmaier/CASE/QR_results/analysis/run_$run$/sig_$sig_name$/xsec_$sig_xsec$/loss_$loss_strat$',
}

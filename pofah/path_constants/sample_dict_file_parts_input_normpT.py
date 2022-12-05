import copy
import pofah.path_constants.sample_dict as sd

path_dict = copy.deepcopy(sd.path_dict)

update_dict = {
	#'base_dir' : '/data/t3home000/bmaier/CASE/',
	'base_dir' : '/local/bmaier/case/',

	'sample_dir' : {
		'qcdSig': 'BB_UL_MC_small_sig_v2_normpT',
		'qcdSide': 'BB_UL_MC_small_side_v2_normpT/train',
		#'qcdSide': 'BB_UL_MC_small_side_v2/train_small',
		'qcdSideExt': 'BB_UL_MC_small_side_v2_normpT/test',
		#'qcdSideExt': 'BB_UL_MC_small_side_v2/test_small',
                'gravitonSig': 'graviton_v2_normpT',
                'wkkSig': 'wkk_v2',
                'wpSig': 'wp_v2',
                'bstarSig': 'bstar_v2',
                #'qcdSigReco': 'BB_UL_MC_small_sig_with_loss'            
	},
}

path_dict.update(update_dict)

import copy
import pofah.path_constants.sample_dict as sd

path_dict = copy.deepcopy(sd.path_dict)

update_dict = {

	'base_dir' : '/eos/user/k/kiwoznia/data/VAE_data/baby_events',

	'sample_dir' : {
				'qcdSideAll': 'qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND',
				'qcdSig': 'qcd_sqrtshatTeV_13TeV_PU40',
                'GtoWW35na': 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV',
	},

        'file_names' : {},
}

path_dict.update(update_dict)

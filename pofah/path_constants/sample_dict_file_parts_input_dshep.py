import copy
import pofah.path_constants.sample_dict as sd

path_dict = copy.deepcopy(sd.path_dict)

update_dict = {
	'base_dir' : '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/',

	'sample_dir' : {
                'qcdSide': 'qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND',
                'qcdSideBis' : 'qcd_sqrtshatTeV_13TeV_PU40_BIS_SIDEBAND',
		'qcdSig': 'qcd_sqrtshatTeV_13TeV_PU40',
                'qcdSigBis': 'qcd_sqrtshatTeV_13TeV_PU40_BIS',
                'qcdNew': 'qcd_sqrtshatTeV_13TeV_PU40_NEW',
                'GtoWW15na': 'RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV',
                'GtoWW15br': 'RSGraviton_WW_BROAD_13TeV_PU40_1.5TeV',
                'GtoWW25na': 'RSGraviton_WW_NARROW_13TeV_PU40_2.5TeV',
                'GtoWW25br': 'RSGraviton_WW_BROAD_13TeV_PU40_2.5TeV',
                'GtoWW35na': 'RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV',
                'GtoWW35br': 'RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV',
                'GtoWW45na': 'RSGraviton_WW_NARROW_13TeV_PU40_4.5TeV',
                'GtoWW45br': 'RSGraviton_WW_BROAD_13TeV_PU40_4.5TeV',
	},
}

path_dict.update(update_dict)

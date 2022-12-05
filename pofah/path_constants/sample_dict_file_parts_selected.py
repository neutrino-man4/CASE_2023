import copy
import pofah.path_constants.sample_dict as sd

path_dict = copy.deepcopy(sd.path_dict)

update_dict = {
	
        'base_dir' : '/data/t3home000/bmaier/CASE/QR_results/events/run_$run$/sig_$sig_name$/xsec_$sig_xsec$/loss_$loss_strat$',

        # no sample directory, as all events of a data sample merged into single file
        'sample_dir' : {
                'qcdSideReco': '',
                'qcdSideExtReco' : '',
                'qcdSigReco': '',
                'qcdSigExtReco': '',
                'wkkSigReco' : '',
                'GtoWW15naReco': '',
                'GtoWW15brReco': '',
                'GtoWW25naReco': '',
                'GtoWW25brReco': '',
                'GtoWW35naReco': '',
                'GtoWW35brReco': '',
                'GtoWW45naReco': '',
                'GtoWW45brReco': '',
                'WkkToWRadionToWWW_M3000_Mr170Reco': '',
                'XToYYprimeTo4Q_MX3000_MY400_MYprime170_narrowReco': '',
                # prepared train and test split data for QR training
                'qcdSigAllTrainReco': '',
                'qcdSigAllTestReco': '',
                'qcdSigQRTrainReco': '',
                'qcdSigQRTestReco': 'lalal',
                'qcdSigMCOrigReco': '',
    },

}

path_dict.update(update_dict)

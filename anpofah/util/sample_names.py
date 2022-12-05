#BG_SR_sample = 'qcdSigTestReco'
BG_SR_sample = 'qcdSigMCOrigReco'
BG_SROrig_sample = 'qcdSigMCOrigReco'
SIG_samples_br = []
SIG_samples_na = []


mass_centers = [2000,2000,2000,2000,3000,3000,3000,3000,2000,3000,2000,2000,3000,3000,2000,2000,2000,2000,3000,3000,3000,3000]

SIG_samples = [
            'QstarToQW_M_2000_mW_170Reco',
            'QstarToQW_M_2000_mW_25Reco',
                'QstarToQW_M_2000_mW_400Reco',
                'QstarToQW_M_2000_mW_80Reco',
                'QstarToQW_M_3000_mW_170Reco',
                'QstarToQW_M_3000_mW_25Reco',
                'QstarToQW_M_3000_mW_400Reco',
                'QstarToQW_M_3000_mW_80Reco',
#                'QstarToQW_M_5000_mW_170Reco',
#                'QstarToQW_M_5000_mW_25Reco',
#                'QstarToQW_M_5000_mW_400Reco',
#                'QstarToQW_M_5000_mW_80Reco',
#            'RSGravitonToGluonGluon_kMpl01_M_1000Reco',
                'RSGravitonToGluonGluon_kMpl01_M_2000Reco',
                'RSGravitonToGluonGluon_kMpl01_M_3000Reco',
#                'RSGravitonToGluonGluon_kMpl01_M_5000Reco',
            'WkkToWRadionToWWW_M2000_Mr170Reco',
                'WkkToWRadionToWWW_M2000_Mr400Reco',
                'WkkToWRadionToWWW_M3000_Mr170Reco',
                'WkkToWRadionToWWW_M3000_Mr400Reco',
#                'WkkToWRadionToWWW_M5000_Mr170Reco',
#                'WkkToWRadionToWWW_M5000_Mr400Reco',
                'WpToBpT_Wp2000_Bp170_Top170_ZbtReco',
                'WpToBpT_Wp2000_Bp25_Top170_ZbtReco',
                'WpToBpT_Wp2000_Bp400_Top170_ZbtReco',
                'WpToBpT_Wp2000_Bp80_Top170_ZbtReco',
                'WpToBpT_Wp3000_Bp170_Top170_ZbtReco',
                'WpToBpT_Wp3000_Bp25_Top170_ZbtReco',
                'WpToBpT_Wp3000_Bp400_Top170_ZbtReco',
                'WpToBpT_Wp3000_Bp80_Top170_ZbtReco'
                 'XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrowReco',
#                'WpToBpT_Wp5000_Bp170_Top170_ZbtReco',
#                'WpToBpT_Wp5000_Bp25_Top170_ZbtReco',
#                'WpToBpT_Wp5000_Bp400_Top170_ZbtReco',
#                'WpToBpT_Wp5000_Bp80_Top170_ZbtReco'

]


SIG_QStar_samples = [
            'QstarToQW_M_2000_mW_170Reco',
            'QstarToQW_M_2000_mW_25Reco',
                'QstarToQW_M_2000_mW_400Reco',
                'QstarToQW_M_2000_mW_80Reco',
                'QstarToQW_M_3000_mW_170Reco',
                'QstarToQW_M_3000_mW_25Reco',
                'QstarToQW_M_3000_mW_400Reco',
                'QstarToQW_M_3000_mW_80Reco'
]

SIG_Graviton_samples = [
                'RSGravitonToGluonGluon_kMpl01_M_2000Reco',
                'RSGravitonToGluonGluon_kMpl01_M_3000Reco'
]


SIG_Wkk_samples = [
            'WkkToWRadionToWWW_M2000_Mr170Reco',
                'WkkToWRadionToWWW_M2000_Mr400Reco',
                'WkkToWRadionToWWW_M3000_Mr170Reco',
                'WkkToWRadionToWWW_M3000_Mr400Reco'
]

SIG_WpToBpT_samples = [
                'WpToBpT_Wp2000_Bp170_Top170_ZbtReco',
                'WpToBpT_Wp2000_Bp25_Top170_ZbtReco',
                'WpToBpT_Wp2000_Bp400_Top170_ZbtReco',
                'WpToBpT_Wp2000_Bp80_Top170_ZbtReco',
                'WpToBpT_Wp3000_Bp170_Top170_ZbtReco',
                'WpToBpT_Wp3000_Bp25_Top170_ZbtReco',
                'WpToBpT_Wp3000_Bp400_Top170_ZbtReco',
                'WpToBpT_Wp3000_Bp80_Top170_ZbtReco'
]

SIG_XToYY_samples = [
                'XToYYprimeTo4Q_MX3000_MY80_MYprime170_narrowReco'
]

all_samples = [BG_SR_sample] + [BG_SROrig_sample] + SIG_QStar_samples + SIG_Graviton_samples + SIG_Wkk_samples + SIG_WpToBpT_samples + SIG_XToYY_samples
#all_samples = [BG_SROrig_sample] + SIG_QStar_samples + SIG_Graviton_samples + SIG_Wkk_samples + SIG_WpToBpT_samples

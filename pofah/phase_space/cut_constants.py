
global_cuts = {
    
#    'mJJ' : 1200.,
#    'j1Eta': 2.4,
#    'j2Eta': 2.4,

}

sideband_cuts = {
    
    'sideband': 1.4,
    # 'jXPt': 200, # require *either* of the jets to have pt > 200

}

sideband_cuts.update(**global_cuts)

signalregion_cuts = {

    
#    'signalregion': 1.4,
#    'j1Pt': 200,
#    'j2Pt': 200,
}

signalregion_cuts.update(**global_cuts)

signalregion_Uppercuts = {

     'signalregionUpper' : 3800,
#    'signalregion': 1.4,
#    'j1Pt': 200,
#    'j2Pt': 200,
}

signalregion_Uppercuts.update(**global_cuts)

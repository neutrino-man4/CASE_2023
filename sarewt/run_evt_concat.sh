IN="/eos/project/d/dshep/TOPCLASS/DijetAnomaly/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT"
OUT="/eos/user/k/kiwoznia/data/VAE_data/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion.h5"
mkdir -p `dirname "${OUT}"`
python3 -u event_concatenate_serialization.py -in "${IN}" -out "${OUT}" -mb 2000 --signal


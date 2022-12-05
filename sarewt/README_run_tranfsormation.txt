export IN=/eos/user/k/kiwoznia/data/VAE_data/concat_events
export OUT=/eos/user/k/kiwoznia/data/VAE_data/march_2020_data/input/images/54px
python3 -u event_to_image_serialization.py -in /eos/user/k/kiwoznia/data/VAE_data/concat_events/qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND_concat_1.5M.h5 -out /eos/user/k/kiwoznia/data/VAE_data/march_2020_data/input/images/54px/qcd_sqrtshatTeV_13TeV_PU40_SIDEBAND_mjj_cut_1.2M_pt_img_54px.h5 -n 1200000 -bin 54

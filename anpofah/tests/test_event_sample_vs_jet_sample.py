import pofah.jet_sample as js
import pofah.util.event_sample as es
import pofah.path_constants.sample_dict_file_parts_input as sdfi
import pofah.util.sample_factory as sf
import anpofah.util.plotting_util as pu
import pofah.phase_space.cut_constants as cuts


paths = sf.SamplePathDirFactory(sdfi.path_dict)

# qcd_side = es.EventSample.from_input_dir('qcdSide', paths.sample_dir_path('qcdSide'), read_n=int(1e6))
# qcd_sig = es.EventSample.from_input_dir('qcdSig', paths.sample_dir_path('qcdSig'), read_n=int(1e6))

fig_dir = 'fig/merged_data_for_VAE'

# pu.plot_hist(qcd_side['j1Eta'], xlabel='j1Eta', title='qcd side no cuts j1Eta', plot_name='j1Eta_qcd_side_no_cuts', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
# pu.plot_hist(qcd_side['DeltaEtaJJ'], xlabel='DeltaEtaJJ', title='qcd side no cuts DeltaEtaJJ', plot_name='DeltaEtaJJ_qcd_side_no_cuts', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
# pu.plot_hist((qcd_side['DeltaEtaJJ']+qcd_side['j1Eta']), xlabel='j2Eta', title='qcd side no cuts j2Eta', plot_name='j2Eta_qcd_side_no_cuts', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)

qcd_side_cuts_event = es.EventSample.from_input_dir('qcdSide', paths.sample_dir_path('qcdSide'), read_n=int(1e5), **cuts.sideband_cuts)
qcd_side_cuts_jet = js.JetSample.from_input_dir('qcdSide', paths.sample_dir_path('qcdSide'), read_n=int(1e5), **cuts.sideband_cuts)
qcd_sig_cuts_event = es.EventSample.from_input_dir('qcdSig', paths.sample_dir_path('qcdSig'), read_n=int(1e5), **cuts.signalregion_cuts)
qcd_sig_cuts_jet = js.JetSample.from_input_dir('qcdSig', paths.sample_dir_path('qcdSig'), read_n=int(1e5), **cuts.signalregion_cuts)


# pu.plot_hist(qcd_side_cuts['j1Eta'], xlabel='j1Eta', title='qcd side cuts j1Eta', plot_name='j1Eta_qcd_side_cuts', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
# pu.plot_hist(qcd_side_cuts['DeltaEtaJJ'], xlabel='DeltaEtaJJ', title='qcd side no DeltaEtaJJ', plot_name='DeltaEtaJJ_qcd_side_cuts', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
# pu.plot_hist((qcd_side_cuts['DeltaEtaJJ']+qcd_side_cuts['j1Eta']), xlabel='j2Eta', title='qcd side cuts j2Eta', plot_name='j2Eta_qcd_side_cuts', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)

map_fun = lambda ff : ff['DeltaEtaJJ'] + ff['j1Eta'] # compute j2Eta
pu.plot_hist(map_fun(qcd_side_cuts_event), xlabel='j2Eta', title='qcd side cuts j2Eta', plot_name='j2Eta_qcd_side_cuts_mapfun_event', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(map_fun(qcd_side_cuts_jet), xlabel='j2Eta', title='qcd side cuts j2Eta', plot_name='j2Eta_qcd_side_cuts_mapfun_jet', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_side_cuts_event['j1Eta'], xlabel='j1Eta', title='qcd side cuts j1Eta', plot_name='j1Eta_qcd_side_cuts_event', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_side_cuts_jet['j1Eta'], xlabel='j1Eta', title='qcd side cuts j1Eta', plot_name='j1Eta_qcd_side_cuts_jet', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_side_cuts_event['mJJ'], xlabel='mJJ', title='qcd side cuts mJJ', plot_name='mJJ_qcd_side_cuts_event', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_side_cuts_jet['mJJ'], xlabel='mJJ', title='qcd side cuts mJJ', plot_name='mJJ_qcd_side_cuts_jet', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_side_cuts_event['DeltaEtaJJ'], xlabel='DeltaEtaJJ', title='qcd side cuts DeltaEtaJJ', plot_name='DeltaEtaJJ_qcd_side_cuts_event', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_side_cuts_jet['DeltaEtaJJ'], xlabel='DeltaEtaJJ', title='qcd side cuts DeltaEtaJJ', plot_name='DeltaEtaJJ_qcd_side_cuts_jet', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)

pu.plot_hist(map_fun(qcd_sig_cuts_event), xlabel='j2Eta', title='qcd sig cuts j2Eta', plot_name='j2Eta_qcd_sig_cuts_mapfun_event', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(map_fun(qcd_sig_cuts_jet), xlabel='j2Eta', title='qcd sig cuts j2Eta', plot_name='j2Eta_qcd_sig_cuts_mapfun_jet', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_sig_cuts_event['j1Eta'], xlabel='j1Eta', title='qcd sig cuts j1Eta', plot_name='j1Eta_qcd_sig_cuts_event', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_sig_cuts_jet['j1Eta'], xlabel='j1Eta', title='qcd sig cuts j1Eta', plot_name='j1Eta_qcd_sig_cuts_jet', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_sig_cuts_event['mJJ'], xlabel='mJJ', title='qcd sig cuts mJJ', plot_name='mJJ_qcd_sig_cuts_event', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_sig_cuts_jet['mJJ'], xlabel='mJJ', title='qcd sig cuts mJJ', plot_name='mJJ_qcd_sig_cuts_jet', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_sig_cuts_event['DeltaEtaJJ'], xlabel='DeltaEtaJJ', title='qcd sig cuts DeltaEtaJJ', plot_name='DeltaEtaJJ_qcd_sig_cuts_event', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)
pu.plot_hist(qcd_sig_cuts_jet['DeltaEtaJJ'], xlabel='DeltaEtaJJ', title='qcd sig cuts DeltaEtaJJ', plot_name='DeltaEtaJJ_qcd_sig_cuts_jet', fig_dir=fig_dir, ylogscale=True, normed=True, clip_outlier=False)

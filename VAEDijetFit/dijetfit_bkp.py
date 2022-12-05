import h5py, math, commands, random
from array import array
import numpy as np
import time, sys, os, optparse, json
import pathlib2

from Utils import f_test, calculateChi2, PlotFitResults, checkSBFit

import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
import CMS_lumi, tdrstyle
tdrstyle.setTDRStyle()
ROOT.gROOT.SetBatch(True)
ROOT.RooRandom.randomGenerator().SetSeed(random.randint(0, 1e+6))

from Fitter import Fitter
from DataCardMaker import DataCardMaker
from Utils import *
 
def get_generated_events(filename):

   with open('files_count.json') as f:
      data = json.load(f)
 
   N = 0
   found = False
   for k in data.keys():
      if k in filename or k.replace('_EXT','') in filename: 
         N += data[k][0]
         found = True
   
   if not found: 
      print ( "######### no matching key in files_count.json for file "+filename+", EXIT !!!!!!!!!!!!!!")
      sys.exit()

   print ( " in get_generated_events N = ",N) 
   return N 
 
def makeData(options, dataFile, q, iq, quantiles, hdata, minMJJ=0, maxMJJ=1e+04):
 
   file = h5py.File(options.inputDir+"/"+dataFile,'r')
   sel_key_q = 'sel_q90' if q == 'q100' else 'sel_' + q # selection column for quantile q (use rejected events of q90 for q100)
   print "Current quantile file: %s, reading quantile %s" % (file, sel_key_q)

   data = file['eventFeatures'][()] 
   mjj_idx = np.where(file['eventFeatureNames'][()] == 'mJJ')[0][0]

   # if quantile = total, fill histogram with all data and return
   if q=='total':
    for e in range(data.shape[0]): hdata.Fill(data[e][mjj_idx])
    return

   # else, if quantile = real quantile, fill with orthogonal data
   sel_idx = np.where(file['eventFeatureNames'][()] == sel_key_q)[0][0] # 0=rejected 1=accepted
 
   if q=='q01':
    for e in range(data.shape[0]):
     #if data[e][mjj_idx] < minMJJ or data[e][mjj_idx] > maxMJJ: continue
     if data[e][sel_idx]==1: hdata.Fill(data[e][mjj_idx])
   elif q=='q100': #if 90% quantile is rejected then events are in the 100-90% slice
    for e in range(data.shape[0]):
     #if data[e][mjj_idx] < minMJJ or data[e][mjj_idx] > maxMJJ: continue
     if data[e][sel_idx]==0: hdata.Fill(data[e][mjj_idx]) 
   else: 
    print ".... checking orthogonality wrt",quantiles[iq-1],"quantile...."
    sel_key_iq = 'sel_' + quantiles[iq-1] # selection column for quantile q
    sel_idx_iq = np.where(file['eventFeatureNames'][()] == sel_key_iq)[0] # 0=rejected 1=accepted
    for e in range(data.shape[0]):
     #if data[e][mjj_idx] < minMJJ or data[e][mjj_idx] > maxMJJ: continue
     if data[e][sel_idx_iq]==0 and data[e][sel_idx]==1: hdata.Fill(data[e][mjj_idx])

def checkSBFit(filename,quantile,roobins,plotname, out_dir):
 
   fin = ROOT.TFile.Open(filename,'READ')
   workspace = fin.w
   
   model = workspace.pdf('model_s')
   model.Print("v")
   var = workspace.var('mjj')
   data = workspace.data('data_obs')
   
   fres = model.fitTo(data,ROOT.RooFit.SumW2Error(0),ROOT.RooFit.Minos(0),ROOT.RooFit.Verbose(0),ROOT.RooFit.Save(1),ROOT.RooFit.NumCPU(8)) 
   fres.Print()
   
   frame = var.frame()
   data.plotOn(frame,ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson), ROOT.RooFit.Binning(roobins),ROOT.RooFit.Name("data_obs"),ROOT.RooFit.Invisible())
   model.getPdf('JJ_%s'%quantile).plotOn(frame,ROOT.RooFit.VisualizeError(fres,1),ROOT.RooFit.FillColor(ROOT.kRed-7),ROOT.RooFit.LineColor(ROOT.kRed-7),ROOT.RooFit.Name(fres.GetName()), ROOT.RooFit.Binning(roobins))
   model.getPdf('JJ_%s'%quantile).plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed+1),ROOT.RooFit.Name("model_s"))

   frame3 = var.frame()
   hpull = frame.pullHist("data_obs","model_s",True)
   hpull2 = ROOT.TH1F("hpull2","hpull2",len(binsx)-1, binsx[0], binsx[-1])
   for p in range(hpull.GetN()):
      x = ROOT.Double(0.)
      y = ROOT.Double(0.)
      hpull.GetPoint(p,x,y)
      bin = hpull2.GetXaxis().FindBin(x)
      hpull2.SetBinContent(p+1,y)

   frame3.addPlotable(hpull,"X0 P E1")
   chi2 = frame.chiSquare()
   print chi2
   ndof = 1
   
   data.plotOn(frame,ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson), ROOT.RooFit.Binning(roobins),ROOT.RooFit.Name("data_obs"),ROOT.RooFit.XErrorSize(0))
   dhist = ROOT.RooHist(frame.findObject('data_obs', ROOT.RooHist.Class()))
   chi2, ndof = calculateChi2(hpull, nPars + 1, excludeZeros = True, dataHist = dhist)
   #chi2,ndof = calculateChi2(histos_qcd[index],nPars[index],hpull)
   
   PlotFitResults(frame,fres.GetName(),nPars,frame3,"data_obs","model_s",chi2,ndof, 'sbFit_'+plotname, out_dir)


def prepare_output_directory(out_dir, clean_up=True):
    if not os.path.exists(out_dir):
        pathlib2.Path(out_dir).mkdir(parents=True, exist_ok=True)
        return
    if clean_up:
        os.system('rm '+out_dir+'/{*.root,*.txt,*.C}') 

    
if __name__ == "__main__":

   #python dijetfit.py -i inputdir --sig RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_parts/RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_reco.h5 --qcd qcd_sqrtshatTeV_13TeV_PU40_ALL_parts/qcd_sqrtshatTeV_13TeV_PU40_ALL_reco.h5
   #python dijetfit.py -i inputdir --sig RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_parts/RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_reco.h5 --qcd qcd_sqrtshatTeV_13TeV_PU40_ALL_parts/qcd_sqrtshatTeV_13TeV_PU40_ALL_reco.h5 --xsec 0.0 -M 1500.0

   #some configuration
   parser = optparse.OptionParser()
   parser.add_option("--xsec","--xsec",dest="xsec",type=float,default=0.0006,help="Injected signal cross section (suggested range 0-0.03)")
   parser.add_option("-M","-M",dest="mass",type=float,default=3500.,help="Injected signal mass")
   parser.add_option("-i","--inputDir",dest="inputDir",default='./',help="directory with all quantiles h5 files")
   parser.add_option("--qcd","--qcd",dest="qcdFile",default='qcd.h5',help="QCD h5 file")
   parser.add_option("--sig","--sig",dest="sigFile",default='signal.h5',help="Signal h5 file")
   parser.add_option("-l","--load_data",dest="load_data",action="store_true",help="Load orthogonal data")
   parser.add_option("--res", "--res", dest="sig_res", type="choice", choices=("na", "br"), default="na", help="resonance type: narrow [na] or broad [br]")
   parser.add_option("--out", '--out', dest="out_dir", type=str, default="./", help='output directory to store all results (plots, datacards, root files, etc.)')
   parser.add_option("-b", "--blinded", dest="blinded", action="store_true",
                      default=False,
                      help="Blinding the signal region for the fit.")
   (options,args) = parser.parse_args()
    
   xsec = options.xsec
   mass = options.mass
   sig_res = options.sig_res
   out_dir = options.out_dir
   prepare_output_directory(out_dir)
   binsx = [1126,1181,1246,1313,1383,1455,1530,1607,1687,1770,1856,1945,2037,2132,2231,2332,2438,2546,2659,2775,2895,3019,3147,3279,3416,3558,3704,3854,4010,4171,4337,4509,4686,4869,5058,5253,5500,5663,5877,6099,6328,6564,6808]
   shift = 1200 - binsx[0] # shift to mjj cut

   binsx = [1455,1530,1607,1687,1770,1856,1945,2037,2132,2231,2332,2438,2546,2659,2775,2895,3019,3147,3279,3416,3558,3704,3854,4010,4171,4337,4509,4686,4869,5058,5253,5500,5663,5877,6099,6328,6564,6808]
   shift = 0


   binsx = [e+shift for e in binsx]
   roobins = ROOT.RooBinning(len(binsx)-1, array('d',binsx), "mjjbins")
   bins_fine = int(binsx[-1]-binsx[0])
   # quantiles = ['q1','q5','q10','q30','q50','q70','q90','q100','total']
   quantiles = ['q01', 'q10', 'q30', 'q50', 'q70', 'q90','q100','total']
   nPars = 2 # DO THESE NEED TO BE DIFFERENT DEPENDING ON QUANTILE???

   # change signal fit intervall according to resonance breadth
   if sig_res == "na":
      sig_mjj_limits = (0.8*mass,1.2*mass)
   else:
      sig_mjj_limits = (0.4*mass,1.4*mass)

   bins_sig_fit = array('f',truncate([binsx[0]+ib for ib in range(bins_fine+1)],*sig_mjj_limits))
   large_bins_sig_fit = array('f',truncate(binsx,*sig_mjj_limits))
   roobins_sig_fit = ROOT.RooBinning(len(large_bins_sig_fit)-1, array('d',large_bins_sig_fit), "mjjbins_sig")
   print " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  "
   qr_test_share = 0.8
   print " qr_test_share ",qr_test_share
   qcd_xsec = 8.73e6 # [fb]
   print "get_generated_events(options.qcdFile) ",get_generated_events(options.qcdFile)
   qcd_gen_events = qr_test_share*get_generated_events(options.qcdFile) # 80% of SR qcd events => why applying dEta (SR) cut here but not all the other cuts??
   print "qcd_gen_events ",qcd_gen_events
   sig_xsec = 1000. # metric [fb] = 1 [pb]???
   print " sig_xsec ",sig_xsec," qcd_xsec ",qcd_xsec
   print " get_generated_events(options.sigFile) ",get_generated_events(options.sigFile)
   sig_gen_events = qr_test_share*get_generated_events(options.sigFile) # 80% of signal events (but this corresponds to an enormous xsec?!)
   print " sig_gen_events ",sig_gen_events
   lumi = qcd_gen_events/qcd_xsec # qcd SR lumi 51
   print " lumi ",lumi
   print " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  "
   ################################### FIRST PREPARE DATA ###################################
   '''
    MAKE HISTOGRAMS:
    for each quantile + 'bottom rejected' 10% + all events:
      fill mjj histogram for 
      - signal: histos_sig
      - background: histos_qcd
      and save in data_mjj_X_qY.root
   '''       
   histos_sig = []
   histos_qcd = []

   if not options.load_data:
      
      #Signal data preparation 
      for iq,q in enumerate(quantiles):
         
         #histos_sig.append( ROOT.TH1F("mjj_sig_%s"%q,"mjj_sig_%s"%q,bins_fine,binsx[0],binsx[-1]) )
         histos_sig.append( ROOT.TH1F("mjj_sig_%s"%q,"mjj_sig_%s"%q,len(bins_sig_fit)-1,bins_sig_fit) )
         print
         makeData(options,options.sigFile,q,iq,quantiles,histos_sig[-1],*sig_mjj_limits) #first fill orthogonal data histos
         print "************ Found",histos_sig[-1].GetEntries(),"signal events for quantile",q
         print

      #Background data preparation
      for iq,q in enumerate(quantiles):
      
         histos_qcd.append( ROOT.TH1F("mjj_qcd_%s"%q,"mjj_qcd_%s"%q,bins_fine,binsx[0],binsx[-1]) )
         print
         makeData(options,options.qcdFile,q,iq,quantiles,histos_qcd[-1]) #first fill orthogonal data histos
         print "************ Found",histos_qcd[-1].GetEntries(),"background events for quantile",q
         print

      for h in histos_sig: h.SaveAs(os.path.join(out_dir, "data_"+h.GetName()+".root"))
      for h in histos_qcd: h.SaveAs(os.path.join(out_dir, "data_"+h.GetName()+".root"))
      
   else: #let's make it faster if you have run once already!
      print('=== loading histogram data from file ===')
      #Load signal data
      for q in quantiles:
      
         fname = os.path.join(out_dir, "data_mjj_sig_%s.root"%q)
         q_datafile = ROOT.TFile.Open(fname,'READ')
         histos_sig.append(q_datafile.Get("mjj_sig_%s"%q))
         histos_sig[-1].SetDirectory(ROOT.gROOT)
         q_datafile.Close()

      #Load background data
      for q in quantiles:
         
         fname = os.path.join(out_dir, "data_mjj_qcd_%s.root"%q)
         q_datafile = ROOT.TFile.Open(fname,'READ')
         histos_qcd.append(q_datafile.Get("mjj_qcd_%s"%q))
         histos_qcd[-1].SetDirectory(ROOT.gROOT)
         q_datafile.Close()
        
      for q,h in enumerate(histos_sig): print "************ Found",h.GetEntries(),"signal events for quantile",quantiles[q]
      for q,h in enumerate(histos_qcd): print "************ Found",h.GetEntries(),"background events for quantile",quantiles[q]

   sum_n_histos_qcd = sum([h.GetEntries() for h in histos_qcd[:-1]])
   sum_n_histos_sig = sum([h.GetEntries() for h in histos_sig[:-1]])

   print "************************************************************************************** "
   print "TOTAL SIGNAL EVENTS",histos_sig[-1].GetEntries(), " (sum histos = )", sum_n_histos_sig
   print "TOTAL BACKGROUND EVENTS",histos_qcd[-1].GetEntries(), " (sum histos = )", sum_n_histos_qcd
   print
   print "************************************************************************************** "
   ################################### NOW MAKE THE FITS ###################################
   '''
    for each quantile:
      - fit signal shape -> gauss mu & std ??
      - fit background shape -> exponential lambda ??
      - chi2 ??
   ''' 
   nParsToTry = [2, 3, 4]
   best_i = [0]*len(quantiles)
   nPars_QCD = [0]*len(quantiles)
   qcd_fname = [0]*len(quantiles)
   chi2s = [[0]*len(nParsToTry)]*len(quantiles)
   ndofs = [[0]*len(nParsToTry)]*len(quantiles)
   probs = [[0]*len(nParsToTry)]*len(quantiles)
   qcd_fnames = [[""]*len(nParsToTry)]*len(quantiles)

   for iq,q in enumerate(quantiles):
      
      print "########## FIT SIGNAL AND SAVE PARAMETERS for quantile "+q+"    ############"
      sig_outfile = ROOT.TFile(os.path.join(out_dir, "sig_fit_%s.root"%q),"RECREATE")

      ### create signal model: gaussian centered at mass-center with sigma in {2%,10%of mass center} + crystal ball for asymmetric tail (pure functional form)

      # setup fit parameters according to resonance type
      if sig_res == "na": # fit narrow signal
        print('===> fitting narrow signal model')
        alpha = (0.6, 0.45, 1.05)
      else: # else fit broad signal
        print('===> fitting broad signal model')
        alpha = None #use default alpha 
      
      fitter=Fitter(['mjj_fine'])
      fitter.signalResonance('model_s',"mjj_fine", mass=mass, alpha=alpha)
       
      ### fit the signal model to actual sig histogram data

      fitter.w.var("MH").setVal(mass)
      fitter.importBinnedData(histos_sig[iq],['mjj_fine'],'data')
      fres = fitter.fit('model_s','data',[ROOT.RooFit.Save(1)])
      fres.Print()

      ### compute chi-square of compatibility of signal-histogram and signal model for sanity ch05eck
      # plot fit result to signal_fit_q.png for each quantile q

      mjj_fine = fitter.getVar('mjj_fine')
      mjj_fine.setBins(len(bins_sig_fit))
      chi2_fine = fitter.projection("model_s","data","mjj_fine", os.path.join(out_dir, "signal_fit_%s.png"%q))
      fitter.projection("model_s","data","mjj_fine", os.path.join(out_dir, "signal_fit_%s_log.png"%q), 0, True)
      chi2 = fitter.projection("model_s","data","mjj_fine", os.path.join(out_dir, "signal_fit_%s_binned.png"%q), roobins_sig_fit)
      fitter.projection("model_s","data","mjj_fine", os.path.join(out_dir, "signal_fit_%s_log_binned.png"%q), roobins_sig_fit, True)

      # write signal histogram and model with params

      sig_outfile.cd()
      histos_sig[iq].Write()

      graphs={'mean':ROOT.TGraphErrors(),'sigma':ROOT.TGraphErrors(),'alpha':ROOT.TGraphErrors(),'sign':ROOT.TGraphErrors(),'scalesigma':ROOT.TGraphErrors(),'sigfrac':ROOT.TGraphErrors()}
      for var,graph in graphs.iteritems():
         value,error=fitter.fetch(var)
         graph.SetPoint(0,mass,value)
         graph.SetPointError(0,0.0,error)

      sig_outfile.cd()
      for name,graph in graphs.iteritems(): graph.Write(name)
      
      sig_outfile.Close() 

      print "#############################"
      print "for quantile ",q
      print "signal fit chi2 (fine binning)",chi2_fine
      print "signal fit chi2 (large binning)",chi2
      print "#############################"

      print
      print
      print "############# FIT BACKGROUND AND SAVE PARAMETERS for quantile "+q+"   ###########"

      sb1_edge = 2232
      sb2_edge = 2776

      regions = [("SB1", binsx[0], sb1_edge),
                 ("SB2", sb2_edge, binsx[-1]),
                 ("SR", sb1_edge, sb2_edge),
                 ("FULL", binsx[0], binsx[-1])]

      blind_range = ROOT.RooFit.Range("SB1,SB2")
      full_range = ROOT.RooFit.Range("FULL")
      fit_ranges = [(binsx[0], sb1_edge), (sb2_edge, binsx[-1])]

      histos_sb_blind = []
      h = apply_blinding(histos_qcd[iq], ranges=fit_ranges)
      histos_sb_blind.append(h )
      num_blind = histos_sb_blind[-1].Integral()

      if options.blinded:
         fitting_histogram = histos_sb_blind[-1]
         data_name = "data_qcd_blind"
         fit_range = blind_range
         chi2_range = fit_ranges
         norm = ROOT.RooFit.Normalization(num_blind, ROOT.RooAbsReal.NumEvent)

      else:
         fitting_histogram = histos_qcd[iq]
         data_name = "data_qcd"
         fit_range = full_range
         chi2_range = None
         norm = ROOT.RooFit.Normalization(histos_qcd[iq].Integral(),
                                        ROOT.RooAbsReal.NumEvent)

      for i, nPars in enumerate(nParsToTry):
         print("Trying %i parameter background fit" % nPars)
         qcd_fnames[i] = str(nPars) + 'par_qcd_fit%i_quantile%s.root' % (i,q)
         # print "          FIXMEEEEEE QCD NAMES NEED QUANTILEEEEEEE "
         # sys.exit()

         qcd_outfile = ROOT.TFile(os.path.join(out_dir, qcd_fnames[i]),'RECREATE')

         ### create background model: 2-parameter (p1 & p2) exponential (generic functional form, not based on data)

         fitter_QCD=Fitter(['mjj_fine'])
         fitter_QCD.qcdShape('model_b','mjj_fine',nPars)

         ### fit background model to actual qcd histogram data (all cuts applied)

         fitter_QCD.importBinnedData(histos_qcd[iq],['mjj_fine'],'data_qcd')
         fres = fitter_QCD.fit('model_b','data_qcd',[ROOT.RooFit.Save(1)])
         fres.Print()

         ### compute chi-square of compatibility of qcd-histogram and background model for sanity check
         # plot fit result to qcd_fit_q_binned.png for each quantile q

         chi2_fine = fitter_QCD.projection("model_b","data_qcd","mjj_fine", os.path.join(out_dir, qcd_fnames[i].replace(".root",".png")), 0, True) # chi2 => sanity check
         chi2_binned = fitter_QCD.projection("model_b","data_qcd","mjj_fine", os.path.join(out_dir, qcd_fnames[i].replace(".root","_binned.png")), roobins, True)

         ### write background histogram

         qcd_outfile.cd()
         histos_qcd[iq].Write() # ??? => write histo into qcd_outfile

         ### plot background data with fit

         mjj = fitter_QCD.getVar('mjj_fine')
         mjj.setBins(bins_fine)
         model = fitter_QCD.getFunc('model_b')
         dataset = fitter_QCD.getData('data_qcd')

         frame = mjj.frame()
         dataset.plotOn(frame,ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson), ROOT.RooFit.Name("data_qcd"),ROOT.RooFit.Invisible(),ROOT.RooFit.Binning(roobins))
         model.plotOn(frame,ROOT.RooFit.VisualizeError(fres,1),ROOT.RooFit.FillColor(ROOT.kRed-7),ROOT.RooFit.LineColor(ROOT.kRed-7),ROOT.RooFit.Name(fres.GetName()),ROOT.RooFit.Binning(roobins))
         model.plotOn(frame,ROOT.RooFit.LineColor(ROOT.kRed+1),ROOT.RooFit.Name("model_b"),ROOT.RooFit.Binning(roobins))

         framePulls = mjj.frame()
         hpull = frame.pullHist("data_qcd","model_b",True) # ??? pull => second canvas in pull
         #hpull2 = ROOT.TH1F("hpull2","hpull2",len(binsx)-1, binsx[0], binsx[-1])
         #for p in range(hpull.GetN()):
         # x = ROOT.Double(0.)
         # y = ROOT.Double(0.)
         # hpull.GetPoint(p,x,y)
         # #print p,x,y
         # bin = hpull2.GetXaxis().FindBin(x)
         # hpull2.SetBinContent(p+1,y)

         framePulls.addPlotable(hpull,"X0 P E1")
         chi2 = frame.chiSquare()
         ndof = 1
         print "chi2 frame:",frame.chiSquare()

         dataset.plotOn(frame,ROOT.RooFit.DataError(ROOT.RooAbsData.Poisson),ROOT.RooFit.Name("data_qcd"),ROOT.RooFit.XErrorSize(0),ROOT.RooFit.Binning(roobins))
         dhist = ROOT.RooHist(frame.findObject(data_name, ROOT.RooHist.Class()))
         my_chi2, my_ndof = calculateChi2(hpull, nPars, ranges=chi2_range, excludeZeros = True, dataHist = dhist)
         my_prob = ROOT.TMath.Prob(my_chi2, my_ndof)

         # plot full qcd fit results with pull factors to mjj_qcd_q.pdf for each quantile q

         PlotFitResults(frame,fres.GetName(),nPars,framePulls,"data_qcd",
                        "model_b",my_chi2, my_ndof,
                        histos_qcd[iq].GetName()+"{}".format(
                           "_blinded" if options.blinded else ""), out_dir)

         # write qcd model with params

         graphs = {}
         for p in range(nPars): graphs['p%i'%(p+1)] = ROOT.TGraphErrors()
         for var,graph in graphs.iteritems():
            print var
            value,error=fitter_QCD.fetch(var)
            graph.SetPoint(0,mass,value)
            graph.SetPointError(0,0.0,error)

         qcd_outfile.cd()       
         for name,graph in graphs.iteritems(): graph.Write(name) # ??? => saving params of bg fit -> load later for datacard

         qcd_outfile.Close()

         print "#############################"
         print " for quantile ",q
         print "bkg fit chi2 (fine binning)",chi2_fine
         print "bkg fit chi2 (large binning)",chi2_binned
         print "bkg fit chi2",chi2
         print "#############################"

         print("bkg fit chi2/nbins (fine binning) ", chi2_fine)
         print("My chi2, ndof, prob", my_chi2, my_ndof, my_prob)
         print("My chi/ndof, chi2/nbins", my_chi2/my_ndof,
              my_chi2/(my_ndof + nPars))
         print("#############################")

         chi2s[iq][i] = my_chi2
         ndofs[iq][i] = my_ndof
         probs[iq][i] = my_prob
         fitter_QCD.delete()

      best_i[iq] = f_test(nParsToTry, ndofs[iq], chi2s[iq])
      nPars_QCD[iq] = nParsToTry[best_i[iq]]
      qcd_fname[iq] = qcd_fnames[best_i[iq]]
      print ( " qcd_fname[iq] ",qcd_fname[iq])
      print("\n Chose %i parameters based on F-test ! \n" % nPars_QCD[iq])

      print
      print 
      print "############# GENERATE SIGNAL+BACKGROUND DATA FROM PDFs for quantile "+q+"       ###########"

      
      f = ROOT.TFile("/tmp/%s/cache%i.root"%(commands.getoutput("whoami"),random.randint(0, 1e+6)),"RECREATE")
      f.cd()
      w=ROOT.RooWorkspace("w","w")

      fitter_QCD=Fitter(['mjj_fine'])
      fitter_QCD.qcdShape('model_b','mjj_fine',nPars_QCD[iq])
      fitter_QCD.importBinnedData(histos_qcd[iq],['mjj_fine'],'data_qcd')
      fitter_QCD.fit('model_b','data_qcd',[ROOT.RooFit.Save(1)])

      model_b = fitter_QCD.getFunc('model_b')
      model_s = fitter.getFunc('model_s')
      
      model_b.Print("v")
      model_s.Print("v")

      print " ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
      print " quantile ",q
      print "Generate histos_qcd[iq].Integral() ",histos_qcd[iq].Integral(),"background events from model_b" # ??? taking qcd number events as they are => all cuts applied (!!!)
      mjj = fitter_QCD.getVar('mjj_fine')
      mjj.setBins(bins_fine)
      dataqcd = model_b.generateBinned(ROOT.RooArgSet(mjj),histos_qcd[iq].Integral())
      hdataqcd = dataqcd.createHistogram("mjj_fine")
      hdataqcd.SetName("mjj_generate_qcd_%s"%q)

      # signal xsec set to 0 by default, so hdatasig hist not filled !
      if xsec != 0:
         
         print "Generate histos_sig[iq].Integral() ",histos_sig[iq].Integral()," * xsec ",xsec," = ",int(histos_sig[iq].Integral()*xsec),"signal events from model_s"  # ??? taking sig number events scaled by xsec
         num_sig_evts = int(histos_sig[iq].Integral()*xsec)
         if num_sig_evts > 0:
            datasig = model_s.generateBinned(ROOT.RooArgSet(mjj),int(histos_sig[iq].Integral()*xsec))
            hdatasig = datasig.createHistogram("mjj_fine")
         else:
            hdatasig = ROOT.TH1F("mjj_generate_sig_%s"%q,"mjj_generate_sig_%s"%q,histos_qcd[iq].GetNbinsX(),histos_qcd[iq].GetXaxis().GetXmin(),histos_qcd[iq].GetXaxis().GetXmax())

      else: # just set same bins as qcd hist, do not fill!
         print (" xsec is zero!")
         hdatasig = ROOT.TH1F("mjj_generate_sig_%s"%q,"mjj_generate_sig_%s"%q,histos_qcd[iq].GetNbinsX(),histos_qcd[iq].GetXaxis().GetXmin(),histos_qcd[iq].GetXaxis().GetXmax())
      print " ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"  
      hdatasig.SetName("mjj_generate_sig_%s"%q)
      
      # signal+background fit (total histo) => since signal xsec = 0 per default, this is only background data & fit!

      sb_outfile = ROOT.TFile(os.path.join(out_dir, 'sb_fit_%s.root'%q),'RECREATE')
      sb_outfile.cd()
      htot = ROOT.TH1F()
      htot = hdataqcd.Clone("mjj_generate_tot_%s"%q)
      print "so far so good"
      htot.Add(hdatasig)
      print "Add works!"
      hdatasig.Write("mjj_generate_sig_%s"%q)
      hdataqcd.Write("mjj_generate_qcd_%s"%q)
      htot.Write("mjj_generate_tot_%s"%q)

      w.Delete()
      f.Close()
      f.Delete()
      sb_outfile.Close()
      fitter_QCD.delete()
      fitter.delete()

      print
      print
      print "############ MAKE PER CATEGORY (quantile ",q," ) DATACARD AND WORKSPACE AND RUN COMBINE #############"

      card=DataCardMaker(q, out_dir)

      card.addSignalShape('model_signal_mjj','mjj', os.path.join(out_dir, 'sig_fit_%s.root'%q), {'CMS_scale_j':1.0}, {'CMS_res_j':1.0})
      
      ### !!! compute amount of signal to be injected 
      
      # sig_xsec = 1000fb, lumi = qcd SR lumi, sig_gen_events ~ 80% of 900K signal => have to take signal SR share ???
      # constant = number of signal events expected with sig_xsec 1000 fb / number of signals generated = < 1 == scaling factor for histogram data
      print " sig_gen_events ",sig_gen_events
      constant = sig_xsec*lumi/sig_gen_events  
      if xsec==0: constant = 1.0*constant
      else: constant=xsec*constant
      print " constant = number of signal events expected with sig_xsec 1000 fb / number of signals generated = ",constant
      # add signal pdf from model_s, taking integral number of events with constant scaling factor for sig
      card.addFixedYieldFromFile('model_signal_mjj',0, os.path.join(out_dir, 'sig_fit_%s.root'%q), histos_sig[iq].GetName(), constant=constant)
      card.addSystematic("CMS_scale_j","param",[0.0,0.012])
      card.addSystematic("CMS_res_j","param",[0.0,0.08]) 

      # add bg pdf
      card.addQCDShapeNoTag('model_qcd_mjj','mjj', os.path.join(out_dir, qcd_fname[iq]), nPars_QCD[iq])
      card.addFloatingYield('model_qcd_mjj',1, os.path.join(out_dir, qcd_fname[iq]), histos_qcd[iq].GetName())
      for i in range(1,nPars+1): card.addSystematic("CMS_JJ_p%i"%i,"flatParam",[])
      card.addSystematic("model_qcd_mjj_JJ_%s_norm"%q,"flatParam",[]) # integral -> anzahl events -> fuer skalierung der genormten roofit histogramm

      card.importBinnedData(os.path.join(out_dir, 'sb_fit_%s.root'%q), 'mjj_generate_tot_%s'%q,["mjj"],'data_obs',1.0)
      card.makeCard()
      card.delete()


      # run combine on datacard -> create workspaces workspace_JJ_0.0_quantile.root
      # -M Significance: profile likelihood
      cmd = 'cd {out_dir} && ' \
            'text2workspace.py datacard_JJ_{label}.txt -o workspace_JJ_{xsec}_{label}.root && ' \
            'combine -M Significance workspace_JJ_{xsec}_{label}.root -m {mass} -n significance_{xsec}_{label} && ' \
            'combine -M Significance workspace_JJ_{xsec}_{label}.root -m {mass} --pvalue -n pvalue_{xsec}_{label}'.format(out_dir=out_dir, mass=mass, xsec=xsec, label=q)
      print cmd
      os.system(cmd)
      
      #run and visualize s+b fit as sanity check (sb_fit_mjj_qcd_q.root.pdf)
      checkSBFit('{out_dir}/workspace_JJ_{xsec}_{label}.root'.format(out_dir=out_dir, xsec=xsec,label=q),q,roobins,histos_qcd[iq].GetName()+".root", out_dir)

   print
   print
   print "############ MAKE N-CATEGORY DATACARD AND WORKSPACE AND RUN COMBINE #############"
   #The difference here is that the background shape comes from one specific quantile (rather than from its own as above)

   # import ipdb; ipdb.set_trace()
   
   cmdCombine = 'cd {out_dir} && combineCards.py '.format(out_dir=out_dir)

   for iq,q in enumerate(quantiles):  
   
      if q == 'total': continue

      card=DataCardMaker(q+"_4combo", out_dir)

      print "********************** Add signal shape to datacard **********************"
      card.addSignalShape('model_signal_mjj','mjj', os.path.join(out_dir, 'sig_fit_%s.root'%q), {'CMS_scale_j':1.0},{'CMS_res_j':1.0})
      # TODO: check data!
      constant = sig_xsec*lumi/sig_gen_events
      if xsec==0: constant = 1.0*constant
      else: constant=xsec*constant
      card.addFixedYieldFromFile('model_signal_mjj',0, os.path.join(out_dir, 'sig_fit_%s.root'%q), histos_sig[iq].GetName(), constant=constant)
      card.addSystematic("CMS_scale_j","param",[0.0,0.012])
      card.addSystematic("CMS_res_j","param",[0.0,0.08]) 

      #TAKE BACKGROUND SHAPE COMES FROM BACKGROUND-ENRICHED QUANTILE SLICE --> WHICH ONE? TRY THE Q100 SLICE!
      card.addQCDShapeNoTag('model_qcd_mjj','mjj', os.path.join(out_dir, qcd_fname[len(quantiles)-2]), nPars_QCD[len(quantiles)-2])
      card.addFloatingYield('model_qcd_mjj',1, os.path.join(out_dir, qcd_fname[iq]), histos_qcd[iq].GetName())
      for i in range(1,nPars_QCD[len(quantiles)-2]+1): card.addSystematic("CMS_JJ_p%i"%i,"flatParam",[])
      card.addSystematic("model_qcd_mjj_JJ_q100_4combo_norm","flatParam",[])

      card.importBinnedData(os.path.join(out_dir, 'sb_fit_%s.root'%q), 'mjj_generate_tot_%s'%q,["mjj"],'data_obs',1.0)
      card.makeCard()
      card.delete()
      
      cmdCombine += "quantile_{quantile}=datacard_JJ_{label}.txt ".format(quantile=q,label=q+"_4combo",xsec=xsec)
   
   #MAKE FINAL DATACARD (needs some cosmetics as below) 
   cmdCombine += '&> datacard_{xsec}_final.txt'.format(xsec=xsec)
   print cmdCombine
   os.system(cmdCombine)


   d = open(os.path.join(out_dir, 'datacard_tmp.txt'),'w')
   dorig = open('{out_dir}/datacard_{xsec}_final.txt'.format(out_dir=out_dir, xsec=xsec),'r')
   for l in dorig.readlines(): d.write(l)
   d.write('quantile_q100_rate     rateParam       quantile_q100  model_qcd_mjj   1\n')
   d.write('quantile_q90_rate      rateParam       quantile_q90  model_qcd_mjj   (0.20*@0)/0.10  quantile_q100_rate\n')
   d.write('quantile_q70_rate      rateParam       quantile_q70  model_qcd_mjj   (0.20*@0)/0.10  quantile_q100_rate\n')
   d.write('quantile_q50_rate      rateParam       quantile_q50  model_qcd_mjj   (0.20*@0)/0.10  quantile_q100_rate\n')
   d.write('quantile_q30_rate      rateParam       quantile_q30  model_qcd_mjj   (0.20*@0)/0.10  quantile_q100_rate\n') 
   d.write('quantile_q10_rate      rateParam       quantile_q10  model_qcd_mjj   (0.05*@0)/0.10  quantile_q100_rate\n')
   # d.write('quantile_q5_rate      rateParam       quantile_q5  model_qcd_mjj   (0.04*@0)/0.10  quantile_q100_rate\n')
   d.write('quantile_q01_rate      rateParam       quantile_q01  model_qcd_mjj   (0.01*@0)/0.10  quantile_q100_rate\n')
   d.close()

   cmd = 'cd {out_dir} && ' \
          'mv datacard_tmp.txt datacard_{xsec}_final.txt && ' \
          'text2workspace.py datacard_{xsec}_final.txt -o workspace_{xsec}_final.root && ' \
          'combine -M Significance workspace_{xsec}_final.root -m {mass} -n significance_{xsec} && '\
          'combine -M Significance workspace_{xsec}_final.root -m {mass} --pvalue -n pvalue_{xsec}'.format(out_dir=out_dir, mass=mass,xsec=xsec)
   print cmd
   os.system(cmd)
   print " DONE! "
   #checkSBFit('workspace_{xsec}_{label}.root'.format(xsec=xsec,label='final'),q+"_4combo",roobins,'final.root')

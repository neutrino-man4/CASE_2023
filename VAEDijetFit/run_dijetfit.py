import os,sys,time
import numpy as np
from array import array
import optparse, json

import ROOT
import CMS_lumi, tdrstyle

from Utils import *


def plotPValue(xsec_scan, quantiles, plot_name_suffix='', out_dir=''):

    xmin = xsec_scan[0]*1000.
    xmax = (xsec_scan[-1]+xsec_scan[-1]*0.1)*1000.
    
    canv = get_canvas("c_Significance")
    canv.cd()

    hrl_SM = canv.DrawFrame(xmin,1e-12,xmax, 1) 
    ytitle = "p-value"
    hrl_SM.GetYaxis().SetTitle(ytitle)
    hrl_SM.GetYaxis().SetTitleSize(0.045)
    hrl_SM.GetXaxis().SetTitleSize(0.045)
    hrl_SM.GetXaxis().SetLabelSize(0.03)
    hrl_SM.GetYaxis().SetLabelSize(0.03)
    hrl_SM.GetYaxis().SetTitleOffset(1.2)
    hrl_SM.GetXaxis().SetTitleOffset(1.1)
    hrl_SM.GetXaxis().SetTitle("cross-section [fb]")

    graphs = []
    palette = get_palette('gv')
    col = ROOT.TColor()
 
    for iq,q in enumerate(quantiles):
    
        x = array('d', xsec_scan*1000.)
        ys = array('d', [])
        yp = array('d',[])

        fin = open(os.path.join(out_dir,'results_%s.txt'%q), 'r')
        for l in fin.readlines():
            l = l.split('\t')
            yp.append(float(l[1]))
            ys.append(float(l[2]))
        fin.close()
    
        nPoints=len(x)
        gp = ROOT.TGraph(nPoints,x,yp)
        gp.SetName("PValue_%s"%q)
        if q!='total':
            gp.SetLineColor(col.GetColor(palette[iq]))
            gp.SetMarkerColor(col.GetColor(palette[iq]))
        else: 
            gp.SetLineColor(1)
            gp.SetMarkerColor(1)
        gp.SetMarkerStyle(20)
        gp.SetLineWidth(2)
        gp.SetMarkerSize(1.)
        graphs.append(gp)
        
    pvalues = [ ROOT.RooStats.SignificanceToPValue(i) for i in range(1,7) ]
    lines = [ ROOT.TLine(xmin,pvalues[i-1],xmax,pvalues[i-1]) for i in range(1,7) ]
    for l in lines:
        l.SetLineColor(ROOT.kGray+1)
        l.SetLineWidth(2)
        l.SetLineStyle(ROOT.kDashed)
     
    bans = [ ROOT.TLatex(xmax*0.93,pvalues[i-1],("%i #sigma"%(i))) for i in range(1,7) ]
    for b in bans:
        b.SetTextSize(0.028)
        b.SetTextColor(ROOT.kGray+1)

    legend = ROOT.TLegend(0.18,0.2183362,0.28,0.419833)
    legend.SetTextSize(0.032)
    legend.SetLineColor(0)
    legend.SetShadowColor(0)
    legend.SetLineStyle(1)
    legend.SetLineWidth(1)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetMargin(0.35)

    for iq,q in enumerate(quantiles):
        if q == 'total': q = 'bump hunt'
        if q == 'final' : q = 'AD bump hunt'
        legend.AddEntry(graphs[iq],q,'LP') 

    graphs[0].Draw('LP')
    for g in range(1,len(graphs)): graphs[g].Draw("LPsame")
    for l in lines:
        print(l)
        l.Draw("same")

    for b in bans: b.Draw()

    legend.Draw()
    canv.Update() 
    canv.cd()
    canv.Update()
    canv.RedrawAxis()
    canv.RedrawAxis("g")
    frame = canv.GetFrame()
    frame.Draw() 
    canv.cd()
    canv.Update()
 
    canv.SaveAs(os.path.join(out_dir, "pvalue"+plot_name_suffix+".png"))
    time.sleep(10)


if __name__ == "__main__":

    #python run_dijetfit.py --run --i inputdir -M 1500 --sig RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_parts/RSGraviton_WW_NARROW_13TeV_PU40_1.5TeV_reco.h5 --qcd qcd_sqrtshatTeV_13TeV_PU40_ALL_parts/qcd_sqrtshatTeV_13TeV_PU40_ALL_reco.h5

    parser = optparse.OptionParser()
    parser.add_option("--run","--run", dest="run", default=False, action="store_true", help="Run scan")
    parser.add_option("-n","-n",dest="run_n", type=int, default=0, help="Experiment number")
    parser.add_option("-M","-M", dest="mass", type=float, default=3500., help="Injected signal mass")
    parser.add_option("-i","--inputDir",dest="inputDir", default='./', help="directory with all quantiles h5 files")
    parser.add_option("--qcd","--qcd", dest="qcdFile", default='qcd.h5', help="QCD h5 file")
    parser.add_option("--sig","--sig", dest="sigFile", default='signal.h5', help="Signal h5 file")
    parser.add_option("-x", "--sigxsec", dest="sigXsec", default=10, help="true signal cross-section")
    parser.add_option("--res", "--res", dest="sigRes", type="choice", choices=("na", "br"), default="na", help="resonance type: narrow [na] or broad [br]")
    parser.add_option("-l", "--loss", dest="lossId", type=str, default="rk5_10", help="loss combination strategy")
    (options,args) = parser.parse_args()

    run = options.run
    mass = options.mass
    sigFile = options.sigFile
    qcdFile = options.qcdFile
    inputDir = options.inputDir
    sigRes = options.sigRes
    xsec = np.array(get_xsec_scan(options.sigFile))

    # distinctive run string
    run_str = make_run_str(sig_name=options.sigFile, sig_xsec=options.sigXsec, run_n=options.run_n, loss_id=options.lossId)
    out_dir = run_str[1:]

    if len(xsec) == 0:
        print "ERROR: set the cross sections to scan for signal",sigFile,"in the files_count.json file!"
        sys.exit()

    # quantiles = ['q1','q5','q10','q30','q50','q70','q90','q100','total']
    quantiles = ['q01', 'q10', 'q30', 'q50', 'q70', 'q90','q100','total']

    #if you have already run the scan, results are saved in txt files 
    if run == 0:
        plotPValue(xsec, quantiles+['final'])
        sys.exit()

    #first make workspaces (signal xsec set default to 0! -> assuming 1000fb of signal -> deriving sig histo scaling constant from that (???))
    cmd = "python dijetfit.py -i {inputdir} --sig {sigfile} --qcd {qcdfile} --xsec 0.0 -M {mass} --res {res} --out {out_dir}".format(inputdir=inputDir, sigfile=sigFile, qcdfile=qcdFile, mass=mass, res=sigRes, out_dir=out_dir)
    print cmd
    os.system(cmd)

    #now run the scan
    x = array('d', xsec)
    ysig = {}
    ypvalue = {}
    outfiles = []
    for q in quantiles:
        ysig[q] = []
        ypvalue[q] = []
        outfiles.append(open(out_dir+'/results_%s.txt'%q,'w'))

    # [0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1] (of 1 pico-barn)
    for x in xsec:
        for iq,q in enumerate(quantiles):
     
            cmd = 'cd {out_dir} && combine -M Significance workspace_JJ_0.0_{label}.root -m {mass} --expectSignal={xsec} -n significance_{xsec}_{label} -t -1'.format(out_dir=out_dir, xsec=x, label=q, mass=int(mass))
            print cmd
            os.system(cmd)

            cmd = 'cd {out_dir} && combine -M Significance workspace_JJ_0.0_{label}.root -m {mass} --expectSignal={xsec} -n pvalue_{xsec}_{label} -t -1 --pvalue'.format(out_dir=out_dir, xsec=x, label=q, mass=int(mass))
            print cmd
            os.system(cmd)
                
            tf = ROOT.TFile.Open('{out_dir}/higgsCombinesignificance_{xsec}_{label}.Significance.mH{mass}.root'.format(out_dir=out_dir, xsec=x,label=q,mass=int(mass)),'READ')
            tree = tf.limit
            tree.GetEntry(0)         
            ysig[q].append(tree.limit)
            print "Xsec",x,"quantile",q,"significance",ysig[q][-1]       
            tf.Close()

            tf = ROOT.TFile.Open('{out_dir}/higgsCombinepvalue_{xsec}_{label}.Significance.mH{mass}.root'.format(out_dir=out_dir, xsec=x,label=q,mass=int(mass)),'READ')
            tree = tf.limit
            tree.GetEntry(0)         
            ypvalue[q].append(tree.limit)        
            tf.Close()

            outfiles[iq].write('{xsec}\t{pvalue}\t{sig}\n'.format(xsec=x,pvalue=ypvalue[q][-1],sig=ysig[q][-1]))
                 
    for iq,q in enumerate(quantiles): outfiles[iq].close() 
 
    ysig['combo'] = []
    ypvalue['combo'] = []
    outfiles.append(open(out_dir+'/results_final.txt','w'))

    for x in xsec:

        cmd = 'cd {out_dir} && combine -M Significance workspace_0.0_{label}.root -m {mass} --expectSignal={xsec} -n significance_{xsec} -t -1'.format(out_dir=out_dir, xsec=x,label='final',mass=int(mass))
        print cmd
        os.system(cmd)

        cmd = 'cd {out_dir} && combine -M Significance workspace_0.0_{label}.root -m {mass} --expectSignal={xsec} -n pvalue_{xsec} -t -1 --pvalue'.format(out_dir=out_dir, xsec=x,label='final',mass=int(mass))
        print cmd
        os.system(cmd)
            
        tf = ROOT.TFile.Open('{out_dir}/higgsCombinesignificance_{xsec}.Significance.mH{mass}.root'.format(out_dir=out_dir, xsec=x,mass=int(mass)),'READ')
        tree = tf.limit
        tree.GetEntry(0)             
        ysig['combo'].append(tree.limit)             
        print "Xsec",x,"COMBO significance",ysig['combo'][-1]        
        tf.Close()

        tf = ROOT.TFile.Open('{out_dir}/higgsCombinepvalue_{xsec}.Significance.mH{mass}.root'.format(out_dir=out_dir, xsec=x,mass=int(mass)),'READ')
        tree = tf.limit
        tree.GetEntry(0)             
        ypvalue['combo'].append(tree.limit)          
        tf.Close()

        outfiles[-1].write('{xsec}\t{pvalue}\t{sig}\n'.format(xsec=x, pvalue=ypvalue['combo'][-1], sig=ysig['combo'][-1]))  
 
    outfiles[-1].close()
    
    print ysig
    print ypvalue

    plotPValue(xsec, quantiles + ['final'], run_str, out_dir=out_dir)
  

import h5py
import numpy as np
from re import sub
import sys
import subprocess
import fastjet
import awkward as ak
import os

filename = sys.argv[1]
year = sys.argv[3]
normalize = 1
deta_jj = 1.4
mjj = 1450
jPt = 300
yearstring = "run2" if int(year) == 999 else year

def xyze_to_eppt(constituents,normalize):
    ''' converts an array [N x 100, 4] of particles
from px, py, pz, E to eta, phi, pt (mass omitted)
    '''
    PX, PY, PZ, E = range(4)
    pt = np.sqrt(np.float_power(constituents[:,:,PX], 2) + np.float_power(constituents[:,:,PY], 2), dtype='float32') # numpy.float16 dtype -> float power to avoid overflow
    eta = np.arcsinh(np.divide(constituents[:,:,PZ], pt, out=np.zeros_like(pt), where=pt!=0.), dtype='float32')
    phi = np.arctan2(constituents[:,:,PY], constituents[:,:,PX], dtype='float32')

    if normalize == 1:
        print("Hi")
        return np.stack([pt, eta, phi], axis=2)
    if normalize == 0:
        print("Ho")
        return np.stack([constituents[:,:,PX], constituents[:,:,PY], constituents[:,:,PZ]], axis=2)


def recluster(array,normalize,jetkinematics,jetindex=0):
    j1s = np.array(array)
    totlength = len(j1s)
    for i in range(totlength):
        if i % 10000 == 0:
            print(i,' / ',totlength)
        j1_orig = j1s[i]
        #print(j1_orig)
        j1 = j1_orig[j1_orig[:,3]>0]
        cands = ak.zip({
            "px": j1[:,0].astype(np.float16),
            "py": j1[:,1].astype(np.float16),
            "pz": j1[:,2].astype(np.float16),
            "E": j1[:,3].astype(np.float16)
        }, with_name="Momentum4D")
        cluster = fastjet.ClusterSequence(cands, jetdef)
        jets = cluster.inclusive_jets(min_pt=jPt)
        chist = ak.Array(cluster.unique_history_order().to_list())
        chist = chist[chist<len(j1[:,0])]
        #print(chist)
        j1out = j1[chist,...]
        #print(j1out)
        j1final= np.pad(j1out,((0,100-len(j1out[:,0])),(0,0)),'constant')
        #print(j1final)
        j1s[i] = j1final

    j1s = xyze_to_eppt(j1s,normalize=normalize)
    if normalize == 1:
        j1s[:,:,0] = j1s[:,:,0]
        if jetindex == 0:
            j1s[:,:,1] = j1s[:,:,1]#-np.reshape(np.array(jetkinematics[:,3]),(-1,1))*(j1s[:,:,0]>0)
            j1s[:,:,2] = j1s[:,:,2]#-np.reshape(np.array(jetkinematics[:,4]),(-1,1))*(j1s[:,:,0]>0)
        if jetindex == 1:
            j1s[:,:,1] = j1s[:,:,1]#-np.reshape(np.array(jetkinematics[:,7]),(-1,1))*(j1s[:,:,0]>0)
            j1s[:,:,2] = j1s[:,:,2]#-np.reshape(np.array(jetkinematics[:,8]),(-1,1))*(j1s[:,:,0]>0)
        j1s[:,:,2] = np.where((j1s[:,:,2]<np.pi),j1s[:,:,2],j1s[:,:,2]-2*np.pi)
        j1s[:,:,2] = np.where((j1s[:,:,2]>-1*np.pi),j1s[:,:,2],j1s[:,:,2]+2*np.pi)

    return j1s.astype(np.float32)


        
with h5py.File(filename, "r") as f:

    types = {"0":"qcd","1":"graviton","2":"wp","3":"wkk","4":"bstar","-1":"singletop","-2":"ttbar","-3":"vjets","-4":"bkg"}
    type_name = types[sys.argv[2]]
    
    print("File %s"%filename)
    print("Creating %s"%type_name)

    subprocess.call("cp %s ."%filename,shell=True)
    localfile = filename.split("/")[-1]
    outfolder = filename.replace(localfile,"")
        

    f_side = sub('\.h5$', '_weird_side_%s_%s.h5'%(yearstring,type_name), localfile)

    if os.path.exists("%s/%s.lock"%(outfolder,f_side)):
        print("Job already running. Exiting.")
        exit(1)

    if os.path.exists("%s/%s"%(outfolder,f_side)):
        print("File already exists. Exiting.")
        exit(1)

    subprocess.call("touch %s/%s.lock"%(outfolder,f_side),shell=True)

    with open("tmplock.txt",'w') as textfile:
        textfile.write("%s/%s.lock"%(outfolder,f_side))
        textfile.write("\n")


    side_hf = h5py.File(f_side, 'w')

    #if int(sys.argv[2]) > 0: 
    #    type_mask = (f["truth_label"][:,0:1][:,0] == int(sys.argv[2]))
    #else:
    #    type_mask = (f["truth_label"][:,0:1][:,0] < 1)

    type_mask = (f["truth_label"][:,0:1][:,0] == int(sys.argv[2]))
    if int(sys.argv[2]) == -4:
        type_mask = (f["truth_label"][:,0:1][:,0] < 1)
    print(type_mask)


    pt1 = f["jet_kinematics"][:,2:3][:,0]
    pt2 = f["jet_kinematics"][:,6:7][:,0]
    pt3 = f["jet_kinematics"][:,10:11][:,0]
    mjj = f["jet_kinematics"][:,0:1][:,0]
    dEta = f["jet_kinematics"][:,1:2][:,0]

    side_mask = (dEta > 2.0) & (dEta < 2.5) & (pt1 > 300) & (pt2 > 300) & (pt3 < 300) & (  ((2 * pt1 * pt2 * (np.cosh(dEta)+1)/ (mjj*mjj))>1.0)  |      ((2 * pt1 * pt2 * (np.cosh(dEta)+1)/ (mjj*mjj))<0.95)  |    ( np.abs((pt1-pt2)/(pt1+pt2))>0.1  )   ) & type_mask & (f["event_info"][:,6:7][:,0] == int(year))
    if int(year) == 999:
        side_mask = (dEta > 2.0) & (dEta < 2.5) & (pt1 > 300) & (pt2 > 300) & (pt3 < 300) & (  ((2 * pt1 * pt2 * (np.cosh(dEta)+1)/ (mjj*mjj))>1.0)  |      ((2 * pt1 * pt2 * (np.cosh(dEta)+1)/ (mjj*mjj))<0.95)  |    ( np.abs((pt1-pt2)/(pt1+pt2))>0.1  )   ) & type_mask 

    side_event = np.array(f["event_info"])[side_mask].astype(np.float32)
    side_jj = np.array(f["jet_kinematics"])[side_mask].astype(np.float32)
    side_j1extra = np.array(f["jet1_extraInfo"])[side_mask].astype(np.float32)
    side_j2extra = np.array(f["jet2_extraInfo"])[side_mask].astype(np.float32)
    side_truth = np.array(f["truth_label"])[side_mask].astype(np.float32)
    
    # Reclustering
    #jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8)
    jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)

    # First jet    
    pf1 = np.array(f["jet1_PFCands"])
    side_pf1 = pf1[side_mask].astype(np.float32)  
    side_pf1 = recluster(side_pf1,normalize,side_jj)

    # Second jet    
    pf2 = np.array(f["jet2_PFCands"])
    side_pf2 = pf2[side_mask].astype(np.float32)  
    side_pf2 = recluster(side_pf2,normalize,side_jj,jetindex=1)

    j1_kinematics = np.stack((np.log(side_jj[:,2:3][:,0]),side_jj[:,3:4][:,0],side_jj[:,4:5][:,0],side_jj[:,5:6][:,0],side_jj[:,5:6][:,0]),axis=1)
    j2_kinematics = np.stack((np.log(side_jj[:,6:7][:,0]),side_jj[:,7:8][:,0],side_jj[:,8:9][:,0],side_jj[:,9:10][:,0],side_jj[:,9:10]
[:,0]),axis=1)

    jet_PFCands = np.concatenate((side_pf1,side_pf2),axis=0)
    jet_extra = np.concatenate((side_j1extra,side_j2extra),axis=0)
    jet_kinematics = np.concatenate((j1_kinematics,j2_kinematics),axis=0)

    side_hf.create_dataset('jet_PFCands', data=jet_PFCands, chunks = True, maxshape=(None,jet_PFCands.shape[1],4))
    side_hf.create_dataset('jet_kinematics', data=jet_kinematics, chunks = True, maxshape=(None,jet_kinematics.shape[1]))
    side_hf.create_dataset('jet_extraInfo', data=jet_extra, chunks = True, maxshape=(None,jet_extra.shape[1]))

    side_hf.close()

    subprocess.call("cp *_side_*.h5 %s"%outfolder,shell=True)

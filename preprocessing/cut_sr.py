import h5py
import numpy as np
from re import sub
import sys
import subprocess
import fastjet
import awkward as ak
import os

filename = sys.argv[1]
year = sys.argv[2]
normalize = 1
deta_jj = 1.4
mjj = 1450
jPt = 300

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

    
    print("File %s"%filename)

    subprocess.call("cp %s ."%filename,shell=True)
    localfile = filename.split("/")[-1]
    outfolder = filename.replace(localfile,"")
        

    f_sig = sub('\.h5$', '_sig_cambridge.h5', localfile)

    if os.path.exists("%s/%s.lock"%(outfolder,f_sig)):
        print("Job already running. Exiting.")
        exit(1)

    if os.path.exists("%s/%s"%(outfolder,f_sig)):
        print("File already exists. Exiting.")
        exit(1)

    subprocess.call("touch %s/%s.lock"%(outfolder,f_sig),shell=True)

    with open("tmplock.txt",'w') as textfile:
        textfile.write("%s/%s.lock"%(outfolder,f_sig))
        textfile.write("\n")

    if normalize == 0:
        f_sig = sub('_sig', '_unnorm_sig', localfile)

    sig_hf = h5py.File(f_sig, 'w')

    #if int(sys.argv[2]) > 0: 
    #    type_mask = (f["truth_label"][:,0:1][:,0] == int(sys.argv[2]))
    #else:
    #    type_mask = (f["truth_label"][:,0:1][:,0] < 1)


    sig_mask = (f["jet_kinematics"][:,1:2][:,0] < deta_jj) & (f["jet_kinematics"][:,2:3][:,0] > jPt) & (f["jet_kinematics"][:,6:7][:,0] > jPt) & (f["jet_kinematics"][:,0:1][:,0] > mjj) & (f["event_info"][:,6:7][:,0] == int(year)) 

    sig_event = np.array(f["event_info"])[sig_mask].astype(np.float32)
    sig_jj = np.array(f["jet_kinematics"])[sig_mask].astype(np.float32)
    sig_j1extra = np.array(f["jet1_extraInfo"])[sig_mask].astype(np.float32)
    sig_j2extra = np.array(f["jet2_extraInfo"])[sig_mask].astype(np.float32)
    sig_truth = np.array(f["truth_label"])[sig_mask].astype(np.float32)
    
    # Reclustering
    #jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8)
    jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 0.8)

    # First jet    
    pf1 = np.array(f["jet1_PFCands"])
    sig_pf1 = pf1[sig_mask].astype(np.float32)
    sig_pf1 = recluster(sig_pf1,normalize,sig_jj)

    # Second jet    
    pf2 = np.array(f["jet2_PFCands"])
    sig_pf2 = pf2[sig_mask].astype(np.float32)
    sig_pf2 = recluster(sig_pf2,normalize,sig_jj,jetindex=1)

    sig_hf.create_dataset('jet1_PFCands', data=sig_pf1)
    sig_hf.create_dataset('jet1_PFCands_shape', data=sig_pf1.shape)
    sig_hf.create_dataset('jet2_PFCands', data=sig_pf2)
    sig_hf.create_dataset('jet2_PFCands_shape', data=sig_pf2.shape)
    sig_hf.create_dataset('jet1_extra', data=sig_j1extra)
    sig_hf.create_dataset('jet1_extra_shape', data=sig_j1extra.shape)
    sig_hf.create_dataset('jet2_extra', data=sig_j2extra)
    sig_hf.create_dataset('jet2_extra_shape', data=sig_j2extra.shape)
    sig_hf.create_dataset('jet_kinematics', data=sig_jj)
    sig_hf.create_dataset('jet_kinematics_shape', data=sig_jj.shape)
    sig_hf.create_dataset('event_info', data=sig_event)
    sig_hf.create_dataset('event_info_shape', data=sig_event.shape)
    sig_hf.create_dataset('truth_label', data=sig_truth)
    sig_hf.create_dataset('truth_label_shape', data=sig_truth.shape)

    sig_hf.close()

    subprocess.call("cp *_sig*.h5 %s"%outfolder,shell=True)

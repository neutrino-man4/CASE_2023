import h5py
import numpy as np
import sys



filename = sys.argv[1]


print(filename)

with h5py.File(filename, "r") as f:
        # List all groups
        #print("Keys: %s" % f.keys())
    #a_group_key = list(f.keys())[0]
    print(f.keys())
    print(f['eventFeatureNames'][-6])
    print(f['eventFeatureNames'][-5])
    print(f['eventFeatureNames'][-4])
    print(f['eventFeatureNames'][-3])
    print(f['eventFeatureNames'][-2])
    print(f['eventFeatureNames'][-1])
    print(f['eventFeatures'])
    print(f['eventFeatures'].shape)
    print(f['eventFeatures'][0])


import h5py
import os


particles_key = 'jetConstituentsList'
particles_orig_key = 'jetOrigConstituentsList'
particles_names_key = 'particleFeatureNames'
jet_features_key = 'eventFeatures'
jet_feature_names_key = 'eventFeatureNames'


def write_jet_sample_to_file(jet_features, jet_feature_names, file_path):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(jet_features_key, data=jet_features,  compression='gzip', dtype='float32')
        f.create_dataset(jet_feature_names_key, data=[n.encode("utf-8") for n in jet_feature_names])


def write_event_sample_to_file(particles, event_features, particle_feature_names, event_feature_names, path ):
    '''
    write particles, particle_feature_names, jet_features, jet_feature_names to file
    '''
    with h5py.File(path,'w') as f:
        f.create_dataset(particles_key,data=particles,compression='gzip',dtype='float32')
        f.create_dataset(particles_names_key, data=[n.encode('utf-8') for n in particle_feature_names])
        f.create_dataset(jet_features_key, data=event_features, compression='gzip', dtype='float32')
        f.create_dataset(jet_feature_names_key, data=[n.encode('utf-8') for n in event_feature_names])


def write_event_sample_to_file_with_orig(particles, event_features, particle_feature_names, event_feature_names, path, orig_particles):
    '''
    write particles, particle_feature_names, jet_features, jet_feature_names to file
    '''
    with h5py.File(path,'w') as f:
        f.create_dataset(particles_key,data=particles,compression='gzip',dtype='float32')
        f.create_dataset(particles_orig_key,data=orig_particles,compression='gzip',dtype='float32')
        f.create_dataset(particles_names_key, data=[n.encode('utf-8') for n in particle_feature_names])
        f.create_dataset(jet_features_key, data=event_features, compression='gzip', dtype='float32')
        f.create_dataset(jet_feature_names_key, data=[n.encode('utf-8') for n in event_feature_names])


def write_bin_counts_to_file(counting_exp_dict, bin_edges, file_path):
    with h5py.File( file_path, 'w' ) as file_bin_counts:
        for sample, counts in counting_exp_dict.items():
            file_bin_counts.create_dataset(sample, data=counts)
        file_bin_counts.create_dataset('bin_count_labels',data=[n.encode("utf-8") for n in ['total','sig-like','bg-like']])
        file_bin_counts.create_dataset('bin_edges',data=bin_edges)


import numpy as np
import pandas as pd
import os
import operator
import pofah.util.result_writer as rw
import pofah.path_constants.sample_dict as sd
import pofah.util.data_converter as conv
import pofah.util.utility_fun as utfu
import sarewt.data_reader as dare


class EventSample():

    def __init__(self, name, particles=None, jet_features=None, particle_feature_names=None, jet_feature_names=None):
        '''
        datastructure that holds set of N events with each having two components: data of particles and data of jet features
        :param name: name of the sample
        :param particles: particle features like eta, phi, pt (numpy array) (todo: extend to preprocessed form like images (implement subclass?))
        :param jet_features: N x F_n features (pandas dataframe)
        '''
        self.name = name
        self.particles = np.asarray(particles) # numpy array [ N events x 2 (jets) x 100 particles x 3 features ]
        self.particle_feature_names = particle_feature_names
        # transform jet features to dataframe if passed as numpy array
        if not isinstance(jet_features, pd.DataFrame):
            jet_features = pd.DataFrame(jet_features, columns=jet_feature_names)
        self.jet_features =  jet_features

    @classmethod
    def from_input_file(cls, name, path, **cuts):
        reader = dare.DataReader(path)
        constituents, jet_features = reader.read_events_from_file(**cuts)
        constituents_feature_names, jet_feature_names = reader.read_labels_from_file()
        return cls(name, constituents, jet_features, constituents_feature_names, jet_feature_names)

    @classmethod
    def from_input_dir(cls, name, path, read_n=None, **cuts):
        ''' reading data in all files in 'path' to event sample'''
        reader = dare.DataReader(path)
        # import ipdb; ipdb.set_trace()
        
        constituents, constituents_feature_names, jet_features, jet_feature_names = reader.read_events_from_dir(read_n=read_n, **cuts)
        return cls(name, constituents, jet_features, constituents_feature_names, jet_feature_names)

    def __len__(self):
        return len(self.jet_features)

    def __getitem__(self, idx):
        # if idx is a string, return jet feature
        if isinstance(idx, str):
            return self.jet_features[idx]
        # else return sliced event sample
        return EventSample(name=self.name+'Sliced', particles=self.particles[idx], jet_features=self.jet_features[idx], particle_feature_names=self.particle_feature_names)

    def get_particles(self):
        ''' returning particles per jet as [2 x N x 100 x 3] '''
        return [self.particles[:,0,:,:], self.particles[:,1,:,:]]

    def get_event_features(self):
        return self.jet_features

    def add_event_feature(self, label, value):
        self.jet_features[label] = value

    def convert_to_cartesian(self, inplace=True):
        ''' transform cylindrical (eta, phi, pt) constituents to cartesian (px, py, pz) constituents '''
        converted_particles = conv.eppt_to_xyz(self.particles)
        if inplace:
            self.particles = converted_particles
            self.particle_feature_names = ['px', 'py', 'pz']
        else:
            self.converted_particles = converted_particles
        return self

    def dump(self,path):
        rw.write_event_sample_to_file(self.particles, self.jet_features.values, self.particle_feature_names, list(self.jet_features.columns), path)


class CCCCaseEventSample(EventSample):

    @classmethod
    def from_input_dir(cls, path, names=['qcdSig', 'GravToZZ_M3500_sig'], truth_ids=range(4)):
        reader = dare.CaseDataReader(path)
        constituents, constituents_names, features, features_names, truth_labels = reader.read_events_from_dir()
        samples = []
        for name, label in zip(names, truth_ids):
            # get examples for sample with truth label 'label'
            sample_const, sample_feat = utfu.filter_arrays_on_value(constituents, features, filter_arr=truth_labels.squeeze(), filter_val=label, comp=operator.eq)
            # convert particles from px, py, pz, E to eta, phi, pt (if not converting, drop E column)
            #converted_sample_const = conv.xyze_to_eppt(sample_const)
            # delete nan and inf values produced in conversion
            sample_const, sample_feat = conv.delete_nan_and_inf_events(sample_const, sample_feat)
            samples.append(cls(name, sample_const, sample_feat, features_names)) # constituents format [N x 2 x 100 x 3]  
        return samples



class CaseEventSample():

    def __init__(self, name, particles=None, jet_features=None, particle_feature_names=None, jet_feature_names=None, orig_particles=None):
        '''
        datastructure that holds set of N events with each having two components: data of particles and data of jet features
        :param name: name of the sample
        :param particles: particle features like eta, phi, pt (numpy array) (todo: extend to preprocessed form like images (implement subclass?))
        :param jet_features: N x F_n features (pandas dataframe)
        '''
        self.name = name
        self.particles = np.asarray(particles) # numpy array [ N events x 2 (jets) x 100 particles x 3 features ]
        if orig_particles == None:
            self.orig_particles = None
        else:
            self.orig_particles = np.asarray(orig_particles)

        #x = np.argsort(np.asarray(particles)[...,0], axis=2)
        #self.particles = np.take_along_axis(np.asarray(particles), x[...,None], axis=2)

        self.particle_feature_names = particle_feature_names
        # transform jet features to dataframe if passed as numpy array
        if not isinstance(jet_features, pd.DataFrame):
            try:
                jet_features = pd.DataFrame(jet_features, columns=jet_feature_names)
            except:
                jet_features = pd.DataFrame(jet_features, columns=jet_feature_names[:10])
        self.jet_features =  jet_features

    @classmethod
    def from_input_file(cls, name, path, **cuts):
        reader = dare.CaseDataReader(path)
        constituents, jet_features = reader.read_events_from_file(**cuts)
        constituents_feature_names, jet_feature_names = reader.read_labels_from_file()
        return cls(name, constituents, jet_features, constituents_feature_names, jet_feature_names)

    @classmethod
    def from_input_dir(cls, path, names=['qcdSig', 'GravToZZ_M3500_sig'], truth_ids=range(4)):
        reader = dare.CaseDataReader(path)
        constituents, constituents_names, features, features_names, truth_labels = reader.read_events_from_dir()
        samples = []
        for name, label in zip(names, truth_ids):
            # get examples for sample with truth label 'label'
            sample_const, sample_feat = utfu.filter_arrays_on_value(constituents, features, filter_arr=truth_labels.squeeze(), filter_val=label, comp=operator.eq)
            # convert particles from px, py, pz, E to eta, phi, pt (if not converting, drop E column)
            #converted_sample_const = conv.xyze_to_eppt(sample_const)
            # delete nan and inf values produced in conversion
            sample_const, sample_feat = conv.delete_nan_and_inf_events(sample_const, sample_feat)
            samples.append(cls(name, sample_const, sample_feat, features_names)) # constituents format [N x 2 x 100 x 3]  
        return samples

    def __len__(self):
        return len(self.jet_features)

    def __getitem__(self, idx):
        # if idx is a string, return jet feature
        if isinstance(idx, str):
            return self.jet_features[idx]
        # else return sliced event sample
        return EventSample(name=self.name+'Sliced', particles=self.particles[idx], jet_features=self.jet_features[idx], particle_feature_names=self.particle_feature_names)

    def get_particles(self):
        ''' returning particles per jet as [2 x N x 100 x 3] '''
        return [self.particles[:,0,:,:], self.particles[:,1,:,:]]

    def get_event_features(self):
        return self.jet_features

    def add_event_feature(self, label, value):
        self.jet_features[label] = value

    def convert_to_cartesian(self, inplace=True):
        ''' transform cylindrical (eta, phi, pt) constituents to cartesian (px, py, pz) constituents '''
        converted_particles = conv.eppt_to_xyz(self.particles)
        if inplace:
            self.particles = converted_particles
            self.particle_feature_names = ['px', 'py', 'pz']
        else:
            self.converted_particles = converted_particles
        return self

    def dump(self,path):
        rw.write_event_sample_to_file(self.particles, self.jet_features.values, self.particle_feature_names, list(self.jet_features.columns), path)

    def dump_with_orig(self,path):
        rw.write_event_sample_to_file_with_orig(self.particles, self.jet_features.values, self.particle_feature_names, list(self.jet_features.columns), path, self.orig_particles)

import os
from collections import OrderedDict
import pathlib
import pofah.path_constants.sample_dict as sd
import pofah.jet_sample as jesa
import pofah.util.event_sample as evsa
import pofah.util.utility_fun as utfu


class SamplePathDirFactory():

    def __init__(self, path_dict):
        self.base_dir = path_dict['base_dir']
        self.sample_dir = path_dict['sample_dir']
        self.sample_files = path_dict['file_names']

    def update_base_path(self, repl_dict):
        self.base_dir = utfu.multi_replace(self.base_dir, repl_dict)
        return self

    def sample_dir_path(self, id, mkdir=False):
        #print(self.sample_dir)
        #print("hereeee")
        #print(self.sample_dir)

        #print(id)

        if id in self.sample_dir:
            print("YES!")
        else:
            print("WTF!!!")

        s_path = os.path.join(self.base_dir, self.sample_dir[id])

        #print("WEIRRD!")
        if mkdir:
            pathlib.Path(s_path).mkdir(parents=True, exist_ok=True) # have to create directory for each sample here when writing results, not optimal, TODO: fix
        return s_path

    def sample_file_path(self, id, mkdir=False):
        #print("XXX")
        #print(self.sample_files)
        s_path = self.sample_dir_path(id, mkdir)

        #print("HAAAAAAA")
        #print(self.sample_files)
        #print(id)
        return os.path.join(s_path, self.sample_files[id]+'.h5')


##### utility functions

def read_inputs_to_sample_dict_from_file(sample_ids, paths):
    data = OrderedDict()
    for sample_id in sample_ids:
        data[sample_id] = js.JetSample.from_input_file(sample_id, read_fun(sample_id))
    return data

def read_inputs_to_sample_dict_from_dir(sample_ids, paths, cls, read_n=None, **cuts):
    data = OrderedDict()
    for sample_id in sample_ids:
        print('reading ', paths.sample_dir_path(sample_id))
        data[sample_id] = cls.from_input_dir(sample_id, paths.sample_dir_path(sample_id), read_n=read_n, **cuts)
    return data

def read_inputs_to_jet_sample_dict_from_dir(sample_ids, paths, read_n=None, **cuts):
    ''' read dictionary of JetSamples '''
    return read_inputs_to_sample_dict_from_dir(sample_ids, paths, jesa.JetSample, read_n=read_n, **cuts)

def read_inputs_to_event_sample_dict_from_dir(sample_ids, paths, read_n=None, **cuts):
    ''' read dictionary of EventSamples '''
    return read_inputs_to_sample_dict_from_dir(sample_ids, paths, evsa.EventSample, read_n=read_n, **cuts)

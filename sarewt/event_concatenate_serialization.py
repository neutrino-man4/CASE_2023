import argparse
import numpy as np

import data_reader as dr
import data_writer as dw

def compute_num_file_parts(constituents, features, mb_sz):
    mb_sz_total = (constituents.nbytes + features.nbytes) / 1024**2
    n_parts = int(np.ceil(mb_sz_total/mb_sz))
    print('divinding dataset of size {:6.2f} MB into {:d} chunks of size {:6.2f} MB'.format(mb_sz_total, n_parts, mb_sz))
    return n_parts


def split_concat_data(constituents, features, n_parts):
    return [np.array_split(constituents, n_parts), np.array_split(features, n_parts)]


def encode_uf8(data):
    return [[l.encode('utf-8') for l in dd] for dd in data]


def write_file_parts(constituents, constituent_names, features, feature_names, keys, file_name, mb_sz):
    n_file_parts = compute_num_file_parts(constituents, features, mb_sz)
    constituents_parts, features_parts = split_concat_data(constituents, features, n_file_parts)
    for i, (constituents_i, features_i) in enumerate(zip(constituents_parts, features_parts)):
        write_single_file([constituents_i, constituent_names, features_i, feature_names], keys, file_name, i)        


def write_single_file_part(data, keys, file_name, part_n):
    ext_idx = file_name.rindex('.')
    file_name_part = file_name[:ext_idx] + "_{:03d}".format(part_n) + file_name[ext_idx:] 
    write_file(data, keys, file_name_part)        


def write_file(data, keys, file_name):
    print('writing {} events to {}'.format(data[0].shape[0], file_name))
    dw.write_data_to_file(data, keys, file_name)


def read_concat_write(indir, file_name, max_n, mb_sz, side, sigreg):    
    reader = dr.DataReader(indir)
    cuts = {'mJJ': 1100.}
    if side:
        cuts['sideband'] = 1.4
    if sigreg:
        cuts['signalregion'] = 1.4
    
    keys = [l.encode('utf-8') for l in ['jetConstituentsList', 'particleFeatureNames', 'eventFeatures', 'eventFeatureNames']]
    particle_feature_names, dijet_feature_names = encode_uf8(reader.read_labels_from_dir())
    # write multiple file parts
    if mb_sz:
        for part_n, (constituents_concat, features_concat) in enumerate(reader.generate_event_parts_from_dir(min_sz_mb=mb_sz, **cuts)):
            write_single_file_part([constituents_concat, particle_feature_names, features_concat, dijet_feature_names], keys=keys, file_name=file_name, part_n=part_n)
    # write single concat file
    else: 
        constituents_concat, _, features_concat, _ = reader.read_events_from_dir(max_n=max_n, **cuts)
        write_file([constituents_concat, particle_feature_names, features_concat, dijet_feature_names], keys=keys, file_name=file_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='read, concatenate and write events from all files in directory')
    parser.add_argument('-in', dest='indir', type=str, help='input directory')
    parser.add_argument('-out', dest='outfile', type=str, default='out.h5', help='output file name/path')
    parser.add_argument('-n', dest='num_evts', type=int, default=1e9, help='max number of events for output dataset')
    parser.add_argument('-mb', dest='mb_sz', type=int, help='split concatenated dataset in multiple files, each of size mb [MB]')
    parser.add_argument('--side', dest='side', action='store_true', help='|dEta| > 1.4 sideband')
    parser.add_argument('--signal', dest='sigreg', action='store_true', help='|dEta| <= 1.4  signalregion')

    args = parser.parse_args()

    print('concatenating data in', args.indir)

    read_concat_write(args.indir, args.outfile, args.num_evts, args.mb_sz, args.side, args.sigreg)
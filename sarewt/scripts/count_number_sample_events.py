import sarewt.data_reader as dare
import os
import glob
import argparse

def count_number_events_recursively(base_dir, **cuts):

    sample_dirs =  glob.glob(base_dir+'/*')

    print('*'*10+'\ncounting number of events and files with {} in subdirectories of {}\n'.format((cuts or 'no'), base_dir)+'*'*10)

    num_files = []
    num_events = []

    for sample_dir in sample_dirs:
    	print('reading events in {}'.format(sample_dir))
    	reader = dare.DataReader(sample_dir)
    	n_files, n_events = reader.count_files_events_in_dir(**cuts)
    	num_files.append(n_files)
    	num_events.append(n_events)

    for sample_dir, nn, ff in zip(sample_dirs, num_events, num_files):
    	print("{: <52}: {: >10} events in {: >10} files".format(os.path.basename(sample_dir), nn, ff))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='count number of events in subdirectories of indir')
    parser.add_argument('-d', dest='base_dir', type=str, default='/eos/user/k/kiwoznia/data/VAE_data/events', help='input')
    parser.add_argument('--side', dest='side', action='store_true', help='|dEta| > 1.4 sideband')
    parser.add_argument('--signal', dest='sigreg', action='store_true', help='|dEta| <= 1.4  signalregion')
    args = parser.parse_args()

    cuts = {}
    if args.side:
        cuts['sideband'] = 1.4
    if args.sigreg:
        cuts['signalregion'] = 1.4

    count_number_events_recursively(args.base_dir, **cuts)
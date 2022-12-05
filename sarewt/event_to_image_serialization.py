import argparse
import numpy as np
import h5py

import data_reader as dr
import util as ut

class ImageSerializer():

    def __init__(self, n_bins):
        self.mjj_cut = 1100.
        self.n_bins = n_bins

    def read_file(self, path ):
        event_reader = dr.DataReader( path )
        events = event_reader.read_jet_constituents()
        dijet_features = event_reader.read_jet_features()
        labels = event_reader.read_labels()
        return [events[:, 0, :, :], events[:, 1, :, :], dijet_features, labels]


    def bin_data_to_image( self, events, bin_borders ):

        eventImagesShape = ( events.shape[0], self.n_bins, self.n_bins )
        images = np.zeros(eventImagesShape, dtype="float32")

        for eventNo, event in enumerate(events):  # for each event (100x3) populate eta-phi binned image with pt values
            # bin eta and phi of event event
            binIdxEta = np.digitize(event[:, 0], bin_borders, right=True) - 1  # np.digitize starts binning with 1
            binIdxPhi = np.digitize(event[:, 1], bin_borders, right=True) - 1
            for particle in range(event.shape[0]):
                images[eventNo, binIdxEta[particle], binIdxPhi[particle]] += event[particle, 2]  # add pt to bin of jet image

        return images


    def convert_events_to_image( self, events_j1, events_j2 ):

        minAngle = -0.8;
        maxAngle = 0.8
        bin_borders = np.linspace(minAngle, maxAngle, num=self.n_bins)  # bins for eta & phi

        return [ self.bin_data_to_image( events_j1, bin_borders ), self.bin_data_to_image( events_j2, bin_borders ) ]


    def normalize_by_jet_pt(self, images_j1, images_j2, jet_features, labels):
        idx_ptj1 = labels.index('j1Pt')
        idx_ptj2 = labels.index('j2Pt')
        images_j1 = np.divide(images_j1, jet_features[:, idx_ptj1, None, None])
        images_j2 = np.divide(images_j2, jet_features[:, idx_ptj2, None, None])
        return [images_j1, images_j2]


    def write_transformed(self, images, dijet_features, labels, out_path ):
        with h5py.File(out_path,'w') as f:
            f.create_dataset('images_j1_j2', data=images, compression='gzip', dtype='float32')
            f.create_dataset('eventFeatures', data=dijet_features, compression='gzip', dtype='float32')
            f.create_dataset('eventFeatureNames',data=[l.encode('utf-8') for l in labels]) # encode python3 unicode for h5py
        print('wrote {0} event image pairs to {1}'.format(dijet_features.shape[0],out_path))

    def shuffle_unisono(self, jet1_data, jet2_data, dijet_data):
        assert len(jet1_data) == len(jet2_data) == len(dijet_data)
        p = np.random.permutation(len(jet1_data))
        return [jet1_data[p], jet2_data[p], dijet_data[p]]

    def read_events_write_images(self, in_path, out_path, n_evts ):
        ''' reads: jet-constituents (2 jets x n events x m particles x 3 features ), dijet-features and labels,
            writes: jet-images ( 2 jets x n events x n_bins x n_bins x 1 channel ), dijet-features (unmodified), lables (unmodified)
            with events cut at mjj_cut and images normalized by pt
        '''
        # read events
        events_j1, events_j2, dijet_features, labels = self.read_file( in_path )
        n_evts_read = dijet_features.shape[0]
        print('read {} events'.format(n_evts_read))

        # reduce to number of events given
        if n_evts_read > n_evts:
            events_j1, events_j2, dijet_features = self.shuffle_unisono(events_j1, events_j2, dijet_features)
            events_j1, events_j2, dijet_features = events_j1[:n_evts], events_j2[:n_evts], dijet_features[:n_evts]

        # mass cut
        mjj_idx = labels.index('mJJ')
        events_j1, events_j2, dijet_features = ut.filter_arrays_on_value( events_j1, events_j2, dijet_features, filter_arr=dijet_features[:,mjj_idx], filter_val=self.mjj_cut )
        # convert to images
        images_j1, images_j2 = self.convert_events_to_image( events_j1, events_j2 )
        # normalize by pt
        images_j1, images_j2 = self.normalize_by_jet_pt( images_j1, images_j2, dijet_features, labels )
        # write images to file ( dim = 2 (jets) X n_events X n_bins X n_bins
        image_data = np.array([images_j1, images_j2], dtype="float32")
        self.write_transformed( image_data, dijet_features, labels, out_path )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='transform event data to jet image')
    parser.add_argument('-in', dest='infile', type=str, help='input file name/path')
    parser.add_argument('-out', dest='outfile', type=str, default='out.h5', help='output file name/path')
    parser.add_argument('-bin', dest='n_bins', type=int, default=32, help='number of bins in jet image')
    parser.add_argument('-n', dest='num_evts', type=int, default=1e9, help='number of events for output dataset')

    args = parser.parse_args()

    print('converting data in file', args.infile)

    serializer = ImageSerializer( args.n_bins )
    serializer.read_events_write_images( args.infile, args.outfile, args.num_evts )

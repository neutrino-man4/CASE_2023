import h5py

def write_data_to_file( datasets, dataset_names, file_path ):
    with h5py.File(file_path, 'w') as f:
        for dat, dat_name in zip(datasets,dataset_names):
            f.create_dataset(dat_name, data=dat, compression='gzip')

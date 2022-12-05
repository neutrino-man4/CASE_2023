import sarewt.data_reader as dare

#filename = 'BB_batch1.h5'
filename = '/data/t3home000/bmaier/CASE/BB_v3_2500/BB_batch3.h5'

reader = dare.CaseDataReader('.')
constituents, constituents_names, features, features_names, truth_labels = reader.read_events_from_dir()

print(constituents)

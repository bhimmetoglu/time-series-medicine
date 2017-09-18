import numpy as np
import pickle
from utils.get_prepare_data_full import * # Full features

def read_data_create_features(experiment_name, reshuffle = False):
	# Path
	path_name = "../data/" + experiment_name

	# Read data
	clips_interictal, clips_preictal, clips_test = get_data(data_folder = path_name)

	## Construct
	# Limits of frequency bands
	freq_bands = np.array([0.1, 4, 8, 12, 30, 70, 180])

	# Collect features
	X_0, Y_0, nb0 = features(clips_interictal, 0, freq_bands)
	X_1, Y_1, nb1 = features(clips_preictal, 1, freq_bands)
	X_test, nbt, test_dict = features(clips_test, None, freq_bands)

	if ( (nb0 != nb1) | (nb0 != nbt) | (nb1 != nbt)):
		raise ValueError('Mistmach in block size') 

	# Combine interictal and preictal
	X = np.concatenate((X_0, X_1), axis=0)
	Y = np.squeeze(np.concatenate((Y_0, Y_1), axis=0)).astype(int)

	# Save id numbers for each clip by block size
	clip_ids = [nb0*[i] for i in range(len(clips_interictal+clips_preictal))]
	clip_ids = np.array(clip_ids, dtype=int).reshape(-1)

	# Reshuffle
	if (reshuffle):
		np.random.seed(1)
		shuffle = np.random.choice(np.arange(len(Y)), size=len(Y), replace=False)
		X = X[shuffle]
		Y = Y[shuffle]
		clips_ids = clip_ids[shuffle]

	# Save for later
	fil_X = path_name + "_X.npy"
	fil_Y = path_name + "_Y.npy"
	fil_clip = path_name + '_clip_ids.npy'
	fil_test = path_name + "_test.npy"
	np.save(file = fil_X, arr = X)
	np.save(file = fil_Y, arr = Y)
	np.save(file = fil_clip, arr = clip_ids)
	np.save(file = fil_test, arr = X_test)

	# Dictionary for test set
	with open(path_name +'_test_dict.pickle', 'wb') as handle:
		pickle.dump(test_dict, handle)

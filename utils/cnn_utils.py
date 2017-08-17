## EEG classification utilities for 1d CNN
# Author: B. Himmetoglu
# 8/14/2017

import numpy as np
import pickle
import os
from scipy.io import loadmat
from utils.get_prepare_data_full import get_data

## Construct seuqnces from one segment
def cnn_tensor(input_segment, target, n_channels, n_blocks, block_len):
	""" Function for generating blocks of LSTM input tensors 
		input_segment : The EEG segment
		target        : 1/0 (preictal/interictial); None for test
		n_channels    : Number of channels
		n_blocks      : Number of blocks
		block_len     : Length of each block (= sampling_freq * block_s )
	"""

	# Dimensions
	n_channels, T_segment = input_segment.shape

	# Check block dimensions
	n_blk = (T_segment-1) // block_len # Number of blocks
	assert n_blk == n_blocks, "Block number mistmatch!, {:d} vs {:d}".format(n_blk, n_blocks)

	# Split into blocks
	input_segment = input_segment[:,:n_blocks*block_len]
	X = input_segment.reshape(-1, n_blocks, block_len)
	X = np.transpose(X, [1,2,0])
	# final shape: n_blocks, block_len, n_channels

	# Fill in the target
	if (target == 1):
		Y = np.ones(n_blocks)
	elif(target == 0):
		Y = np.zeros(n_blocks)
	else:
		Y = None

	# Return
	return X, Y

## Collect all the segments to build a tesnsor input for LSTM. Uses the function lstm_sequence
def cnn_build_input(clips, target, n_blocks = 6, n_ch = 16, s_freq = 600, block_s = 60):
	""" Collect all the data and build sequences for LSTM
		clips              : List of clips
		target             : 1/0 (preictal/interictial); None for test set
		s_freq             : Sampling frequency
		block_s            : Size of the block in seconds (default = 60)
	"""

	# Number of clips
	n_clips = len(clips)

	# For test only
	if target is None:
		test_dict = {}

	# Initiate
	block_len = s_freq * block_s   # Length of each block 
	X_full = np.zeros((n_clips*n_blocks, block_len, n_ch))
	Y_full = np.zeros((n_clips*n_blocks,1))

	# Loop over all clips and store data
	iclip = 0
	print("Target: ", target if target is not None else "test")
	for fil in clips:
		clip = loadmat(fil)
		segment_name = [el for el in list(clip.keys()) if "segment" in el][0] # Get segment name
		input_segment = clip[segment_name][0][0][0] # Get electrode data
		sampling_freq = np.squeeze(clip[segment_name][0][0][1]) # Sampling frequency
		assert sampling_freq == s_freq, "Wrong sampling frequency!"

		# Get number of channels
		n_channels = clip[segment_name][0][0][0].shape[0]
		assert n_channels == n_ch, "Wrong number of channels!"

		# Get tensor input and targets from blocks
		X, Y = cnn_tensor(input_segment, target, n_channels, n_blocks, block_len)

		X_full[iclip*n_blocks:(iclip+1)*n_blocks,:,:] = X
		if target is not None:
			Y_full[iclip*n_blocks:(iclip+1)*n_blocks] = Y[:,None]
		else:
			Y = None

			# Test set
			clip_name = os.path.split(fil)[-1]
			start = iclip * n_blocks
			stop = start + n_blocks
			test_dict[clip_name] = np.arange(start,stop, dtype=int).tolist()

		iclip +=1
		print("Processed {:d} of {:d} clips".format(iclip, len(clips)))


	# Return
	if target is not None:
		return X_full, Y_full
	else:
		return X_full, test_dict

## Collect all the data and print tensors for CNN
def read_data_create_tensors(experiment_name, n_blocks = 6, reshuffle = False, save = False):
	""" Read data from all clips and create tensors
	"""
	# Path
	path_name = "../data/" + experiment_name

	# Read data
	clips_interictal, clips_preictal, clips_test = get_data(data_folder = path_name)

	# Collect all tensors
	X_0, Y_0 = cnn_build_input(clips_interictal, 0)
	X_1, Y_1 = cnn_build_input(clips_preictal, 1)
	#X_test, test_dict = cnn_build_input(clips_preictal, None)

	# Combine interictal and preictal
	X = np.concatenate((X_0, X_1), axis=0)
	Y = np.squeeze(np.concatenate((Y_0, Y_1), axis=0)).astype(int)

	# Save id numbers for each clip by block size
	clip_ids = [n_blocks*[i] for i in range(len(clips_interictal+clips_preictal))]
	clip_ids = np.array(clip_ids, dtype=int).reshape(-1)

	# Reshuffle
	if (reshuffle):
		np.random.seed(1)
		shuffle = np.random.choice(np.arange(len(Y)), size=len(Y), replace=False)
		X = X[shuffle]
		Y = Y[shuffle]
		clips_ids = clip_ids[shuffle]

	# Save for later
	if (save):
		save_path = "../data/cnn/"
		fil_X = save_path + experiment_name + "_X.npy"
		fil_Y = save_path + experiment_name + "_Y.npy"
		fil_clip = save_path + experiment_name + '_clip_ids.npy'
		#fil_test = save_path + experiment_name + "_test.npy"
		np.save(file = fil_X, arr = X)
		np.save(file = fil_Y, arr = Y)
		np.save(file = fil_clip, arr = clip_ids)
		#np.save(file = fil_test, arr = X_test)

		# Dictionary for test set
		#with open(save_path + experiment_name +'_test_dict.pickle', 'wb') as handle:
		#	pickle.dump(test_dict, handle)
	else:
		return X, Y, clip_ids #, X_test

def concat_Dogs(block_s, save = True):
	""" Concatenate data from all dogs """

	# List of tensors
	tensor_list_X = []
	tensor_list_Y = []
	clip_id_dict = {}
	#tensor_list_Xtest = []

	dogs = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']
	for dog in dogs:
		X, Y, clip_ids, X_test = read_data_create_tensors(dog)
		tensor_list_X.append(X)
		tensor_list_Y.append(Y)
		#tensor_list_Xtest.append(X_test)
		clip_id_dict[dog] = clip_id_dict

	# Concatenate the data
	X = np.concatenate(tensor_list_X)
	Y = np.concatenate(tensor_list_Y)
	#X_test = np.concatenate(tensor_list_Xtest)

	# Save
	if (save):
		save_path = "../data/cnn/"
		prefix = str(window) + "_" + str(stride) + "_" + str(block_s)
		fil_X = save_path + prefix + "_X.npy"
		fil_Y = save_path + prefix + "_Y.npy"
		#fil_test = save_path + prefix + "_test.npy"
		np.save(file = fil_X, arr = X)
		np.save(file = fil_Y, arr = Y)
		#np.save(file = fil_test, arr = X_test)

		# Dictionary for test set
		#with open(save_path + prefix +'_clipId_dict.pickle', 'wb') as handle:
		#	pickle.dump(clip_id_dict, handle)
	else:
		return X, Y, clip_id_dict # X_test
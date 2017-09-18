## EEG classification utilities for 1d CNN
# Author: B. Himmetoglu
# Original    : 8/15/2017
# Last Update : 9/1/2017

from random import shuffle
import numpy as np
import pickle
import os
import glob
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, StratifiedKFold

## Get list of data
def list_data(data_path):

	clips_interictal = glob.glob(os.path.join(data_path, "*interictal*"))
	clips_preictal = glob.glob(os.path.join(data_path, "*preictal*"))
	clips_test = glob.glob(os.path.join(data_path, "*test*"))

	return clips_interictal, clips_preictal, clips_test

## Split into train/validation (stratified)
def split_data(subjects, validation_size = 0.2, random_state = 123):
	
	train_files = []
	labels = []
	# Loop over subjects
	for subj in subjects:
		path_name = "../data/" + subj

		# Get data
		interictal, preictal, test = list_data(path_name)

		# Splitting into batches
		files = interictal + preictal
		for el in files:
			t = 1 if "preictal" in el else 0
			labels.append(t)
			train_files.append(el)

	# Split into train/validation
	fil_tr, fil_vld, lab_tr, lab_vld = train_test_split(train_files, labels, stratify=labels, 
		test_size = validation_size, random_state = random_state)

	return fil_tr, fil_vld, lab_tr, lab_vld


## Batch generator: 
def get_batches(fil, lab, batch_size = 60, n_blk = 6, n_ch = 16, s_freq = 600, blk_s = 60):

	# For stratified splits
	n_preictal = sum(lab) # Number of preictal segments
	n_splits = len(fil) // n_preictal 
	skf = StratifiedKFold(n_splits = n_splits, random_state = 1234)

	for _, ii in skf.split(fil, lab):
		batch_segments = [fil[ifil] for ifil in ii]
		data_batch, label_batch = cnn_build_input(batch_segments, 
			n_blocks = n_blk, n_ch = n_ch, s_freq = s_freq, block_s = blk_s)

		n_batch = len(data_batch) // batch_size
		data_batch, label_batch = data_batch[:n_batch*batch_size], label_batch[:n_batch*batch_size]



## Batch generator (old) Note: final batch will have length batch_size*n_blocks
def get_batches_(fil, lab, batch_size = 60, n_blk = 6, n_ch = 16, s_freq = 600, blk_s = 60):

	# Number of batches
	n_batches = len(fil) // batch_size

	# Make sure that each batch has similar distribution
	skf = StratifiedKFold(n_splits = n_batches, random_state = 1234)

	for _, ii in skf.split(fil, lab):
		batch_segments = [fil[ifil] for ifil in ii]
		data_batch, label_batch = cnn_build_input(batch_segments, 
			n_blocks = n_blk, n_ch = n_ch, s_freq = s_freq, block_s = blk_s)
		
		yield data_batch, label_batch


## Collect all the segments to build a tesnsor input for CNN
def cnn_build_input(clips, n_blocks = 6, n_ch = 16, s_freq = 600, block_s = 60):
	""" Collect all the data and build sequences 
		clips              : List of clips
		s_freq             : Sampling frequency
		block_s            : Size of the block in seconds (default = 60)
	"""

	# Number of clips
	n_clips = len(clips)

	# Initiate
	block_len = s_freq * block_s   # Length of each block 
	X_full = np.zeros((n_clips*n_blocks, block_len, n_ch))
	Y_full = np.zeros((n_clips*n_blocks,1))

	# Loop over all clips and store data
	iclip = 0
	for fil in clips:
		clip = loadmat(fil)
		segment_name = [el for el in list(clip.keys()) if "segment" in el][0] # Get segment name
		input_segment = clip[segment_name][0][0][0] # Get electrode data
		sampling_freq = np.squeeze(clip[segment_name][0][0][1]) # Sampling frequency
		assert sampling_freq == s_freq, "Wrong sampling frequency!"

		# Get number of channels
		n_channels = clip[segment_name][0][0][0].shape[0]
		assert n_channels == n_ch, "Wrong number of channels!"

		# Get target
		target = 1 if "preictal" in segment_name else 0

		# Get tensor
		X, Y = cnn_tensor(input_segment, target, n_ch, n_blocks, block_len)

		# Save in full tensor
		X_full[iclip*n_blocks:(iclip+1)*n_blocks,:,:] = X
		Y_full[iclip*n_blocks:(iclip+1)*n_blocks] = Y[:,None]

		iclip += 1

	# Return
	return X_full, Y_full

## Construct seuqnces from one segment
def cnn_tensor(input_segment, target, n_channels, n_blocks, block_len):
	""" Function for generating blocks of input tensors 
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

	# Return
	return X, Y
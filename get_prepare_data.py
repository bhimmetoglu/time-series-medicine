## EEG classification utilities
# Author: B. Himmetoglu
# 7/6/2017

import numpy as np
import os
import h5py
import glob
from scipy.io import loadmat

# Function for reading an processing data
def get_data(data_folder):
	""" Read and process """

	# Get all clips
	clips = os.listdir(data_folder)

	# Store data from all the segments
	# Interictial
	clips_interictial = glob.glob(os.path.join(data_folder, "*interictal*"))

	# Preictal
	clips_preictal = glob.glob(os.path.join(data_folder, "*preictal*"))

	# Test clips (unclassified)
	clips_test = glob.glob(os.path.join(data_folder, "*test*"))

	# Return
	return clips_interictial, clips_preictal, clips_test

# Function for constructing basic features based on inter/intra channel correlations
def create_basic_features(clips, target, n = 16):
	""" Construct features 
	    clips  : List of clips
	    target : 1/0 (preictal/interictial)
	    n      : Number of channels
	"""

	num_observations = len(clips)
	num_features = n + n*(n-1)//2 
	
	# Initiate design matrix and outcomes
	X_train = np.zeros((num_observations, num_features))
	if (target == 0):
		Y_train = np.zeros(num_observations) # interictial
	else:
		Y_train = np.ones(num_observations) # preictal

	index = 0
	for fil in clips:
		clip = loadmat(fil)
		segment = list(clip.keys())[3] # Get segment name
		data = clip[segment][0][0][0] # Get electrode data

		if (clip[segment][0][0][0].shape[0] != n):
			raise ValueError('Wrong number of channels!')

		# Fill X
		intra_channel_var = np.var(data, axis=1)

		mat_corr = np.corrcoef(data)
		inter_channel_cor = mat_corr[np.triu_indices(n,k=1)]

		# Join
		X_train[index,:] = np.concatenate((intra_channel_var, inter_channel_cor), axis=0)

		index +=1
	
	# Return
	return X_train, Y_train

## Generate features based on blocks of each EEG segment. Increases the number of 
## data points. Following Howbert et. al 
def create_PIB_features(input_segment, target, freq_bands, sampling_freq, block_s = 60):
	""" Create Power in band (PIB) features
	    input_segment : The EEG segment
	    target        : 1/0 (preictal/interictial)
	    freq_bands    : The frequency bands where power is calculated
	    sampling_freq : Samplig frequency
	    block_s       : Size of the block in seconds (default = 60)
	"""

	# Dimensions
	n_channels, T_segment = input_segment.shape

	# Determine block dimensions
	block_len = sampling_freq * block_s   # Length of each block 
	n_blocks = (T_segment-1) // block_len # Number of blocks

	# Initiate design matrix
	n_features = n_channels * (len(freq_bands)-1)
	X = np.zeros((n_blocks, n_features))
	blocks = [block for block in range(0,(n_blocks+1)*block_len,block_len)]

	# Loop over blocks and fill X
	for ib in range(n_blocks):
		# Get block
		data_block = input_segment[:, blocks[ib]:blocks[ib+1]]

		## Construct Power features
		PS = np.abs(np.fft.fft(data_block))

		# Set zero frequency component to 0 (i.e. remove average) for all channels
		PS[:,0] = 0

		# Normalize
		PS = np.divide(PS, np.sum(PS, axis=1)[:,None])

		# Frequncy bands
		bands = np.round(PS.shape[1] / sampling_freq * freq_bands).astype(int)

		# Compute power spectrum for each band
		power_spect = np.zeros((n_channels, len(freq_bands)-1))
		for bb in range(len(freq_bands)-1):
			power_spect[:,bb] = 2.0*np.sum(PS[:,bands[bb]:bands[bb+1]], axis=1)

		# Store features in X
		X[ib,:] = power_spect.reshape([-1])

	# Targets
	if (target == 1):
		Y = np.ones(n_blocks)
	else:
		Y = np.zeros(n_blocks)

	# Return
	return X, Y

## Uses create_PIB_features to construct the full design matrix
def PIB(clips, target, freq_bands, block_s = 60, n_channels = 16):
	""" Collect all the data and construct PIB features 
        clips  : List of clips
        target : 1/0 (preictal/interictial)
        freq_bands    : The frequency bands where power is calculated
        block_s       : Size of the block in seconds (default = 60)
        n_channels    : Number of channels
	"""

	# Number of clips
	n_clips = len(clips)

	# Loop over all clips and store data
	iclip = 0
	for fil in clips:
		clip = loadmat(fil)
		segment_name = list(clip.keys())[3] # Get segment name
		input_segment = clip[segment_name][0][0][0] # Get electrode data
		sampling_freq = np.squeeze(clip[segment_name][0][0][1]) # Sampling frequency

		if (clip[segment_name][0][0][0].shape[0] != n_channels):
			raise ValueError('Wrong number of channels!')

		# Get PIB features
		X,Y = create_PIB_features(input_segment, target, freq_bands, sampling_freq, block_s)

		# Concatenate design matrix and target vector
		# This implementation takes care of unequal number of blocks 
		if (iclip == 0):
			X_train = X
			Y_train = Y[:,None]
		else:
			X_train = np.vstack((X_train, X))
			Y_train = np.vstack((Y_train, Y[:,None]))

		iclip += 1

	# Return 
	return X_train, Y_train



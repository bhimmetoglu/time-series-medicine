## EEG classification: Feature generation
# Author: B. Himmetoglu
# 7/19/2017

# Based on https://github.com/drewabbot/kaggle-seizure-prediction

import numpy as np
import os
import h5py
import glob
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.stats import skew

## all_data > segment > block

## Function for reading an processing data
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

## Get features from frequency band intervals per block
def get_band_features(power_block, freq_bands, sampling_freq, top_freq, p_power):
	""" 
	    power_block        : Power spectrum of given block
	    freq_bands         : The frequency bands where power is calculated
	    sampling_freq      : Samplig frequency
	    top_freq           : High frequency cut-off for computing spectral edge frequency
	    p_power            : Quantile for spectral edge frequency calculation 
	"""
	# Shapes
	n_channels, block_len = power_block.shape

	# Construct frequency bands
	bands = np.round(block_len / sampling_freq * freq_bands).astype(int)

	# Compute power spectrum (for each band)
	power_spect = np.zeros((n_channels, len(freq_bands)-1))
	for bb in range(len(freq_bands)-1):
		power_spect[:,bb] = 2.0*np.sum(power_block[:,bands[bb]:bands[bb+1]], axis = 1)

	# Compute entropy for each channel
	S_entropy = -np.sum(power_spect * np.log(power_spect), axis=1)

	# Spectral Edge Frequency
	top_freq_point = np.int(np.round(block_len / sampling_freq * top_freq))
	S_edge = np.percentile(power_block[:,0:top_freq_point], q = 100*p_power, axis = 1)

	# Eigenvalues of correlation matrix between channels
	C = np.corrcoef(power_spect) # Between channels
	C[np.isnan(C)] = 0
	C[np.isinf(C)] = 0
	eigenval, _ = np.linalg.eig(C)
	l_corr_channels = np.sort(np.real(eigenval))

	# Eigenvalues of correlation matrix between bands
	C = np.corrcoef(power_spect.T) # Between bands
	C[np.isnan(C)] = 0
	C[np.isinf(C)] = 0
	eigenval, _ = np.linalg.eig(C)
	l_corr_bands = np.sort(np.real(eigenval))

	# Return 
	return np.hstack((S_entropy, S_edge, l_corr_channels, l_corr_bands))

## Get features from dyadic levels per block
def get_dyadic_features(power_block):
	""" 
	    power_block : Power spectrum of given block
	"""
	# Shapes
	n_channels, block_len = power_block.shape

	# Initiate dyadic levels
	ldat = block_len // 2
	n_levels = np.floor(np.log2(ldat)).astype(int)

	# Compute power spectrum (for each dyadic level)
	power_spect = np.zeros((n_channels, n_levels))
	for bb in range(n_levels-1, -1, -1):
		power_spect[:,bb] = 2.0*np.sum(power_block[:,(ldat//2):ldat+1], axis = 1)
		ldat = ldat // 2 # Update length of sequence

	# Compute entropy for each channel
	S_entropy = -np.sum(power_spect * np.log(power_spect), axis=1)

	# Eigenvalues of correlation matrix between channels 
	C = np.corrcoef(power_spect) # Between channels
	C[np.isnan(C)] = 0
	C[np.isinf(C)] = 0
	eigenval, _ = np.linalg.eig(C)
	l_corr_channels = np.sort(np.real(eigenval))


	# Return 
	return np.hstack((S_entropy, l_corr_channels))

## Get features Hjorth parameters per block
def get_hjorth_parameters(data_block):
	""" 
	    Hjorth parameters per block for all channels
	"""
	# Activity
	activity = np.var(data_block, axis = 1)

	# Mobility
	mobility = np.std(np.diff(data_block),axis=1) / np.std(data_block, axis = 1)

	# Complexity
	complexity = np.std(np.diff(np.diff(data_block)), axis=1) / np.std(np.diff(data_block)) / mobility

	return np.hstack((activity, mobility, complexity))

## Get features from skewness and kurtosis per block
def get_stats(data_block):
	""" 
		Get skewness and kurtosis for all channels
	"""

	# Skewness
	sk = skew(data_block, axis = 1)

	# Kurtosis
	krt = kurtosis(data_block, axis = 1)

	return np.hstack((sk, krt))


## Generate features per segment. Uses the above functions.
def get_features(input_segment, target, freq_bands, sampling_freq, block_s = 60, top_freq = 40, p_power = 0.5):
	""" Engineer features from time-series 
	    input_segment      : The EEG segment
	    target             : 1/0 (preictal/interictial); None for test set
	    freq_bands         : The frequency bands where power is calculated
	    sampling_freq      : Samplig frequency
	    block_s            : Size of the block in seconds (default = 60)
	    top_freq           : High frequency cut-off for computing spectral edge frequency
	    p_power            : Quantile for spectral edge frequency calculation 
	"""

	# Get dimensions
	n_channels, T_segment = input_segment.shape
	block_len = sampling_freq * block_s   # Length of each block
	n_blocks = (T_segment-1) // block_len # Number of blocks

	# Initiate feature vectors: This will be completed later
	#feats = np.zeros(None*n_samples)

	feats = [] # Initiate as a list for now
	
	# Loop over all the blocks
	blocks = [block for block in range(0,(n_blocks+1)*block_len,block_len)]
	
	for ib in range(n_blocks):
		# Get interval from window
		data_block = input_segment[:,blocks[ib]:blocks[ib+1]]

		# Power spectrum (square root)
		power_block = np.abs(np.fft.fft(data_block))

		# Shift mean to 0 and normalize (mean is the zeroth Fourier component)
		power_block[:,0] = 0
		power_block = np.divide(power_block, np.sum(power_block, axis=1)[:,None])

		# Get features from bands
		f_1 = get_band_features(power_block, freq_bands, sampling_freq, top_freq, p_power)

		# Get features from dyadic bands
		f_2 = get_dyadic_features(power_block)

		# Get Hjorth parameters
		f_3 = get_hjorth_parameters(data_block)

		# Get statistical features
		f_4 = get_stats(data_block)

		# Combine and stack
		feats.append(np.hstack((f_1, f_2, f_3, f_4)))

	# Targets
	if (target == 1):
		Y = np.ones(n_blocks)
	elif (target == 0):
		Y = np.zeros(n_blocks)
	else:
		Y = None
		

	# Return
	return feats, Y, n_blocks

## Combine all the fearures to construct the full design matrix
def features(clips, target, freq_bands, block_s = 60, top_freq = 40, p_power = 0.5):
	""" Collect all egnineered features 
	    clips              : List of clips
	    target             : 1/0 (preictal/interictial); None for test set
	    freq_bands         : The frequency bands where power is calculated
	    block_s            : Size of the block in seconds (default = 60)
	    top_freq           : High frequency cut-off for computing spectral edge frequency
	    p_power            : Quantile for spectral edge frequency calculation 
	"""

	# Number of clips
	n_clips = len(clips)

	# For test set only
	if target is None:
		test_dict = {}

	# Loop over all clips and store data
	iclip = 0

	for fil in clips:
		clip = loadmat(fil)
		segment_name = [el for el in list(clip.keys()) if "segment" in el][0] # Get segment name
		input_segment = clip[segment_name][0][0][0] # Get electrode data
		sampling_freq = np.squeeze(clip[segment_name][0][0][1]) # Sampling frequency

		# Get number of channels
		n_channels = clip[segment_name][0][0][0].shape[0]

		# Get features
		X,Y, n_blocks = get_features(input_segment, target, freq_bands, sampling_freq, block_s, top_freq, p_power)

		# Concatenate design matrix and target vector
		if (iclip == 0):
			X_train = np.stack(X)
			Y_train = Y[:,None] if Y is not None else None
		else:
			X_train = np.vstack((X_train, np.stack(X)))
			Y_train = np.vstack((Y_train, Y[:,None])) if Y is not None else None

		# For test set only:
		if target is None:
			clip_name = os.path.split(fil)[-1]
			start = iclip * n_blocks
			stop = start + n_blocks
			test_dict[clip_name] = np.arange(start,stop, dtype=int).tolist()

		iclip += 1

	# Return 
	if target is not None:
		return X_train, Y_train, n_blocks
	else:
		return X_train, n_blocks, test_dict

## EEG classification utilities
# Author: B. Himmetoglu
# 7/20/2017

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

## Class for generating features
class Features(object):
	def __init__ (self, clips, target, freq_bands = None, block_s = 60, n_channels = 16):
		""" 
			Constructor
			clips      : List of clips
			target     : 1/0 (preictal/interictial)
			freq_bands : The frequency bands where power is calculated
			block_s    : Size of the block in seconds (default = 60)
			n_channels : Number of channels
		"""
		self.clips = clips
		self.target = target
		self.freq_bands = freq_bands
		self.block_s = block_s
		self.n_channels = n_channels

	def collect(self, feature_type = "basic"):
		""" Collect all the features that are generated """

		# Number of clips
		n_clips = len(self.clips)

		# Loop over all clips and store data
		iclip = 0
		for fil in self.clips:
			clip = loadmat(fil)
			segment_name = [el for el in list(clip.keys()) if "segment" in el][0] # Get segment name
			input_segment = clip[segment_name][0][0][0] # Get electrode data
			sampling_freq = np.squeeze(clip[segment_name][0][0][1]) # Sampling frequency

			if (clip[segment_name][0][0][0].shape[0] != self.n_channels):
				raise ValueError('Wrong number of channels!')

			# Get features
			if (feature_type == "basic"):
				X,Y = self.basic_features(input_segment)
			elif (feature_type == "PIB"):
				X,Y = self.PIB_features(input_segment, sampling_freq)

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

	def basic_features(self, input_segment):
		""" Create basic features based on inter channel correlations and intra channel variance on the full strip """

		# Number of clips
		n_clips = len(self.clips)

		# Number of features
		n_features = self.n_channels + self.n_channels * (self.n_channels-1) // 2

		# Intra channel variance
		intra_channel_var = np.var(input_segment, axis=1)

		# Inter channel 
		mat_corr = np.corrcoef(input_segment)
		inter_channel_cor = mat_corr[np.triu_indices(self.n_channels,k=1)]

		# Combine
		X = np.hstack((intra_channel_var, inter_channel_cor))
		
		# Targets
		if (self.target == 0):
			Y = np.zeros(1)
		elif(self.target == 1):
			Y = np.ones(1)

		# Return
		return X,Y

	def PIB_features(self, input_segment, sampling_freq):
		""" PIB features """

		# Dimensions
		n_channels, T_segment = input_segment.shape

		# Determine block dimensions
		block_len = sampling_freq * self.block_s   # Length of each block 
		n_blocks = (T_segment-1) // block_len # Number of blocks

		# Initiate design matrix
		n_features = self.n_channels * (len(self.freq_bands)-1)
		X = np.zeros((n_blocks, n_features))
		blocks = [block for block in range(0,(n_blocks+1)*block_len,block_len)]

		# Frequncy bands
		bands = np.round(block_len / sampling_freq * self.freq_bands).astype(int)

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

			# Compute power spectrum for each band
			power_spect = np.zeros((n_channels, len(self.freq_bands)-1))
			for bb in range(len(self.freq_bands)-1):
				power_spect[:,bb] = 2.0*np.sum(PS[:,bands[bb]:bands[bb+1]], axis=1)

			# Store features in X
			X[ib,:] = power_spect.reshape([-1])

		# Targets
		if (self.target == 1):
			Y = np.ones(n_blocks)
		else:
			Y = np.zeros(n_blocks)

		# Return
		return X, Y


## EEG classification utilities for 1d CNN
# Author: B. Himmetoglu
# Original    : 8/15/2017
# Last Update : 9/14/2017

from random import shuffle
import numpy as np
import pickle
import os
import glob
import random
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, StratifiedKFold

## Get list of data
def list_data(data_path):

	clips_interictal = glob.glob(os.path.join(data_path, "*interictal*"))
	clips_preictal = glob.glob(os.path.join(data_path, "*preictal*"))
	clips_test = glob.glob(os.path.join(data_path, "*test*"))

	return clips_interictal, clips_preictal, clips_test

## Split up each clip in pieces and return tensor
def split_segment(fil, new_seq_len = 1200):
	""" fil         : File containing the segment to be split
		new_seq_len : New segment length
		#	
		Given X.shape = (n_channels, T), the returned tensor will have shape
		[T/new_seq_len, seq_len, n_channels] 

	"""

	# Read from file
	clip = loadmat(fil)
	segment_name = [el for el in list(clip.keys()) if "segment" in el][0] # Get segment name
	input_segment = clip[segment_name][0][0][0] # Get electrode data
	sampling_freq = np.squeeze(clip[segment_name][0][0][1]) # Sampling frequency
	
	# Get number of channels
	n_channels = clip[segment_name][0][0][0].shape[0]
	
	# Split by sampling_freq
	n_ch, L_seg = input_segment.shape
	assert n_ch == n_channels, "Wrong number of channels!"

	# Total duration
	T_seg = int(np.ceil(L_seg / sampling_freq))

	# Save in another array and pad by zeros for processing
	X = np.zeros((n_channels, T_seg*sampling_freq))
	X[:,:input_segment.shape[1]] = input_segment

	# Target
	t = 1 if "preictal" in segment_name else 0

	# Initiate list of arrays and targets
	X_ = []
	Y_ = []
	indices = np.arange(X.shape[1]).reshape(-1, new_seq_len)
	for f in range(len(indices)):
		# Extract pieces
		temp = X[:,indices[f]]

		# Reshape into tensor
		X_.append(temp.reshape(1, temp.shape[1], temp.shape[0]))
		Y_.append(t)

	# Return
	return np.concatenate(X_, axis = 0), np.array(Y_, dtype = np.int)


## Generate samples from inputs
def generate_samples(seg_files, T_seg, new_seq_len = 1200, batch_size = 400):
	""" seg_files   : A given set of files containing 0/1 segments
		T_seg       : Exact length of one sequence
		new_seq_len : The new sequence length
		batch_size  : The size of the batch
	"""

	# Get 1/0 segments within a given list of files
	preictal_segs = [f for f in seg_files if "preictal" in f]
	interictal_segs = [f for f in seg_files if "interictal" in f]

	# Determine 0/1 ratio
	ratio_ = int(np.ceil(len(interictal_segs) / len(preictal_segs)))
	n_inter = ratio_ * len(preictal_segs) # Number of interictal segments for one preictal

	# Construct balanced segs
	balanced_segs = []
	for ii,pr in enumerate(preictal_segs):
		balanced_segs.append(pr)
		balanced_segs += interictal_segs[ii*n_inter:(ii+1)*n_inter]

	# Check size (remove later)
	assert len(balanced_segs) == len(seg_files), "Wrong number of segments in the balanced list!"

	# Batch ratio: (Batch size) / (# of blocks for each segment)
	len_data = T_seg // new_seq_len
	batch_ratio = batch_size // len_data

	assert batch_ratio > 0, "Batch size too small!"

	# Loop over segments and yield batches
	for ii in range(0,len(balanced_segs), batch_ratio):
		current_segs = balanced_segs[ii:(ii+batch_ratio)]
		X_ = []
		Y_ = []
		for jj in range(batch_ratio):
			x,y = split_segment(current_segs[jj],new_seq_len)
			X_.append(x)
			Y_.append(y)

		# Yield
		yield np.concatenate(X_, axis = 0), np.concatenate(Y_, axis=0)


## TO BE COMPLETED: A Function to create test samples
def generate_test_samples():
	pass

def split_train_validation(subject, validation_size = 0.2, random_state = 123):
	""" Split into training and test sets """

	# The location of data
	folder_name = os.path.join("../data/", subject)

	# Now divide into train/validation sets 
	interictal, preictal, test = list_data(folder_name)

	# Lists for training files and labels (0/1 : interictal/preictal)
	train_files = []
	labels = []

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


## Generate balanced samples from inputs
def generate_samples_balanced(seg_files, T_seg, new_seq_len = 1200, batch_size = 400):
	""" seg_files   : A given set of files containing 0/1 segments
		new_seq_len : The new sequence length
		batch_size  : 
	"""

	# Get 1/0 segments within a give list of files
	preictal_segs = [f for f in seg_files if "preictal" in f]
	interictal_segs = [f for f in seg_files if "interictal" in f]

	# Determine 0/1 ratio
	ratio_ = int(np.ceil(len(interictal_segs) / len(preictal_segs)))
	n_inter = ratio_ * len(preictal_segs) # Number of interictal segments for one preictal

	# Randomly shuffle each time the generator is called
	random.shuffle(interictal_segs) 

	# Loop over balaned (preictal,interictal) pieces and generate batches
	for ii,pr in enumerate(preictal_segs):
		X_pr, Y_pr = split_segment(pr, new_seq_len)
		#n_batch = len(X_pr) // batch_size
		#X_pr,Y_pr = X_pr[:n_batch*batch_size], Y_pr[:n_batch*batch_size]

		#assert batch_size <= len(X_pr), "Batch size must be at most {:d}".format(len(X_pr))

		for interic in interictal_segs[ii*ratio_:(ii+1)*ratio_]:
			X_itr, Y_itr = split_segment(interic, new_seq_len)
			n_batch = len(X_itr) // batch_size
			X_itr,Y_itr = X_itr[:n_batch*batch_size], Y_itr[:n_batch*batch_size]

			# Yield batches
			for b in range(0,len(X_itr), batch_size):
				X_batch = np.vstack((X_pr, X_itr[b:b+batch_size]))
				Y_batch = np.hstack((Y_pr, Y_itr[b:b+batch_size]))
				yield X_batch, Y_batch
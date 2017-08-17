## Collect time-series data and print into numpy arrays for further use
# Author: B. Himmetoglu
# 8/4/2017

import numpy as np
import pickle
from utils.lstm_utils import * # Full features
from utils.get_prepare_data_full import get_data

## Collect all the data and print tensors for LSTM
def read_data_create_tensors(experiment_name, window, stride, block_s = 60, reshuffle = False, save = False):
    """ Explain here...
    """
    # Path
    path_name = "../data/" + experiment_name

    # Read data
    clips_interictal, clips_preictal, clips_test = get_data(data_folder = path_name)

    # Collect all tensors
    X_0, Y_0, nb0 = lstm_build_input(clips_interictal, 0, window, stride, block_s)
    X_1, Y_1, nb1 = lstm_build_input(clips_preictal, 1, window, stride, block_s)
    X_test, nbt, test_dict = lstm_build_input(clips_preictal, None, window, stride, block_s)

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
    if (save):
        save_path = "../data/lstm/"
        fil_X = save_path + experiment_name + "_X.npy"
        fil_Y = save_path + experiment_name + "_Y.npy"
        fil_clip = save_path + experiment_name + '_clip_ids.npy'
        fil_test = save_path + experiment_name + "_test.npy"
        np.save(file = fil_X, arr = X)
        np.save(file = fil_Y, arr = Y)
        np.save(file = fil_clip, arr = clip_ids)
        np.save(file = fil_test, arr = X_test)

        # Dictionary for test set
        with open(save_path + experiment_name +'_test_dict.pickle', 'wb') as handle:
            pickle.dump(test_dict, handle)
    else:
        return X, Y, clip_ids, X_test

def concat_Dogs(window, stride, block_s, save = True):
    """ Concatenate data from all dogs """

    # List of tensors
    tensor_list_X = []
    tensor_list_Y = []
    clip_id_dict = {}
    tensor_list_Xtest = []

    dogs = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4']
    for dog in dogs:
        X, Y, clip_ids, X_test = read_data_create_tensors(dog, window, stride, block_s)
        tensor_list_X.append(X)
        tensor_list_Y.append(Y)
        tensor_list_Xtest.append(X_test)
        clip_id_dict[dog] = clip_id_dict

    # Concatenate the data
    X = np.concatenate(tensor_list_X)
    Y = np.concatenate(tensor_list_Y)
    X_test = np.concatenate(tensor_list_Xtest)

    # Save
    if (save):
        save_path = "../data/lstm/"
        prefix = str(window) + "_" + str(stride) + "_" + str(block_s)
        fil_X = save_path + prefix + "_X.npy"
        fil_Y = save_path + prefix + "_Y.npy"
        fil_test = save_path + prefix + "_test.npy"
        np.save(file = fil_X, arr = X)
        np.save(file = fil_Y, arr = Y)
        np.save(file = fil_test, arr = X_test)

        # Dictionary for test set
        with open(save_path + prefix +'_clipId_dict.pickle', 'wb') as handle:
            pickle.dump(clip_id_dict, handle)
    else:
        return X, Y, X_test, clip_id_dict

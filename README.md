# seizure-forecast

The aim of this project is to study EEG signals from subjects with epilepsy and predict oncoming seizures. The data is a collection of segments which contain recordings of signals in multiple channels. Each segment is to be classified as
preictal (oncoming seizure) or interictal (no seizure). Two main approaches are adopted for the task of classifying the 
segments:

* Manually engineered features used to train a classifier
* Long short term memory (LSTM) networks which take the (time-averaged) signals as inputs  

The data is available for downlaod from [Kaggle](https://www.kaggle.com/c/seizure-prediction/data). The data contains
clips of EEG time-series data corresponding to interictal (no oncoming seizure) and preictal (oncoming seuizure) pieces. 
Each clip is provided in '.mat' format and needs to be converted into numpy arrays. This is achieved by the functions in 
utils.

## Contents

Docs  | Contains basic documentation about the project                          |
HAR   | Contains study of the related Human Activity Recognition (HAR) problem  |
img   | Contains images used in notebooks                                       |
utils | Contains several utilities for data cleaning and feature engineering    |

## Data processing and cleaning 

The data is made up uf raw EEG signals in a number of channels. The data needs to 
be processed to construct features for modelling. This is achieved by the functions in the **utils** folder. 
Below are the desciption of these utilities. 

A simple summary of data preparation is explained in this [notebook](https://github.com/bhimmetoglu/seizure-forecast/blob/master/data_preparation.ipynb).

#### **get_prepare_data.py**

This piece of code performs data loading and construction of simple features for baseline models. Two types of feature
sets can be generated, described below:

1. basic features

The basic features rely on the variance of whole segments in each channels, as well as the correlation matrix between channels.

2. PIB features

PIB (Power in band) features are constructed using the power spectrum of blocks of 60s within each segment. They power spctra 
are calulated in a range of 6 frequency bands (0.1-4, 4-8, 8-12, 12-30, 30-70, 70-180 Hz). As a result, these features
contain only intra channel correlations. This was used in the original [publication](https://doi.org/10.1371/journal.pone.0081920) for some of the data.  

#### **get_prepare_data_full.py**

Construction of all the features adapted from a [Matlab solution](https://github.com/drewabbot/kaggle-seizure-prediction). Similar to PIB, 
each segment in split into blocks of 60s. Within each block, power spectra within 6 frequency bands (see above) are computed. These power
spectra are used to consruct the following features

* Eigenvalues of the correlation matrix between bands and channels within each block
* Shannon's entropy for the power within each block
* Power at dyadic levels, the eigenvalues of their correlation between channels and Shannon's entropy
* [Hjorth parameters](https://en.wikipedia.org/wiki/Hjorth_parameters)
* Skewness and Kurtosis within each block

#### **lstm_utils.py**

Construction of input tensors for LSTM networks. A 1d convolution is performed on the time-series in order to reduce the sequence length
to a value that LSTM networks can handle. 

#### **construct_save_features.py** and **construct_save_tensors.py**

Wraps the above utilities to process data and saves it for later use

## Data exploration
A simple data exploration is provided in this [notebook](https://github.com/bhimmetoglu/seizure-forecast/blob/master/explore.ipynb),
which includes some data visualization and basic modelling.


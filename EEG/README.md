# seizure-forecast

## Data processing and cleaning 

The data is made up uf raw EEG signals in a number of channels. The data needs to 
be processed to construct features for modelling. This is achieved by the functions in the **utils** folder. 
Below are the desciption of these utilities. 

A simple summary of data preparation is explained in this [notebook](https://github.com/bhimmetoglu/time-series-medicine/blob/master/EEG/data_preparation.ipynb).

Below is a description of data processing steps implemented in `utils`  

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

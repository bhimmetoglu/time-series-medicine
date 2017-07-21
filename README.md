# seizure-forecast

The data is available for downlaod from [Kaggle](https://www.kaggle.com/c/seizure-prediction/data). The data contains
clips of EEG time-series data corresponding to interictal (no oncoming seizure) and preictal (oncoming seuizure) pieces. 
Each clip is provided in '.mat' format and needs to be converted into numpy arrays. This is achieved by the functions in 
utils.

## Data processing and cleaning
Below are the desciption of some files

### **get_prepare_data.py**

The data is available for downlaod from [Kaggle](https://www.kaggle.com/c/seizure-prediction/data). The data contains
clips of EEG time-series data corresponding to interictal (no oncoming seizure) and preictal (oncoming seuizure) pieces. 
Each clip is provided in '.mat' format and needs to be converted into numpy arrays. This is achieved by the *get_data*
function. Once the data is loaded, features are extracted by using options *basic* or *PIB*. 

1. basic features
The basic features rely on the variance of whole segments in each channels, as well as the correlation matrix between channels.

2. PIB features
PIB (Power in band) features are constructed using the power spectrum of blocks of 60s within each segment. They power spctra 
are calulated in a range of 6 frequency bands (0.1-4, 4-8, 8-12, 12-30, 30-70, 70-180 Hz). As a result, these features
contain only intra channel correlations. This was used in the original [publication](https://doi.org/10.1371/journal.pone.0081920) for some of the data.  

### **get_prepare_data_full.py**

Construction of all the features adapted from a [Matlab solution](https://github.com/drewabbot/kaggle-seizure-prediction). Similar to PIB, 
each segment in split into blocks of 60s. Within each block, power spectra within 6 frequency bands (see above) are computed. These power
spectra are used to consruct the following features

* Eigenvalues of the correlation matrix between bands and channels within each block
* Shannon's entropy for the power within each block
* Power at dyadic levels, the eigenvalues of their correlation between channels and Shannon's entropy
* [Hjorth parameters](https://en.wikipedia.org/wiki/Hjorth_parameters)
* Skewness and Kurtosis within each block

## Data exploration
A simple data exploration is provided by the explore [notebook](https://github.com/bhimmetoglu/seizure-forecast/blob/master/explore.ipynb).

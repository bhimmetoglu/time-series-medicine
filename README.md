# seizure-forecast

Code and documentation for seizure prediction from EEG data

## Data processing and cleaning
**get_prepare_data.py**

The data is available for downlaod from [Kaggle](https://www.kaggle.com/c/seizure-prediction/data). The data contains
clips of EEG time-series data corresponding to interictal (no oncoming seizure) and preictal (oncoming seuizure) pieces. 
Each clip is provided in '.mat' format and needs to be converted into numpy arrays. This is achieved by the *get_data*
function. Once the data is loaded, features are extracted by *create_basic_features* or *create_PIB_features*. 

## Data exploration
A simple data exploration is provided by the explore [notebook](https://github.com/bhimmetoglu/seizure-forecast/blob/master/explore.ipynb).

# time-series-medicine

The aim of this project is to study time-series data in healthcare and medical data. 

## Contents

Folder | Contents
------ | --------
Docs   | Contains basic documentation about the project  
EEG    | Contains study of the Seizure Forecast problem                        
HAR    | Contains study of the Human Activity Recognition (HAR) problem 

The first dataset is EEG signals from subjects with epilepsy. The data is a collection of segments which contain recordings of signals in multiple channels. Each segment is to be classified as preictal (oncoming seizure) or interictal (no seizure). Two main approaches are adopted for the task of classifying the segments:

* Manually engineered features used to train a classifier
* Long short term memory (LSTM) networks which take the (time-averaged) signals as inputs  

The data is available for downlaod from [Kaggle](https://www.kaggle.com/c/seizure-prediction/data). The data contains
clips of EEG time-series data corresponding to interictal (no oncoming seizure) and preictal (oncoming seuizure) pieces. 
Each clip is provided in '.mat' format and needs to be converted into numpy arrays. This is achieved by the functions in 
utils.    

The second dataset is the Human Activity Recognition dataset from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones). 




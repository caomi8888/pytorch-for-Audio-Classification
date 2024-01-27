# Urban Sound Classification Project

## Overview
This project is designed to classify audio signals into different categories based on urban sounds. It uses a dataset sourced from New York University's UrbanSound dataset, which contains 8732 labeled sound recordings. The dataset can be downloaded from [UrbanSound Dataset](http://urbansounddataset.weebly.com/). The audio samples are categorized into 10 distinct classes, providing a diverse collection of urban-related sounds for analysis and machine learning purposes.

## Modules
The project is structured into several modules, each handling a specific part of the machine learning pipeline:

- `urban_sounddata_processing.py`: Handles data preprocessing. This module samples the audio signals and performs spectral transformations to prepare the data for model training.

- `train_test_spilit.py`: Responsible for splitting the dataset into training, validation, and test sets with a ratio of 7:2:1, respectively.

- `cnn.py`: Constructs the Convolutional Neural Network (CNN) model architecture used for training on the processed data.

- `cnn_drop.py`: An experimental module where different neural network structures are tested, including the use of dropout layers to reduce overfitting.

- `train.py`: Begins the training process and saves the trained model parameters locally for future use without the need to retrain.

- `test.py`: Utilizes the trained model to perform validation and assess the model's performance.

## Data
To get started with the project, download the UrbanSound dataset from the following link: [UrbanSound Dataset](http://urbansounddataset.weebly.com/). Ensure that the data is correctly placed in the project directory as expected by the data processing scripts.

## Usage
Follow the instructions below to run the modules:

1. Run `urban_sounddata_processing.py` to preprocess the audio files.
2. Use `train_test_spilit.py` to split the data into appropriate datasets.
3. Execute `cnn.py` or `cnn_drop.py` to establish your model architecture.
4. Train the model using `train.py`.
5. Validate and test the model's performance with `test.py`.





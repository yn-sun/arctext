# ArcTextDemo

## Introduction

This project shows how to use NASBench dataset to generate ArcText. Besides, this project give an example that ArcText can be used to predict CNN preformance. 
The codes have been tested on Python 3.6.

Dependent packages:

- nasbench (see https://github.com/google-research/nasbench)
- tensorflow (==1.15.0)
- scikit-learn
- matplotlib
- scipy

Dependent dataset:

- nasbench_only108.tfrecord  (We use NAS-Bech-101 subset of the dataset with only models trained at 108 epochs. You can download it at: https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord and put it under *path* folder. More details are in https://github.com/google-research/nasbench)

## How to use

*demo1_generate_arcText.py* generate an arcText string by using a matrix and type list of a cell.

*demo2_convert_NASBench_to_dataset.py* convert NASBench to the dataset that is used for predicting accuracy of CNNs.

*demo3_run_evaluate_dataset.py* train and evaluate the dataset generate by *demo2_convert_NASBench_to_dataset.py*.
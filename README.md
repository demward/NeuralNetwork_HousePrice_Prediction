# NeuralNetwork_HousePrice_Prediction

## Requisites 

To use on your computer download the *Projet1_HousePricePrediction.ipynb* and the dataset *kc_house_data.csv* and put them in the same folder

## Import Libraries 

These are the libraries that we'll use for this project

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

## Data extraction and manipulation

We read the data using pandas and visualize the dataframe using houses.head()

### Remove outliers 

To remove outliers we count the number of houses with similar prices and delete the groups with length below 5.

### Split the data between test and train set 

We split the data (80% train and 20% test) considering the stratification to mantain similar distributions between the train set and test set

### Normalization and One Hot Encode 

We normalize the quantitative variables in the train test, and apply the same ruler to the test set. We One Hot Encode the Zipcode because it only work for identification purposes (higher zipcode doesn't imply a higher or lower price) 

## Model Implementation 

First we set a random seed, and a initializier to mantain the consistency. 

Afther that we define the architecture of the Neural Network (See image below) the optimizer (**Adam**), the loss function (**Mean Square Error**) and the epochs (we set 200 but use early stopping to prevent the overfitting) 

<img width="568" alt="NeuralNetwork_Arch" src="https://github.com/user-attachments/assets/4b77b5f3-908c-4324-bbfb-5c2280cf59e7">

We can saw an improvement in the loss function and the early stop at 147 epochs. 

![NeuralNetwork_LossFunction](https://github.com/user-attachments/assets/3c7faae1-a021-47fe-b20e-73946a0fc675)

## Results 

We predict the prices of the test set using the features of X test. 

![NeuralNetwork_Residuals](https://github.com/user-attachments/assets/3f007f0c-6819-46f7-b04f-55d614ebbcd6)

The skyblue bars represent the 90% of the data and the gray colored bars represent the 5% tails. 

We can see this Neural Network has an 0.1288 MAPE which means an accuracy of 0.8712. This result is better than the Random Forest one, but I'm stil looking for better parameters (loss function, optimizer, learning rate or another architecture) to improve the model. 

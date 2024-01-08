#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:15:48 2023

@author: pablo
"""

import os
import sys
import torch
import torch.nn as nn
from statistics import *
from astropy.io import fits
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from FGL3Dataset import FGL3Dataset
from LogisticRegression import LogisticRegression
from utils import run_iteration


# https://www.youtube.com/watch?v=qx6y1OX4S6A Astropy for opening fits files
# Use OS to find path https://www.pythoncheatsheet.org/cheatsheet/file-directory-path
wanted_type_col = 73
path = os.path.join(os.getcwd(), "../FITS/gll_psc_v16.fit")
mainfile = fits.open(path)
mainfile.info()
pointsourcecatalogue = mainfile[1]
# print(pointsourcecatalogue.info())
# print(pointsourcecatalogue.header)
# print(list(pointsourcecatalogue.header.keys()))

columns = 0
keymap = {}
mapkey = {}
for key in list(pointsourcecatalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = pointsourcecatalogue.header[key]
        mapkey[pointsourcecatalogue.header[key]] = int(key[5:len(key)])-1
        # print(pointsourcecatalogue.header[key])
        columns += 1
# print(columns)
keymap = OrderedDict(sorted(keymap.items()))
mapkey = OrderedDict(sorted(mapkey.items()))
print("keymap = ", keymap)
print()
print("mapkey = ", mapkey)
# This keymap contains all the names of the corresponding column, keymap[n] contains the name of the n+1th column

# pointsourcecatalogue.data[a][b] corresponds with the a+1th row and b+1th column in the catalogue
# It is recommended you open the fit file to help easily find corresponding values
print("pointsourcecatalogue.data[1][1]) = ", pointsourcecatalogue.data[1][1])

# Creating Training and Test Datasets
# The paper selects these classes of objects to be apart of the dataset
wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "agn", "ssrq", "sey"]
# The paper selects these columns/features to be the inputs that will be taken into account
wantedfeaturenames = ["Spectral_Index", "Variability_Index", "Flux_Density", "Unc_Energy_Flux100", "Signif_Curve"]
wantedfeatureindices = [mapkey[x] for x in wantedfeaturenames]
# The paper wants hardness ratios of fluxes
fluxlevels = ["Flux100_300", "Flux300_1000", "Flux1000_3000", "Flux3000_10000", "Flux10000_100000"]
fluxindices = [mapkey[x] for x in fluxlevels]
# Pytorch Datasets and ML Models are constructed in modular class/OOP style

sc = StandardScaler()
# Now we can make our dataset and dataloader
# More on basics of dataloaders here:
#   https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
# KrishivB: separated class FGL3Dataset(Dataset) into file FGL3Dataset.py and imported it
#           Just call its constructor here
dataset = FGL3Dataset(wanted_type_col, fluxindices, wantedtypes, wantedfeaturenames, pointsourcecatalogue)
print("len(dataset) = ", len(dataset))
torch.manual_seed(42)  # Set shuffle seed to a certain value for reproducibility
dataloader = DataLoader(dataset=dataset, shuffle=True)
# Splitting dataloader into train/dev/test sets
train_size = int(0.7 * len(dataloader.dataset))  # You did a 70%:30% train:test split
test_size = len(dataloader.dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, test_size])
print("train_size = ", train_size)
print("test_size = ", test_size)
print("len(dataloader) = ", len(dataloader))
print("len(dataloader.dataset)) = ", len(dataloader.dataset))
print("len(train_dataset) = ", len(train_dataset))
print("len(test_dataset) = ", len(test_dataset))
print("len(train_dataset.dataset) = ", len(train_dataset.dataset))


"""
dataiter = iter(dataloader)
data = next(dataiter)
features, label = data
print(features, label)
"""

# Logistic Regression Model
#   https://www.youtube.com/watch?v=OGpQxIkR4ao&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=8
# However I used 2 layers of neurons instead of 1 as it yields better results.
# 1 layers of neurons would just be 1/(1 + e^-(input1 * weight1 + input2 * weight2 + input3 * weight3 ... + bias))

# Training Logistic Regression Model/Model Building Procedure

# In 10 epochs, 1 epoch will use a different final subset from one of the 10 subsets for testing
best_p_values = []
for iteration in range(0, 10):
    total_samples = len(train_dataset)
    dev_predictions = []
    dev_inputs = []
    dev_labels = []
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(42)
    LogRegModel = LogisticRegression(train_dataset)  # Start from new model every epoch
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss which is the loss method that would most likely be used in this scenario
    optimizer = torch.optim.SGD(LogRegModel.parameters(), lr=0.0001)  # Stochastic Gradient Descent. I could use Adams but Adams is worse than regular SGD unless you
    # spend ages finetuning it which in that case it performs better, but I have 0 time to spare
    # Train on 9 subsets
    (dev_inputs, dev_labels) = run_iteration(LogRegModel, iteration, train_dataset, total_samples, criterion,
                                             optimizer, dev_inputs, dev_labels, True)

    # Train for more epochs, 1 ain't enough LOL, you need at least 20-30 for good performance '
    for epoch in range(0, 29):
        (dev_inputs, dev_labels) = run_iteration(LogRegModel, iteration, train_dataset, total_samples, criterion,
                                                 optimizer, dev_inputs, dev_labels, False)

    # Test on final subset (different every time) to figure out P threshold value
    with torch.no_grad():
        for i in range(0, len(dev_inputs)):
            dev_prediction = LogRegModel(dev_inputs[i])
            dev_predictions.append(dev_prediction)
        xpoints = []
        ypoints = []
        bestp = 0
        bestscore = 0
        bestpsrscore = 0
        bestagnscore = 0
        bestaccuracy = 0
        n_samples = len(dev_inputs)
        for pp in range(0, 1001):
            p = pp * 0.001
            psrcount = 0
            agncount = 0
            truepositive = 0  # True Positive Sensitivity
            truenegative = 0  # True Negative Specificity
            correct = 0  # Total Correct
            # True Negative = Pulsar Successfully Identified
            # True Positive = AGN Successfully Identified
            for i in range(0, len(dev_inputs)):
                if dev_labels[i] == 1:
                    agncount += 1
                else:
                    psrcount += 1
                if dev_predictions[i] >= p and dev_labels[i] == 1:
                    truepositive += 1
                    correct += 1
                if dev_predictions[i] < p and dev_labels[i]== 0:
                    truenegative += 1
                    correct += 1
            xpoints.append(1 if (agncount == 0) else (truepositive / (agncount + 0.0)))  # AGN Score/True Positive/Sensitivity
            ypoints.append(1 if (psrcount == 0) else (truenegative / (psrcount + 0.0)))  # Pulsar Score/True Negative/Specificity
            score = (1 if (agncount == 0) else (truepositive / (agncount + 0.0))) + (1 if (psrcount == 0) else (truenegative / (psrcount + 0.0)))
            # The paper calculates the score as sensitivity + specificity
            if score > bestscore:
                bestscore = score
                bestp = p
                bestagnscore = 1 if (agncount == 0) else (truepositive / (agncount + 0.0))
                bestpsrscore = 1 if (psrcount == 0) else (truenegative / (psrcount + 0.0))
                bestaccuracy = (correct / (n_samples + 0.0))
        plt.plot(xpoints, ypoints)
        print(f"Best P Threshold Value : {bestp}")
        print(f"true negative : {truenegative}")
        print(f"PSR count : {psrcount}")
        print(f"Best Score : {bestscore}")
        print(f"Best Pulsar/True Negative/Specificity : {bestpsrscore}")
        print(f"true positive : {truepositive}")
        print(f"AGN count : {agncount}")
        print(f"Best AGN/True Positive/Sensitivity : {bestagnscore}")
        print(f"correct : {correct}")
        print(f"No of samples : {n_samples}")
        print(f"Accuracy of Best P Fit : {bestaccuracy}")
        plt.plot(bestagnscore, bestpsrscore, marker="x", markersize=10, markeredgecolor="red")
        plt.xlabel = "AGN Score/True Positive/Sensitivity"
        plt.ylabel = "Pulsar Score/True Negative/Specificity"
        plt.show()
        best_p_values.append(bestp)

# Generating Sensitivity vs Specificity Graph
# We now have a list of best p-values, but I am not sure on how to generate a best-p-value from these values
#   so I will just take the average of them
print()
print("best_p_values = ", best_p_values)
optimal_p = mean(best_p_values)
print("optimal_p = ", optimal_p)
torch.manual_seed(42)
LogRegModel = LogisticRegression(train_dataset)  # Now we found the optimal p value, we are ready to actually start training and testing the model
criterion = nn.BCELoss()  # Binary Cross Entropy Loss which is the loss method that would most likely be used in this scenario
optimizer = torch.optim.SGD(LogRegModel.parameters(), lr=0.0001)  # Stochastic Gradient Descent. I could use Adams but Adams is worse than regular SGD unless you
# spend ages finetuning it which in that case it performs better but I have 0 time to spare
for epoch in range(0, 30):
    for i, (inputs, labels) in enumerate(train_dataset):
        y_predicted = LogRegModel(inputs)  # Insert/fit Model for prediction here
        loss = criterion(y_predicted, labels)
        # Pytorch Gradient Descent Procedure
        loss.backward()  # Backwards pass in back propagation
        optimizer.step()  # Update weights based on gradients
        optimizer.zero_grad()  # Reset gradients to 0
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            correct = 0
            optimal_p = 0
            total = len(test_dataset)
            for i, (inputs, labels) in enumerate(test_dataset):
                y_predicted = LogRegModel(inputs) #Insert/fit Model for prediction here
                if y_predicted < optimal_p and labels.item() == 0 or y_predicted > optimal_p and labels.item() == 1:
                    correct += 1
            print(f"Accuracy: {correct / (total + 0.0)}")
    with torch.no_grad():
        for i in range(0, len(dev_inputs)):
            dev_prediction = LogRegModel(dev_inputs[i])
            dev_predictions.append(dev_prediction)
        xpoints = []
        ypoints = []
        bestp = 0
        bestscore = 0
        bestpsrscore = 0
        bestagnscore = 0
        bestaccuracy = 0
        n_samples = len(dev_inputs)
        for pp in range(0, 1001):
            p = pp * 0.001
            psrcount = 0
            agncount = 0
            truepositive = 0  # True Positive Sensitivity
            truenegative = 0  # True Negative Specificity
            correct = 0  # Total Correct
            # True Negative = Pulsar Successfully Identified
            # True Positive = AGN Successfully Identified
            for i in range(0, len(dev_inputs)):
                if dev_labels[i] == 1:
                    agncount += 1
                else:
                    psrcount += 1
                if dev_predictions[i] >= p and dev_labels[i] == 1:
                    truepositive += 1
                    correct += 1
                if dev_predictions[i] < p and dev_labels[i]== 0:
                    truenegative += 1
                    correct += 1
            xpoints.append(1 if (agncount == 0) else (truepositive / (agncount + 0.0)))  # AGN Score/True Positive/Sensitivity
            ypoints.append(1 if (psrcount == 0) else (truenegative / (psrcount + 0.0)))  # Pulsar Score/True Negative/Specificity
            score = (1 if (agncount == 0) else (truepositive / (agncount + 0.0))) + (1 if (psrcount == 0) else (truenegative / (psrcount + 0.0)))
            # The paper calculates the score as sensitivity + specificity
            if score > bestscore:
                bestscore = score
                bestp = p
                bestagnscore = 1 if (agncount == 0) else (truepositive / (agncount + 0.0))
                bestpsrscore = 1 if (psrcount == 0) else (truenegative / (psrcount + 0.0))
                bestaccuracy = (correct / (n_samples + 0.0))
        # print(n_correct / (n_samples + 0.0))
        #
        plt.plot(xpoints, ypoints)
        print(f"Best P Threshold Value : {bestp}")
        print(f"Best Score : {bestscore}")
        print(f"Best Pulsar/True Negative/Specificity : {bestpsrscore}")
        print(f"Best AGN/True Positive/Sensitivity : {bestagnscore}")
        print(f"Accuracy of Best P Fit : {bestaccuracy}")
        plt.plot(bestagnscore, bestpsrscore, marker="x", markersize=10, markeredgecolor="red")
        plt.xlabel = "AGN Score/True Positive/Sensitivity"
        plt.ylabel = "Pulsar Score/True Negative/Specificity"
        plt.show()
        best_p_values.append(bestp)

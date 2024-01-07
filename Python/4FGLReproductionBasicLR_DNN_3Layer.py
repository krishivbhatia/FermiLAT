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
from FGL4Dataset import FGL4Dataset
from LogisticRegression_DNN_3Layer import LogisticRegression_DNN_3Layer, run_iteration


# https://www.youtube.com/watch?v=qx6y1OX4S6A Astropy for opening fits files
# Use OS to find path https://www.pythoncheatsheet.org/cheatsheet/file-directory-path
# KrishivB: The wanted_type_col has the wantedtypes. This is different between 3FGL and 4FGL file.
#           Define it here and pass it to the FGL4Dataset class constructor
wanted_type_col = 69
# KrishivB: Modified path to 4FGL file
path = os.path.join(os.getcwd(), "../FITS/gll_psc_v33.fit")
mainfile = fits.open(path)
pointsourcecatalogue = mainfile[1]
# print("pointsourcecatalogue.header = ", pointsourcecatalogue.header)
# print("list(pointsourcecatalogue.header.keys() = ", list(pointsourcecatalogue.header.keys()))

columns = 0
keymap = {}
mapkey = {}
for key in list(pointsourcecatalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = pointsourcecatalogue.header[key]
        mapkey[pointsourcecatalogue.header[key]] = int(key[5:len(key)])-1
        # KrishivB: Print column number, name
        # print(columns, pointsourcecatalogue.header[key])
        columns += 1
# KrishivB: Print # columns. Subtract 1 for column header
print("# columns = ", columns-1)
# This keymap contains all the names of the corresponding column, keymap[n] contains the name of the n+1th column
keymap = OrderedDict(sorted(keymap.items()))
mapkey = OrderedDict(sorted(mapkey.items()))
print()

# pointsourcecatalogue.data[a][b] corresponds with the a+1th row and b+1th column in the catalogue
# It is recommended you open the fit file to help easily find corresponding values

# Creating Training and Test Datasets
# The paper selects these classes of objects to be apart of the dataset
# KrishivB: Added AGN in caps
wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "AGN", "agn", "ssrq", "sey"]
# The 3FGL paper selects these columns/features to be the inputs that will be taken into account
# KrishivB: Modified to 4FGL column names. Details in Abdollahi 2020 paper
wantedfeaturenames = ["PL_Index", "LP_Index", "PLEC_IndexS", "Variability_Index", "Unc_PL_Flux_Density",
                      # KrishivB: added it but later commented oput since slightly better performance without it
                      # "Pivot_Energy",
                      "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",
                      "Unc_Flux1000", "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
wantedfeatureindices = [mapkey[x] for x in wantedfeaturenames]
# The 3FGL paper wants hardness ratios of fluxes
# KrishivB: found only 1 in 4FGL column names. Need to know if there are more
fluxlevels = ["Energy_Flux100"]
fluxindices = [mapkey[x] for x in fluxlevels]

sc = StandardScaler()
# Now we can make our dataset and dataloader
# More on basics of dataloaders here:
#   https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
# KrishivB: separated class FGL3Dataset(Dataset) into file FGL4Dataset.py and imported it
#           Just call its constructor here
dataset = FGL4Dataset(wanted_type_col, fluxindices, wantedtypes, wantedfeaturenames, pointsourcecatalogue)
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

# KrishivB: Put class LogisticRegression(nn.Module) in separate file LogisticRegression.py and imported it

best_p_values = []
# Training Logistic Regression Model/Model Building Procedure
# In 10 epochs, 1 epoch will use a different final subset from one of the 10 subsets for testing
# KrishivB: changed # of epochs heere
for iteration in range(0, 30):
    print()
    print("Iteration # ", iteration)
    total_samples = len(train_dataset)
    dev_predictions = []
    dev_inputs = []
    dev_labels = []
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(82)
    LogRegModel = LogisticRegression_DNN_3Layer(train_dataset)  # Start from new model every epoch
    # Binary Cross Entropy Loss which is the loss method that would most likely be used in this scenario
    criterion = nn.BCELoss()
    # Stochastic Gradient Descent. I could use Adams but Adams is worse than regular SGD unless you
    optimizer = torch.optim.SGD(LogRegModel.parameters(), lr=0.0001)
    print("optimizer = ", optimizer.__class__)
    # Train on 10 subsets
    # KrishivB: Put run_iteration code in LogisiticRegression.py as a separate method, imported it, and call here
    (dev_inputs, dev_labels) = run_iteration(LogRegModel, iteration, train_dataset, total_samples, criterion,
                                             optimizer, dev_inputs, dev_labels, True)

    # Train for more epochs, 1 ain't enough LOL, you need at least 20-30 for good performance '
    for epoch in range(0, 30):
        (dev_inputs, dev_labels) = run_iteration(LogRegModel, iteration, train_dataset, total_samples, criterion,
                                                 optimizer, dev_inputs, dev_labels, False)

    # Test on final subset (different every time) to figure out P threshold value
    with (torch.no_grad()):
        for i in range(0, len(dev_inputs)):
            dev_prediction = LogRegModel(dev_inputs[i])
            dev_predictions.append(dev_prediction)
        x_points = []
        y_points = []
        best_p = 0
        best_score = 0
        best_psr_score = 0
        best_agn_score = 0
        best_accuracy = 0
        best_correct = 0
        n_samples = len(dev_inputs)
        for pp in range(0, 1001):
            p = pp * 0.001
            psr_count = 0
            agn_count = 0
            true_positive = 0  # True Positive Sensitivity
            true_negative = 0  # True Negative Specificity
            correct = 0  # Total Correct
            # True Negative = Pulsar Successfully Identified
            # True Positive = AGN Successfully Identified
            for i in range(0, len(dev_inputs)):
                if dev_labels[i] == 1:
                    agn_count += 1
                else:
                    psr_count += 1
                if dev_predictions[i] >= p and dev_labels[i] == 1:
                    true_positive += 1
                    correct += 1
                if dev_predictions[i] < p and dev_labels[i] == 0:
                    true_negative += 1
                    correct += 1
            # AGN Score/True Positive/Sensitivity
            x_points.append(1 if (agn_count == 0) else (true_positive / (agn_count + 0.0)))
            # Pulsar Score/True Negative/Specificity
            y_points.append(1 if (psr_count == 0) else (true_negative / (psr_count + 0.0)))
            score = (1 if (agn_count == 0)
                     else (true_positive / (agn_count + 0.0))) + \
                          (1 if (psr_count == 0) else (true_negative / (psr_count + 0.0)))
            # The paper calculates the score as sensitivity + specificity
            if score > best_score:
                best_score = score
                best_p = p
                best_correct = correct
                best_agn_score = 1 if (agn_count == 0) else (true_positive / (agn_count + 0.0))
                best_psr_score = 1 if (psr_count == 0) else (true_negative / (psr_count + 0.0))
                best_accuracy = (correct / (n_samples + 0.0))
                best_true_positive = true_positive
                best_true_negative = true_negative
        # Krishiv B: Do not want plots in every iteration. Just call in end.
        # plt.plot(x_points, y_points)
        print(f"Best P Threshold Value : {best_p}")
        print("Best true negative (Pulsar Successfully Identified): ", best_true_negative)
        print(f"PSR count : {psr_count}")
        print(f"Best Score : {best_score}")
        print(f"Best Pulsar/True Negative/Specificity : {best_psr_score}")
        print("Best true positive (AGN Successfully Identified): ", best_true_positive)
        print(f"AGN count : {agn_count}")
        print(f"Best AGN/True Positive/Sensitivity : {best_agn_score}")
        print("Best correct : ", best_correct)
        print(f"No of samples : {n_samples}")
        print("Accuracy of Best P Fit: ", best_accuracy)
        # plt.plot(best_agn_score, best_psr_score, marker="x", markersize=10, markeredgecolor="red")
        # plt.xlabel("AGN Score/True Positive/Sensitivity")
        # plt.ylabel("Pulsar Score/True Negative/Specificity")
        # plt.show()
        best_p_values.append(best_p)

# Generating Sensitivity vs Specificity Graph
# Take the average of p-values to get optimal p-value
print()
print("best_p_values = ", best_p_values)
optimal_p = mean(best_p_values)
print("optimal_p = ", optimal_p)
torch.manual_seed(42)
# Now we found the optimal p value, we are ready to actually start training and testing the model
LogRegModel = LogisticRegression_DNN_3Layer(train_dataset)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(LogRegModel.parameters(), lr=0.0001)
for epoch in range(0, 10):
    print()
    print("Optimal_p epoch # ", epoch)
    print
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
                y_predicted = LogRegModel(inputs)  # Insert/fit Model for prediction here
                if y_predicted < optimal_p and labels.item() == 0 or y_predicted > optimal_p and labels.item() == 1:
                    correct += 1
            print(f"Accuracy: {correct / (total + 0.0)}")
    with (torch.no_grad()):
        for i in range(0, len(dev_inputs)):
            dev_prediction = LogRegModel(dev_inputs[i])
            dev_predictions.append(dev_prediction)
        x_points = []
        y_points = []
        best_p = 0
        best_score = 0
        best_psr_score = 0
        best_true_positive = 0
        best_true_negative = 0
        best_psr_score = 0
        best_agn_score = 0
        best_accuracy = 0
        best_correct = 0
        n_samples = len(dev_inputs)
        for pp in range(0, 1001):
            p = pp * 0.001
            psr_count = 0
            agn_count = 0
            true_positive = 0  # True Positive Sensitivity
            true_negative = 0  # True Negative Specificity
            correct = 0  # Total Correct
            # True Negative = Pulsar Successfully Identified
            # True Positive = AGN Successfully Identified
            for i in range(0, len(dev_inputs)):
                if dev_labels[i] == 1:
                    agn_count += 1
                else:
                    psr_count += 1
                if dev_predictions[i] >= p and dev_labels[i] == 1:
                    true_positive += 1
                    correct += 1
                if dev_predictions[i] < p and dev_labels[i] == 0:
                    true_negative += 1
                    correct += 1
            # AGN Score/True Positive/Sensitivity
            x_points.append(1 if (agn_count == 0) else (true_positive / (agn_count + 0.0)))
            # Pulsar Score/True Negative/Specificity
            y_points.append(1 if (psr_count == 0) else (true_negative / (psr_count + 0.0)))
            score = (1 if (agn_count == 0) else (true_positive / (agn_count + 0.0))) + \
                    (1 if (psr_count == 0) else (true_negative / (psr_count + 0.0)))
            # The paper calculates the score as sensitivity + specificity
            if score > best_score:
                best_score = score
                best_p = p
                best_correct = correct
                best_agn_score = 1 if (agn_count == 0) else (true_positive / (agn_count + 0.0))
                best_psr_score = 1 if (psr_count == 0) else (true_negative / (psr_count + 0.0))
                best_true_positive = true_positive
                best_true_negative = true_negative
                # print("pp, p, True Positive, True Negative: ", pp, p, true_positive, true_negative)
                best_accuracy = (correct / (n_samples + 0.0))
        plt.plot(x_points, y_points)
        print(f"Best P Threshold Value : {best_p}")
        print(f"Score : {score}")
        print(f"Best Score : {best_score}")
        print("Best True Positive (AGN Successfully Identified): ", best_true_positive)
        print(f"AGN count : {agn_count}")
        print(f"Best AGN/True Positive/Sensitivity : {best_agn_score}")
        print("Best True Negative (PSR Successfully Identified): ", best_true_negative)
        print(f"PSR count : {psr_count}")
        print(f"Best Pulsar/True Negative/Specificity : {best_psr_score}")
        print(f"Accuracy of Best P Fit : {best_accuracy}")
        plt.plot(best_agn_score, best_psr_score, marker="x", markersize=10, markeredgecolor="red")
        plt.xlabel("AGN Score/True Positive/Sensitivity")
        plt.ylabel("Pulsar Score/True Negative/Specificity")
        print("Best correct : ", best_correct)
        print(f"No of samples : {n_samples}")
        # plt.show()
        best_p_values.append(best_p)
print("best_p_values = ", best_p_values)
plt.show()

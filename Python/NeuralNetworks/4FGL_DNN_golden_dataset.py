#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:15:48 2023

@author: pablo
"""

import os
import sys
import math
import torch
import torch.nn as nn
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from Utils.utils import run_iteration
from DeepNeuralNetwork import DeepNeuralNetworkModel
from Utils.FGL4_GoldenDataset import FGL4_GoldenDataset
from Utils.FGL4Dataset_cls1_null import FGL4Dataset_Cls1_Null


# https://www.youtube.com/watch?v=qx6y1OX4S6A Astropy for opening fits files
# Use OS to find path https://www.pythoncheatsheet.org/cheatsheet/file-directory-path

# KrishivB: The wanted_type_col has the wantedtypes. This is different between 3FGL and 4FGL file.
#           Define it here and pass it to the FGL4Dataset class constructor
wanted_type_col = 74 # 69
# KrishivB: Modified path to 4FGL file
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33.fit")
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v22_4FGL_DR1.fit")
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
# KrishivB: keymap maps column # to name
#           keymap =  OrderedDict([(0, 'Source_Name'), (1, 'RAJ2000'), (2, 'DEJ2000') ...
keymap = OrderedDict(sorted(keymap.items()))
print("keymap = ", keymap)
# KrishivB: mapkey maps column name to #
#           mapkey =  OrderedDict([('0FGL_Name', 64), ('1FGL_Name', 65), ('1FHL_Name', 67) ...
mapkey = OrderedDict(sorted(mapkey.items()))
print("mapkey = ", mapkey)
print()

# pointsourcecatalogue.data[a][b] corresponds with the a+1th row and b+1th column in the catalogue
# It is recommended you open the fit file to help easily find corresponding values

# Creating Training and Test Datasets
# The paper selects these classes of objects to be apart of the dataset
# KrishivB: Added AGN in caps
wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "msp", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "AGN", "agn", "ssrq", "sey"]
# The 3FGL paper selects these columns/features to be the inputs that will be taken into account
# KrishivB: Modified to 4FGL column names. Details in Abdollahi 2020 paper
wantedfeaturenames = ["PL_Index", "LP_Index", "PLEC_Index", "Variability_Index", "Unc_PL_Flux_Density",
                      "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",
                      "Unc_Flux1000", "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
wantedfeatureindices = [mapkey[x] for x in wantedfeaturenames]
# The 3FGL paper wants hardness ratios of fluxes
# KrishivB: found only 1 in 4FGL column names. Need to know if there are more
flux_col = "Flux_Band"

sc = StandardScaler()
# Now we can make our dataset and dataloader
# More on basics of dataloaders here:
#   https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
# KrishivB: separated class FGL4Dataset(Dataset) into file FGL4Dataset.py and imported it
#           Just call its constructor here
dataset = FGL4_GoldenDataset(wanted_type_col, flux_col, wantedtypes, wantedfeaturenames, pointsourcecatalogue)
print("len(dataset) = ", len(dataset))
torch.manual_seed(42)  # Set shuffle seed to a certain value for reproducibility
dataloader = DataLoader(dataset=dataset, shuffle=True)
# Splitting dataloader into train/dev/test sets
train_size = int(1.0 * len(dataloader.dataset))  # You did a 70%:30% train:test split
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
# KrishivB: changed # of epochs here

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(82)
DNNModel = DeepNeuralNetworkModel(train_dataset)
# Stochastic Gradient Descent. I could use Adams but Adams is worse than regular SGD unless you
optimizer = torch.optim.SGD(DNNModel.parameters(), lr=0.0001)
print("optimizer = ", optimizer.__class__)
# Binary Cross Entropy Loss which is the loss method that would most likely be used in this scenario
criterion = nn.BCELoss()
tot_iter = 2000
print("total iterations = ", tot_iter)
for iteration in range(0, tot_iter):
    dev_inputs = dev_labels = []
    print("Iteration # ", iteration)
    total_samples = len(train_dataset)
    # KrishivB: Put run_iteration code in utils.py, imported it, and call here
    for epoch in range(30):
        (dev_inputs, dev_labels) = run_iteration(DNNModel, iteration, train_dataset, total_samples, criterion,
                                                 optimizer, dev_inputs, dev_labels, True)
# correct = correct_agn = correct_psr = total_agn = total_psr = 0
# for i, (inputs, labels) in enumerate(test_dataset):
#     y_predicted = NNModel(inputs)  # Insert/fit Model for prediction here
#     predicted_item = y_predicted.data.item()
#     print("  i, y_predicted, y_predicted.data, predicted_item = ", i, y_predicted, y_predicted.data, predicted_item)
#     predicted_val = 0 if (math.isnan(predicted_item)) else round(predicted_item)
#     label_val = labels.item()
#     total_agn += 1 if label_val == 1 else 0
#     total_psr += 1 if label_val == 0 else 0
#     match = (predicted_val == label_val)
#     if match:
#         correct += 1
#         correct_agn += 1 if label_val == 1 else 0
#         correct_psr += 1 if label_val == 0 else 0
#     print("i, predicted_item, predicted_val, label_val, correct, match = ",
#           i, predicted_item, predicted_val, label_val, correct, match)
#
# print("correct, correct_agn, correct_psr = ", correct, correct_agn, correct_psr)
# print("total, total_agn, total_psr = ", len(test_dataset), total_agn, total_psr)
# print("total sensitivity = ", correct/len(test_dataset))
# print("agn sensitivity = ", correct_agn/total_agn)
# print("psr sensitivity = ", correct_psr/total_psr)

print("*********************************************")
print("Predictions for Unassociated Sources")
print("*********************************************")
unassociated_agn = 0
unassociated_agn_05_06 = 0
unassociated_agn_06_07 = 0
unassociated_agn_07_08 = 0
unassociated_agn_08_09 = 0
unassociated_agn_09_095 = 0
unassociated_agn_095_10 = 0
unassociated_psr = 0
unassociated_psr_00_005 = 0
unassociated_psr_005_01 = 0
unassociated_psr_01_02 = 0
unassociated_psr_02_03 = 0
unassociated_psr_03_04 = 0
unassociated_psr_04_05 = 0
cls1_none_dataset = FGL4Dataset_Cls1_Null(wanted_type_col, flux_col, wantedtypes, wantedfeaturenames, pointsourcecatalogue)
print("len(cls1_none_dataset) = ", cls1_none_dataset.size)
cls1_none_dataloader = DataLoader(dataset=cls1_none_dataset, shuffle=True)
print("cls1_none_dataloader.dataset size=", len(cls1_none_dataloader.dataset))
train_dataset, test_dataset = torch.utils.data.random_split(cls1_none_dataloader.dataset, [0, len(cls1_none_dataloader.dataset)])
print("test_dataset size=", len(test_dataset))
for i, inputs in enumerate(test_dataset):
    y_predicted = DNNModel(inputs)  # Insert/fit Model for prediction here
    predicted_item = y_predicted.data.item()
    print("i, y_predicted, y_predicted.data, predicted_item = ", i, y_predicted, y_predicted.data, predicted_item)
    predicted_val = 0 if (math.isnan(predicted_item)) else round(predicted_item)
    print("  i, predicted_item, predicted_val = ", i, predicted_item, predicted_val)
    if predicted_val == 1:
        unassociated_agn += 1
        if predicted_item >= 0.95:
            unassociated_agn_095_10 += 1
        elif predicted_item >= 0.9:
            unassociated_agn_09_095 += 1
        elif predicted_item >= 0.8:
            unassociated_agn_08_09 += 1
        elif predicted_item >= 0.7:
            unassociated_agn_07_08 += 1
        elif predicted_item >= 0.6:
            unassociated_agn_06_07 += 1
        else:
            unassociated_agn_05_06 += 1
    if predicted_val == 0:
        unassociated_psr += 1
        if predicted_item >= 0.4:
            unassociated_psr_04_05 += 1
        elif predicted_item >= 0.3:
            unassociated_psr_03_04 += 1
        elif predicted_item >= 0.2:
            unassociated_psr_02_03 += 1
        elif predicted_item >= 0.1:
            unassociated_psr_01_02 += 1
        elif predicted_item >= 0.05:
            unassociated_psr_005_01 += 1
        else:
            unassociated_psr_00_005 += 1
print("total ", len(test_dataset))

# create data
df1 = pd.DataFrame([
    ['AGN', unassociated_agn_095_10, unassociated_agn_09_095, unassociated_agn_08_09, unassociated_agn_07_08,
     unassociated_agn_06_07, unassociated_agn_05_06]],
    columns=['Classification-DNN-'+str(tot_iter), '0.95-1.0', '0.90-0.95', '0.8-0.9', '0.7-0.8', '0.6-0.7', '0.5-0.6'])

df2 = pd.DataFrame([
    ['PSR', unassociated_psr_00_005, unassociated_psr_005_01, unassociated_psr_01_02, unassociated_psr_02_03,
     unassociated_psr_03_04, unassociated_psr_04_05]],
    columns=['Classification-DNN-'+str(tot_iter), '0.0-0.05', '0.05-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5'])

# plot grouped bar chart
ax1 = df1.plot(x='Classification-DNN-'+str(tot_iter),
        kind='bar',
        stacked=False,
        title=str(unassociated_agn)+' unassociated sources AGN classification')
for container in ax1.containers:
    ax1.bar_label(container)

ax2 = df2.plot(x='Classification-DNN-'+str(tot_iter),
        kind='bar',
        stacked=False,
        title=str(unassociated_psr)+' unassociated sources PSR classification')
for container in ax2.containers:
    ax2.bar_label(container)

plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:15:48 2023

@author: pablo
"""

import os
import math
import torch
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from random_forest import TorchRandomForestClassifier

from Utils.FGL4_GoldenDataset import FGL4_GoldenDataset
from Utils.FGL4Dataset_cls1_null import FGL4Dataset_Cls1_Null
from Utils.utils import dataset_to_features, dataset_to_features_labels


# https://www.youtube.com/watch?v=qx6y1OX4S6A Astropy for opening fits files
# Use OS to find path https://www.pythoncheatsheet.org/cheatsheet/file-directory-path
wanted_type_col = 69 # 64 # 74
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v22_4FGL_DR1.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v27_4FGL_DR2.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v31_4FGL_DR3.fit")
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33_4FGL_DR4.fit")
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
        # print(columns, pointsourcecatalogue.header[key])
        columns += 1
print("columns = ", columns-1)
keymap = OrderedDict(sorted(keymap.items()))
mapkey = OrderedDict(sorted(mapkey.items()))
print("keymap = ", keymap)
print()
print("mapkey = ", mapkey)
print()
# This keymap contains all the names of the corresponding column, keymap[n] contains the name of the n+1th column

# pointsourcecatalogue.data[a][b] corresponds with the a+1th row and b+1th column in the catalogue
# It is recommended you open the fit file to help easily find corresponding values

# Creating Training and Test Datasets
# The paper selects these classes of objects to be apart of the dataset
wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "msp", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "AGN", "agn", "ssrq", "sey"]
# The paper selects these columns/features to be the inputs that will be taken into account
wantedfeaturenames = ["PL_Index", "LP_Index", "PLEC_IndexS", "Variability_Index", "Unc_PL_Flux_Density",
                      "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",  "Unc_Flux1000",
                      "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
wantedfeatureindices = [mapkey[x] for x in wantedfeaturenames]
# The paper wants hardness ratios of fluxes
flux_col = "Flux_Band"

# Pytorch Datasets and ML Models are constructed in modular class/OOP style

sc = StandardScaler()
# Now we can make our dataset and dataloader
# More on basics of dataloaders here:
#   https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
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

print()
# print("Iteration # ", iteration)
total_samples = len(train_dataset)
dev_predictions = []
dev_inputs = []
dev_labels = []
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42)
# Krishiv Bhatia: Invoke TorchRandomForestClassifier
random_forest = TorchRandomForestClassifier(100, 500, 10)
print("random forest = ", random_forest.nb_trees, random_forest.nb_samples, random_forest.max_depth)

# Krishiv Bhatia: split train dataset into features and labels
train_features, train_labels = dataset_to_features_labels(train_dataset)
random_forest.fit(torch.FloatTensor(train_features), torch.LongTensor(train_labels))

# Krishiv Bhatia: split test dataset into features and labels
# test_features, test_labels = dataset_to_features_labels(test_dataset)
# print("*** Test Features ***")
# print(test_features)
# print("*** Test Labels ***")
# print(test_labels)
# print("test_size = ", len(test_features))
# correct = 0
# for i in range(test_size):
#     predicted_result = random_forest.predict(torch.FloatTensor(test_features[i]))
#     print("test_labels[i] = ", test_labels[i])
#     actual_result = test_labels[i]
#     print("i, predicted_result, actual_result = ", i, predicted_result, actual_result)
#     if int(predicted_result) == int(actual_result):
#         correct += 1
# print("correct = ", correct)
# print("sensitivity = ", correct*100/test_size)

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
test_size = len(test_dataset)
print("test_dataset size=", test_size)
# Krishiv Bhatia: split test dataset into features and labels
test_features = dataset_to_features(test_dataset)
correct = 0
for i in range(test_size):
    predicted_item = random_forest.predict(torch.FloatTensor(test_features[i]))  # Insert/fit Model for prediction here
    print("i, predicted_result = ", i, predicted_item)
    predicted_val = 0 if (math.isnan(predicted_item)) else round(predicted_item)
    print("  i, predicted_item, predicted_val = ", i, predicted_item, predicted_val)
    if predicted_item >= 0.5:
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
    else:
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
    columns=['Classification-RF-'+str(random_forest.nb_trees)+'-'+str(random_forest.nb_samples)+'-'+str(random_forest.max_depth), '0.95-1.0', '0.90-0.95', '0.8-0.9', '0.7-0.8', '0.6-0.7', '0.5-0.6'])

df2 = pd.DataFrame([
    ['PSR', unassociated_psr_00_005, unassociated_psr_005_01, unassociated_psr_01_02, unassociated_psr_02_03,
     unassociated_psr_03_04, unassociated_psr_04_05]],
    columns=['Classification-RF-'+str(random_forest.nb_trees)+'-'+str(random_forest.nb_samples)+'-'+str(random_forest.max_depth), '0.0-0.05', '0.05-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5'])

# plot grouped bar chart
ax1 = df1.plot(x='Classification-RF-'+str(random_forest.nb_trees)+'-'+str(random_forest.nb_samples)+'-'+str(random_forest.max_depth),
        kind='bar',
        stacked=False,
        title=str(unassociated_agn)+' unassociated sources AGN classification')
for container in ax1.containers:
    ax1.bar_label(container)

ax2 = df2.plot(x='Classification-RF-'+str(random_forest.nb_trees)+'-'+str(random_forest.nb_samples)+'-'+str(random_forest.max_depth),
        kind='bar',
        stacked=False,
        title=str(unassociated_psr)+' unassociated sources PSR classification')
for container in ax2.containers:
    ax2.bar_label(container)

plt.show()

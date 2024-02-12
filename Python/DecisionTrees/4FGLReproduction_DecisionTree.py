#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:15:48 2023

@author: pablo
"""

import os

import torch
from astropy.io import fits
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from Utils.FGL4Dataset import FGL4Dataset
from Utils.utils import dataset_to_features_labels
from DecisionTrees.decision_tree import TorchDecisionTreeClassifier


# https://www.youtube.com/watch?v=qx6y1OX4S6A Astropy for opening fits files
# Use OS to find path https://www.pythoncheatsheet.org/cheatsheet/file-directory-path
wanted_type_col = 69
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33.fit")
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
print("columns = ", columns - 1)
keymap = OrderedDict(sorted(keymap.items()))
mapkey = OrderedDict(sorted(mapkey.items()))
# print("keymap = ", keymap)
print()
# print("mapkey = ", mapkey)
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
dataset = FGL4Dataset(wanted_type_col, flux_col, wantedtypes, wantedfeaturenames, pointsourcecatalogue)
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

print()
# print("Iteration # ", iteration)
total_samples = len(train_dataset)
dev_predictions = []
dev_inputs = []
dev_labels = []
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(42)
# Krishiv Bhatia: Invoke TorchDecisionTreeClassifier
decision_tree = TorchDecisionTreeClassifier(20)
print("TorchDecisionTreeClassifier = ", decision_tree.max_depth)

# Krishiv Bhatia: split train dataset into features and labels
train_features, train_labels = dataset_to_features_labels(train_dataset)
decision_tree.fit(torch.FloatTensor(train_features), torch.LongTensor(train_labels))

# Krishiv Bhatia: split test dataset into features and labels
test_features, test_labels = dataset_to_features_labels(test_dataset)
print("test_size = ", len(test_features))
correct = 0
for i in range(test_size):
    print("i = ", i)
    predicted_result = decision_tree.predict(torch.FloatTensor(test_features[i]))
    actual_result = test_labels[i]
    print("i, predicted_result, actual_result = ", i, predicted_result, actual_result)
    if int(predicted_result) == int(actual_result):
        correct += 1
        print("correct = ", correct)
print("Total correct = ", correct)
print("sensitivity = ", correct*100/test_size)
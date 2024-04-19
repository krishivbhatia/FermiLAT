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

from Utils.FGL4Dataset_Reporting import FGL4Dataset_Reporting


# https://www.youtube.com/watch?v=qx6y1OX4S6A Astropy for opening fits files
# Use OS to find path https://www.pythoncheatsheet.org/cheatsheet/file-directory-path

# KrishivB: The wanted_type_col has the wantedtypes. This is different between 3FGL and 4FGL file.
#           Define it here and pass it to the FGL4Dataset class constructor
wanted_type_col = 70
# KrishivB: Modified path to 4FGL file
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33_4FGL_DR4.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
# print("pointsourcecatalogue.header = ", pointsourcecatalogue.header)
# print("list(pointsourcecatalogue.header.keys() = ", list(pointsourcecatalogue.header.keys()))

columns = 0
keymap = {}
mapkey = {}
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        # KrishivB: Print column number, name
        print(columns, key, point_source_catalogue.header[key])
        columns += 1
# KrishivB: Print # columns. Subtract 1 for column header
print("# columns = ", columns-1)
# This keymap contains all the names of the corresponding column, keymap[n] contains the name of the n+1th column
# KrishivB: keymap maps column # to name
#           keymap =  OrderedDict([(0, 'Source_Name'), (1, 'RAJ2000'), (2, 'DEJ2000') ...
keymap = OrderedDict(sorted(keymap.items()))
# KrishivB: mapkey maps column name to #
#           mapkey =  OrderedDict([('0FGL_Name', 64), ('1FGL_Name', 65), ('1FHL_Name', 67) ...
mapkey = OrderedDict(sorted(mapkey.items()))
print()

# pointsourcecatalogue.data[a][b] corresponds with the a+1th row and b+1th column in the catalogue
# It is recommended you open the fit file to help easily find corresponding values

# Creating Training and Test Datasets
# The paper selects these classes of objects to be apart of the dataset
# KrishivB: Added AGN in caps
all_types = ["PSR", "psr", "YNG", "yng", "MSP", "msp", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "AGN", "agn", "ssrq", "sey"]
psr_types =  ["PSR", "psr", "MSP", "msp"]
agn_types = list(set(all_types) - set(psr_types))
# The 3FGL paper selects these columns/features to be the inputs that will be taken into account
# KrishivB: Modified to 4FGL column names. Details in Abdollahi 2020 paper

# Now we can make our dataset and dataloader
# More on basics of dataloaders here:
#   https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
# KrishivB: separated class FGL4Dataset(Dataset) into file FGL4Dataset.py and imported it
#           Just call its constructor here
dataset = FGL4Dataset_Reporting(point_source_catalogue, wanted_type_col, all_types, psr_types, agn_types)
plt.show()

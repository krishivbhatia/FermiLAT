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
from statistics import *
from astropy.io import fits
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from LogisticRegression import LogisticRegression
from Utils.FGL4_GoldenDataset import FGL4_GoldenDataset
from Utils.FGL4Dataset_cls1_null import FGL4Dataset_Cls1_Null
from Utils.utils import run_iteration, read_fits_file, get_keymap_mapkey, read_csv_file


wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "msp", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "AGN", "agn", "ssrq", "sey"]
wantedfeaturenames = ["PL_Index", "LP_Index", "PLEC_IndexS", "Variability_Index", "Unc_PL_Flux_Density",
                      "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",
                      "Unc_Flux1000", "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
flux_col = "Flux_Band"
wanted_type_col = 69 # 64 # 74
sc = StandardScaler()

path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33_4FGL_DR4.fit")
pointsourcecatalogue = read_fits_file(path)
keymap, mapkey = get_keymap_mapkey(pointsourcecatalogue)
wantedfeatureindices = [mapkey[x] for x in wantedfeaturenames]
dataset = FGL4_GoldenDataset(wanted_type_col, flux_col, wantedtypes,
                             wantedfeaturenames, pointsourcecatalogue)
print("len(dataset) = ", len(dataset))

df = read_csv_file("../../CSV/tbl6.txt")
print(df)
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
import pandas as pd
import torch.nn as nn
from astropy.io import fits
from collections import OrderedDict
from torch.utils.data import DataLoader
from Utils.FGL4Dataset_Processing import FGL4Dataset_Process


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# wanted_type_col
#   69 (4FGL_DR4, 4FGL_DR3), 64 (4FGL_DR2), 74 (4FGL_DR1)
#   73 (3FGL), 63 (2FGL), 54 (1FGL), 17 (BSL)
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc3month_BSL_v2.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v03_1FGL.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v09_2FGL.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v16.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v22_4FGL_DR1.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v27_4FGL_DR2.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v31_4FGL_DR3.fit")
# path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33_4FGL_DR4.fit")]

# Read BSL file
print("=====================BSL=======================")
path = os.path.join(os.getcwd(), "../../FITS/gll_psc3month_BSL_v2.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
columns = 0
keymap = {}
mapkey = {}
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        # print(columns, key, point_source_catalogue.headeddr[key])
        columns += 1
keymap = OrderedDict(sorted(keymap.items()))
mapkey = OrderedDict(sorted(mapkey.items()))

# Get BSL dataset
wanted_type_col = 17 # class1
dataset = FGL4Dataset_Process(point_source_catalogue, wanted_type_col, [])
bsl_list = dataset.get_all_list()
# Convert into dataframe
bsl_df = pd.DataFrame(bsl_list, columns=["BSL", "CLASS1-BSL"])
# Save into csv
bsl_df.to_csv('../../CSV/bsl_df.csv')

print("=====================1FGL=======================")
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v03_1FGL.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
columns = 0
keymap = {}
mapkey = {}
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        # print(columns, key, point_source_catalogue.header[key])
        columns += 1

# col 49 is 0FGL_Name:  78 TTYPE50 0FGL_Name
wanted_type_col = 54  # Class1
dataset = FGL4Dataset_Process(point_source_catalogue, wanted_type_col, [49])
fgl1_lst = dataset.get_all_list()
fgl1_df = pd.DataFrame(fgl1_lst, columns = ["1FGL", "CLASS1-1FGL", "1FGL-BSL"])
fgl1_df.to_csv('../../CSV/fgl1_df.csv')

print("=====================2FGL=======================")
columns = 0
keymap = {}
mapkey = {}
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v09_2FGL.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        # print(columns, key, point_source_catalogue.header[key])
        columns += 1

# 145 TTYPE57 0FGL_Name, 146 TTYPE58 1FGL_Name
wanted_type_col = 63 # class1
dataset = FGL4Dataset_Process(point_source_catalogue, wanted_type_col, [56,57])
fgl2_lst = dataset.get_all_list()
fgl2_df = pd.DataFrame(fgl2_lst, columns = ["2FGL", "CLASS1-2FGL", "2FGL-BSL", "2FGL-1FGL"])
fgl2_df.to_csv('../../CSV/fgl2_df.csv')

print("=====================3FGL=======================")
columns = 0
keymap = {}
mapkey = {}
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v16.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        # print(columns, key, point_source_catalogue.header[key])
        columns += 1

# TTYPE65 0FGL_Name, TTYPE66 1FGL_Name, TTYPE67 2FGL_Name
wanted_type_col = 73 # class1
dataset = FGL4Dataset_Process(point_source_catalogue, wanted_type_col, [64,65,66])
fgl3_lst = dataset.get_all_list()
fgl3_df = pd.DataFrame(fgl3_lst, columns = ["3FGL", "CLASS1-3FGL", "3FGL-BSL", "3FGL-1FGL", "3FGL-2FGL"])
fgl3_df.to_csv('../../CSV/fgl3_df.csv')
fgl3_join_4fgl_df = fgl3_df[["3FGL", "CLASS1-3FGL"]]

print("=====================4FGL-DR1=======================")
columns = 0
keymap = {}
mapkey = {}
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v22_4FGL_DR1.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        # print(columns, key, point_source_catalogue.header[key])
        columns += 1

# TTYPE65 0FGL_Name, TTYPE66 1FGL_Name, TTYPE67 2FGL_Name
wanted_type_col = 74 # class1
dataset = FGL4Dataset_Process(point_source_catalogue, wanted_type_col, [67])
fgl4dr1_lst = dataset.get_all_list()
fgl4dr1_df = pd.DataFrame(fgl4dr1_lst, columns = ["4FGLDR1",  "CLASS1-4FGLDR1", "4FGLDR1-3FGL"])
fgl4dr1_df.to_csv('../../CSV/fgl4dr1_df.csv')

print("=====================4FGL-DR2=======================")
columns = 0
keymap = {}
mapkey = {}
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v27_4FGL_DR2.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        # print(columns, key, point_source_catalogue.header[key])
        columns += 1

# TTYPE65 0FGL_Name, TTYPE66 1FGL_Name, TTYPE67 2FGL_Name
wanted_type_col = 64 # class1
dataset = FGL4Dataset_Process(point_source_catalogue, wanted_type_col, [57])
fgl4dr2_lst = dataset.get_all_list()
fgl4dr2_df = pd.DataFrame(fgl4dr2_lst, columns = ["4FGLDR2",  "CLASS1-4FGLDR2", "4FGLDR2-3FGL"])
fgl4dr2_df.to_csv('../../CSV/fgl4dr2_df.csv')

print("=====================4FGL-DR3=======================")
columns = 0
keymap = {}
mapkey = {}
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v31_4FGL_DR3.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        print(columns, key, point_source_catalogue.header[key])
        columns += 1

# TTYPE65 0FGL_Name, TTYPE66 1FGL_Name, TTYPE67 2FGL_Name
wanted_type_col = 69 # class1
dataset = FGL4Dataset_Process(point_source_catalogue, wanted_type_col, [62])
fgl4dr3_lst = dataset.get_all_list()
fgl4dr3_df = pd.DataFrame(fgl4dr3_lst, columns = ["4FGLDR3",  "CLASS1-4FGLDR3", "4FGLDR3-3FGL"])
fgl4dr3_df.to_csv('../../CSV/fgl4dr3_df.csv')

print("=====================4FGL-DR4=======================")
columns = 0
keymap = {}
mapkey = {}
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33_4FGL_DR4.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        print(columns, key, point_source_catalogue.header[key])
        columns += 1

# TTYPE65 0FGL_Name, TTYPE66 1FGL_Name, TTYPE67 2FGL_Name
wanted_type_col = 69 # class1
dataset = FGL4Dataset_Process(point_source_catalogue, wanted_type_col, [62])
fgl4dr4_lst = dataset.get_all_list()
fgl4dr4_df = pd.DataFrame(fgl4dr4_lst, columns = ["4FGLDR4",  "CLASS1-4FGLDR4", "4FGLDR4-3FGL"])
fgl4dr4_df.to_csv('../../CSV/fgl4dr4_df.csv')

print("=========================Merge-BSL-1FGL=========================")
joined_BSL_1FGL = pd.merge(bsl_df, fgl1_df, left_on='BSL', right_on='1FGL-BSL', how='left')
joined_BSL_1FGL.to_csv('../../CSV/joined_BSL_1FGL.csv')
joined_BSL_1FGL_None_CLS1BSL = joined_BSL_1FGL[joined_BSL_1FGL["CLASS1-BSL"]=='']
joined_BSL_1FGL_None_CLS1BSL.to_csv('../../CSV/joined_BSL_1FGL_NoneCLS1_BSL.csv')

print("=========================Merge BSL, 1FGL, 2FGL=========================")
joined_BSL_1FGL_2FGL = pd.merge(joined_BSL_1FGL, fgl2_df, left_on='BSL', right_on='2FGL-BSL', how='left')
joined_BSL_1FGL_2FGL.to_csv('../../CSV/joined_BSL_1FGL_2FGL.csv')
joined_BSL_1FGL_2FGL_NONE_CLS1_BSL = joined_BSL_1FGL_2FGL[joined_BSL_1FGL_2FGL["CLASS1-BSL"]=='']
joined_BSL_1FGL_2FGL_NONE_CLS1_BSL.to_csv('../../CSV/joined_BSL_1FGL_2FGL_NONE_CS1_BSL.csv')

print("=========================Merge BSL, 1FGL, 2FGL, with 3FGL=========================")
joined_BSL_1FGL_2FGL_3FGL = pd.merge(joined_BSL_1FGL_2FGL, fgl3_df, left_on='BSL', right_on='3FGL-BSL', how='left')
joined_BSL_1FGL_2FGL_3FGL.to_csv('../../CSV/joined_BSL_1FGL_2FGL_3FGL.csv')
joined_BSL_1FGL_2FGL_3FGL_NONE_CLS1_BSL = joined_BSL_1FGL_2FGL_3FGL[joined_BSL_1FGL_2FGL_3FGL["CLASS1-BSL"]=='']
joined_BSL_1FGL_2FGL_3FGL_NONE_CLS1_BSL.to_csv('../../CSV/joined_BSL_1FGL_2FGL_3FGL_NONE_CLS1_BSL.csv')

print("=========================Merge 3FGL, 4FGL-DR1=========================")
joined_3FGL_4FGLDR1 = pd.merge(fgl3_join_4fgl_df, fgl4dr1_df, left_on='3FGL', right_on='4FGLDR1-3FGL', how='left')
# joined_3FGL_4FGLDR1.to_csv('../../CSV/joined_3FGL_4FGLDR1.csv')

print("=========================Merge 3FGL, 4FGL-DR1, 4FGL-DR2=========================")
joined_3FGL_4FGLDR1_DR2 = pd.merge(joined_3FGL_4FGLDR1, fgl4dr2_df, left_on='3FGL', right_on='4FGLDR2-3FGL', how='left')
# joined_3FGL_4FGLDR1_DR2.to_csv('../../CSV/joined_3FGL_4FGLDR1_DR2.csv')

print("=========================Merge 3FGL, 4FGL-DR1, 4FGL-DR2, 4FGL-DR3=========================")
joined_3FGL_4FGLDR1_DR2_DR3 = pd.merge(joined_3FGL_4FGLDR1_DR2, fgl4dr3_df, left_on='3FGL', right_on='4FGLDR3-3FGL', how='left')
# joined_3FGL_4FGLDR1_DR2_DR3.to_csv('../../CSV/joined_3FGL_4FGLDR1_DR2_DR3.csv')

def flag_df(df):
    if df['CLASS1-3FGL'] == "" or df['CLASS1-3FGL'] is None:
        return 0
    elif df['CLASS1-4FGLDR1'] == "" or df['CLASS1-4FGLDR1'] is None \
            or df['CLASS1-4FGLDR2'] == "" or df['CLASS1-4FGLDR2'] is None \
            or df['CLASS1-4FGLDR3'] == "" or df['CLASS1-4FGLDR3'] is None \
            or df['CLASS1-4FGLDR4'] == "" or df['CLASS1-4FGLDR4'] is None:
        return 1
    elif ((df['CLASS1-3FGL'] == df['CLASS1-4FGLDR1']) \
            and (df['CLASS1-3FGL'] == df['CLASS1-4FGLDR2']) \
            and (df['CLASS1-3FGL'] == df['CLASS1-4FGLDR3']) \
            and (df['CLASS1-3FGL'] == df['CLASS1-4FGLDR4'])):
        return 2
    elif ((df['CLASS1-3FGL'] != "") \
          and (df['CLASS1-4FGLDR1'] != "") \
          and (df['CLASS1-4FGLDR2'] != "") \
          and (df['CLASS1-4FGLDR3'] != "") \
          and (df['CLASS1-4FGLDR4'] != "")):
        return 3
    else:
        return 4

print("=========================Merge 3FGL, 4FGL-DR1, 4FGL-DR2, 4FGL-DR3, 4FGL-DR4=========================")
joined_3FGL_4FGLDR1_DR2_DR3_DR4 = pd.merge(joined_3FGL_4FGLDR1_DR2_DR3, fgl4dr4_df, left_on='3FGL', right_on='4FGLDR4-3FGL', how='left')
joined_3FGL_4FGLDR1_DR2_DR3_DR4['status'] = joined_3FGL_4FGLDR1_DR2_DR3_DR4.apply(flag_df, axis = 1)
joined_3FGL_4FGLDR1_DR2_DR3_DR4.to_csv('../../CSV/joined_3FGL_4FGLDR1_DR2_DR3_DR4.csv')
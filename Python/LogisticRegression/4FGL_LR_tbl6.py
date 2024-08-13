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
import numpy as np
import pandas as pd
import torch.nn as nn
from statistics import *
from astropy.io import fits
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from Utils.FGL3Dataset_Tbl6 import FGL3Dataset_Tbl6
from Utils.FGL4Dataset_Tbl6 import FGL4Dataset_Tbl6
from Utils.FGL4Dataset_Processing import FGL4Dataset_Process
from Utils.utils import (read_fits_file, read_csv_file, lr_train,
                         predict, nn_train, dnn_train)
from RandomForests.random_forest import rf_fit


wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "msp", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "AGN", "agn", "ssrq", "sey"]
wantedfeaturenames_3fgl = ["Spectral_Index", "Variability_Index", "Flux_Density", "Unc_Energy_Flux100",
                           "Signif_Curve"]
fluxlevels_3fgl = ["Flux100_300", "Flux300_1000", "Flux1000_3000", "Flux3000_10000", "Flux10000_100000"]
wantedfeaturenames_4fgldr1_2 = ["PL_Index", "Variability_Index", "Unc_PL_Flux_Density", "Unc_Energy_Flux100",
                                "Signif_Avg"]
# wantedfeaturenames_4fgldr1_2 = ["PL_Index", "LP_Index", "PLEC_Index", "Variability_Index", "Unc_PL_Flux_Density",
#                                 "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",
#                                 "Unc_Flux1000", "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
wantedfeaturenames_4fgldr3_4 = ["PL_Index", "LP_Index", "PLEC_IndexS", "Variability_Index", "Unc_PL_Flux_Density",
                                "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",
                                "Unc_Flux1000", "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
flux_col = "Flux_Band"
wanted_type_col = 69 # 64 # 74
sc = StandardScaler()

print("########################### Read 3FGL ###########################")
wanted_type_col = 73
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v16.fit")
pointsourcecatalogue_3fgl = read_fits_file(path)
fgl3_dataset = FGL3Dataset_Tbl6(wanted_type_col, fluxlevels_3fgl, wantedtypes,
                                wantedfeaturenames_3fgl, pointsourcecatalogue_3fgl)
fgl3_x, fgl3_y, fgl3_z = fgl3_dataset.get_xyz()
fgl3_data = zip(fgl3_z, fgl3_x, fgl3_y)
fgl3_df = pd.DataFrame(fgl3_data, columns = ["3fgl",  "3fgl_features", "3fgl_labels"])
fgl3_df.to_csv('../../CSV/fgl3_df_cls1_null.csv')
print(fgl3_df)

print("########################### Read Table6 ###########################")
tbl6_df = read_csv_file("../../CSV/tbl6.txt",
                        ['3fgl','signif','ra','decl','lr_p','rf_p','blr'], )
tbl6_src_df = tbl6_df[['3fgl']]
tbl6_df.to_csv('../../CSV/tbl6_df.csv')
print(tbl6_df)

print("########################### Merge Table6 & 3FGL ###########################")
tbl6_fgl3 = pd.merge(tbl6_src_df, fgl3_df, left_on='3fgl', right_on='3fgl', how='left')
print(tbl6_fgl3)
tbl6_fgl3.to_csv('../../CSV/tbl6_fgl3.csv')

print("########################### Read 4FGL-DR1 ###########################")
wanted_type_col = 74 # class1
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v22_4FGL_DR1.fit")
pointsourcecatalogue_dr1 = read_fits_file(path)
dr1_dataset = FGL4Dataset_Process(pointsourcecatalogue_dr1, wanted_type_col, [67])
dr1_lst = dr1_dataset.get_all_list()
dr1_df = pd.DataFrame(dr1_lst, columns = ["4fgldr1",  "4fgldr1-cls1", "4fgldr1-3fgl"])
print(dr1_df)

print("########################### Read 4FGL-DR2 ###########################")
wanted_type_col = 64 # class1
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v27_4FGL_DR2.fit")
pointsourcecatalogue_dr2 = read_fits_file(path)
dr2_dataset = FGL4Dataset_Process(pointsourcecatalogue_dr2, wanted_type_col, [57])
dr2_lst = dr2_dataset.get_all_list()
dr2_df = pd.DataFrame(dr2_lst, columns = ["4fgldr2", "4fgldr2-cls1", "4fgldr2-3fgl"])
print(dr2_df)

print("########################### Read 4FGL-DR3 ###########################")
wanted_type_col = 69 # class1
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v31_4FGL_DR3.fit")
pointsourcecatalogue_dr3 = read_fits_file(path)
dr3_dataset = FGL4Dataset_Process(pointsourcecatalogue_dr3, wanted_type_col, [62])
dr3_lst = dr3_dataset.get_all_list()
dr3_df = pd.DataFrame(dr3_lst, columns = ["4fgldr3",  "4fgldr3-cls1", "4fgldr3-3fgl"])
print(dr3_df)

print("########################### Read 4FGL-DR4 ###########################")
wanted_type_col = 69 # class1
path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33_4FGL_DR4.fit")
pointsourcecatalogue_dr4 = read_fits_file(path)
dr4_dataset = FGL4Dataset_Process(pointsourcecatalogue_dr4, wanted_type_col, [62])
dr4_lst = dr4_dataset.get_all_list()
dr4_df = pd.DataFrame(dr4_lst, columns = ["4fgldr4", "4fgldr4-cls1", "4fgldr4-3fgl"])
print(dr4_df)

print("========================= Merge Tbl6 & dr1 =========================")
tbl6_dr1 = pd.merge(tbl6_df, dr1_df, left_on='3fgl', right_on='4fgldr1-3fgl', how='left')
tbl6_src_dr1 = pd.merge(tbl6_src_df, dr1_df, left_on='3fgl', right_on='4fgldr1-3fgl', how='left')
print(tbl6_dr1)
tbl6_dr1.to_csv('../../CSV/tbl6_dr1.csv')

print("=========================Merge tbl6_dr1 & dr2 =========================")
tbl6_dr1_2 = pd.merge(tbl6_dr1, dr2_df, left_on='3fgl', right_on='4fgldr2-3fgl', how='left')
tbl6_src_dr1_2 = pd.merge(tbl6_src_dr1, dr2_df, left_on='3fgl', right_on='4fgldr2-3fgl', how='left')
print(tbl6_dr1_2)
tbl6_dr1_2.to_csv('../../CSV/tbl6_dr1_2.csv')

print("=========================Merge tbl6_dr1_dr2 & dr3 =========================")
tbl6_dr1_2_3 = pd.merge(tbl6_dr1_2, dr3_df, left_on='3fgl', right_on='4fgldr3-3fgl', how='left')
tbl6_src_dr1_2_3 = pd.merge(tbl6_src_dr1_2, dr3_df, left_on='3fgl', right_on='4fgldr3-3fgl', how='left')
print(tbl6_dr1_2_3)

print("=========================Merge tbl6_dr1_dr2_dr3 & dr4 =========================")
tbl6_dr1_2_3_4 = pd.merge(tbl6_dr1_2_3, dr4_df, left_on='3fgl', right_on='4fgldr4-3fgl', how='left')
tbl6_src_dr1_2_3_4 = pd.merge(tbl6_src_dr1_2_3, dr4_df, left_on='3fgl', right_on='4fgldr4-3fgl', how='left')
print(tbl6_dr1_2_3_4)
tbl6_dr1_2_3_4.to_csv('../../CSV/tbl6_dr1_2_3_4.csv')
tbl6_src_dr1_2_3_4.to_csv('../../CSV/tbl6_src_dr1_2_3_4_0511.csv')

iter_no = 100
trees = 100
samples = 750
depth = 10
print("########################### Predict 4FGL-DR1 ###########################")
wanted_type_col = 74  # class1
dr1_dataset = FGL4Dataset_Tbl6(wanted_type_col, flux_col, wantedtypes,
                               wantedfeaturenames_4fgldr1_2, pointsourcecatalogue_dr1)
LogRegModel = lr_train(dr1_dataset, iter_no)
predict_feature_list = tbl6_fgl3['3fgl_features'].tolist()
predict_features_tensor = [torch.from_numpy(np.array(i)) for i in predict_feature_list]
tbl6_fgl3['predict_lr_dr1'] = [predict(LogRegModel(i)) for i in predict_features_tensor]

NNModel = nn_train(dr1_dataset, iter_no)
tbl6_fgl3['predict_nn_dr1'] = [predict(NNModel(i)) for i in predict_features_tensor]

DNNModel = dnn_train(dr1_dataset, iter_no)
tbl6_fgl3['predict_dnn_dr1'] = [predict(DNNModel(i)) for i in predict_features_tensor]
tbl6_fgl3_lean = tbl6_fgl3[['3fgl','predict_lr_dr1','predict_nn_dr1','predict_dnn_dr1']]

RFModel = rf_fit(dr1_dataset,trees,samples,depth)
tbl6_fgl3['predict_rf_dr1'] = [RFModel.predict_tbl6(torch.FloatTensor(i.numpy())) for i in predict_features_tensor]
tbl6_fgl3_lean = tbl6_fgl3[['3fgl','predict_lr_dr1','predict_nn_dr1','predict_dnn_dr1','predict_rf_dr1']]

print("########################### Predict 4FGL-DR2 ###########################")
wanted_type_col = 64  # class1
dr2_dataset = FGL4Dataset_Tbl6(wanted_type_col, flux_col, wantedtypes,
                               wantedfeaturenames_4fgldr1_2, pointsourcecatalogue_dr2)
LogRegModel = lr_train(dr2_dataset, iter_no)
tbl6_fgl3['predict_lr_dr2'] = [predict(LogRegModel(i)) for i in predict_features_tensor]

NNModel = nn_train(dr2_dataset, iter_no)
tbl6_fgl3['predict_nn_dr2'] = [predict(NNModel(i)) for i in predict_features_tensor]

DNNModel = dnn_train(dr2_dataset, iter_no)
tbl6_fgl3['predict_dnn_dr2'] = [predict(DNNModel(i)) for i in predict_features_tensor]

RFModel = rf_fit(dr2_dataset,trees,samples,depth)
tbl6_fgl3['predict_rf_dr2'] = [RFModel.predict_tbl6(torch.FloatTensor(i.numpy())) for i in predict_features_tensor]
tbl6_fgl3_lean = tbl6_fgl3[['3fgl','predict_lr_dr1','predict_nn_dr1','predict_dnn_dr1','predict_rf_dr1',
                            'predict_lr_dr2','predict_nn_dr2','predict_dnn_dr2','predict_rf_dr2']]

print("########################### Predict 4FGL-DR3 ###########################")
wanted_type_col = 69  # class1
dr3_dataset = FGL4Dataset_Tbl6(wanted_type_col, flux_col, wantedtypes,
                               wantedfeaturenames_4fgldr1_2, pointsourcecatalogue_dr3)
LogRegModel = lr_train(dr3_dataset, iter_no)
tbl6_fgl3['predict_lr_dr3'] = [predict(LogRegModel(i)) for i in predict_features_tensor]

NNModel = nn_train(dr3_dataset, iter_no)
tbl6_fgl3['predict_nn_dr3'] = [predict(NNModel(i)) for i in predict_features_tensor]

DNNModel = dnn_train(dr3_dataset, iter_no)
tbl6_fgl3['predict_dnn_dr3'] = [predict(DNNModel(i)) for i in predict_features_tensor]

RFModel = rf_fit(dr3_dataset,trees,samples,depth)
tbl6_fgl3['predict_rf_dr3'] = [RFModel.predict_tbl6(torch.FloatTensor(i.numpy())) for i in predict_features_tensor]
tbl6_fgl3_lean = tbl6_fgl3[['3fgl','predict_lr_dr1','predict_nn_dr1','predict_dnn_dr1','predict_rf_dr1',
                            'predict_lr_dr2','predict_nn_dr2','predict_dnn_dr2','predict_rf_dr2',
                            'predict_lr_dr3','predict_nn_dr3','predict_dnn_dr3','predict_rf_dr3']]

print("########################### Predict 4FGL-DR4 ###########################")
wanted_type_col = 69  # class1
dr4_dataset = FGL4Dataset_Tbl6(wanted_type_col, flux_col, wantedtypes,
                               wantedfeaturenames_4fgldr1_2, pointsourcecatalogue_dr4)
LogRegModel = lr_train(dr4_dataset, iter_no)
tbl6_fgl3['predict_lr_dr4'] = [predict(LogRegModel(i)) for i in predict_features_tensor]
tbl6_fgl3.to_csv('../../CSV/tbl6_fgl3.csv')

NNModel = nn_train(dr4_dataset, iter_no)
tbl6_fgl3['predict_nn_dr4'] = [predict(NNModel(i)) for i in predict_features_tensor]

DNNModel = dnn_train(dr4_dataset, iter_no)
tbl6_fgl3['predict_dnn_dr4'] = [predict(DNNModel(i)) for i in predict_features_tensor]

RFModel = rf_fit(dr4_dataset,trees,samples,depth)
tbl6_fgl3['predict_rf_dr4'] = [RFModel.predict_tbl6(torch.FloatTensor(i.numpy())) for i in predict_features_tensor]
tbl6_fgl3_lean = tbl6_fgl3[['3fgl','predict_lr_dr1','predict_nn_dr1','predict_dnn_dr1','predict_rf_dr1',
                            'predict_lr_dr2','predict_nn_dr2','predict_dnn_dr2','predict_rf_dr2',
                            'predict_lr_dr3','predict_nn_dr3','predict_dnn_dr3','predict_rf_dr3',
                            'predict_lr_dr4','predict_nn_dr4','predict_dnn_dr4','predict_rf_dr4']]

tbl6_dr1_2_3_4_predict = pd.merge(tbl6_dr1_2_3_4, tbl6_fgl3_lean, left_on='3fgl',
                                  right_on='3fgl', how='left')
tbl6_dr1_2_3_4_predict = tbl6_dr1_2_3_4_predict[['3fgl','signif','ra','decl','lr_p','rf_p','blr',
                            '4fgldr1','4fgldr1-cls1','predict_lr_dr1','predict_nn_dr1','predict_dnn_dr1','predict_rf_dr1',
                            '4fgldr2','4fgldr2-cls1','predict_lr_dr2','predict_nn_dr2','predict_dnn_dr2','predict_rf_dr2',
                            '4fgldr3','4fgldr3-cls1','predict_lr_dr3','predict_nn_dr3','predict_dnn_dr3','predict_rf_dr3',
                            '4fgldr4','4fgldr4-cls1','predict_lr_dr4','predict_nn_dr4','predict_dnn_dr4','predict_rf_dr4']]
tbl6_dr1_2_3_4_predict.to_csv('../../CSV/tbl6_dr1_2_3_4_predict_0511.csv')
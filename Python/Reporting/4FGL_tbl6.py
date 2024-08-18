#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:15:48 2023

@author: pablo
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Utils.FGL4Dataset_Tbl6 import FGL4Dataset_Tbl6
from Utils.FGL4Dataset_Processing import FGL4Dataset_Process
from Utils.utils import (read_fits_file, read_csv_file, lr_train, predict, nn_train, dnn_train)
from RandomForests.random_forest import rf_fit

csv_suffix = '_iter100_'
wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "msp", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "AGN", "agn", "ssrq", "sey"]
wantedfeaturenames_3fgl = ["Spectral_Index", "Variability_Index", "Flux_Density", "Unc_Energy_Flux100",
                           "Signif_Curve"]
fluxlevels_3fgl = ["Flux100_300", "Flux300_1000", "Flux1000_3000", "Flux3000_10000", "Flux10000_100000"]
wantedfeaturenames_4fgldr1_2 = ["PL_Index", "LP_Index", "PLEC_Index", "Variability_Index", "Unc_PL_Flux_Density",
                                "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",
                                "Unc_Flux1000", "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
wantedfeaturenames_4fgldr3_4 = ["PL_Index", "LP_Index", "PLEC_IndexS", "Variability_Index", "Unc_PL_Flux_Density",
                                "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",
                                "Unc_Flux1000", "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
flux_col = "Flux_Band"
wanted_type_col = 69 # 64 # 74
sc = StandardScaler()

print("########################### Read Golden Dataset ###########################")
golden_df = read_csv_file("../../CSV/golden_dataset.csv",
                        ['3fgl','3fgl-cls1','4fgldr1','4fgldr1-cls1','4fgldr2',
                         '4fgldr2-cls1','4fgldr3','4fgldr3-cls1','4fgldr4','4fgldr4-cls1'],
                          ',')
golden_df.to_csv('../../CSV/golden_df'+csv_suffix+'.csv')

print("########################### Read Table6 ###########################")
tbl6_df = read_csv_file("../../CSV/tbl6.txt",
                        ['3fgl','signif','ra','decl','lr_p','rf_p','blr'],
                        ' ')
tbl6_df.to_csv('../../CSV/tbl6_df'+csv_suffix+'.csv')
print(tbl6_df)

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
print(tbl6_dr1)
tbl6_dr1.to_csv('../../CSV/tbl6_dr1'+csv_suffix+'.csv')

print("=========================Merge tbl6_dr1 & dr2 =========================")
tbl6_dr1_2 = pd.merge(tbl6_dr1, dr2_df, left_on='3fgl', right_on='4fgldr2-3fgl', how='left')
print(tbl6_dr1_2)
tbl6_dr1_2.to_csv('../../CSV/tbl6_dr1_2'+csv_suffix+'.csv')

print("=========================Merge tbl6_dr1_dr2 & dr3 =========================")
tbl6_dr1_2_3 = pd.merge(tbl6_dr1_2, dr3_df, left_on='3fgl', right_on='4fgldr3-3fgl', how='left')
print(tbl6_dr1_2_3)

print("=========================Merge tbl6_dr1_dr2_dr3 & dr4 =========================")
tbl6_dr1_2_3_4 = pd.merge(tbl6_dr1_2_3, dr4_df, left_on='3fgl', right_on='4fgldr4-3fgl', how='left')
print(tbl6_dr1_2_3_4)
tbl6_dr1_2_3_4.to_csv('../../CSV/tbl6_dr1_2_3_4'+csv_suffix+'.csv')

iter_no = 100
trees = 100
samples = 750
depth = 10
print("########################### Predict 4FGL-DR1 ###########################")
wanted_type_col = 74  # class1
dr1_dataset = FGL4Dataset_Tbl6(wanted_type_col, flux_col, wantedtypes, wantedfeaturenames_4fgldr1_2,
                               pointsourcecatalogue_dr1, golden_df, "4fgldr1")
dr1_x, dr1_y, dr1_z = dr1_dataset.get_xyz()
dr1_data = zip(dr1_z, dr1_x, dr1_y)
dr1_feat_lbl = pd.DataFrame(dr1_data, columns = ["4fgldr1",  "4fgldr1_features", "4fgldr1_labels"])
tbl6_dr1_feat_lbl = pd.merge(tbl6_dr1_2_3_4, dr1_feat_lbl, left_on='4fgldr1', right_on='4fgldr1', how='left')

predict_feature_list = tbl6_dr1_feat_lbl['4fgldr1_features'].tolist()
predict_features_tensor = [torch.from_numpy(np.array(i)) for i in predict_feature_list]

LogRegModel = lr_train(dr1_dataset, iter_no)
tbl6_dr1_feat_lbl['predict_lr_dr1'] = [predict(LogRegModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
NNModel = nn_train(dr1_dataset, iter_no)
tbl6_dr1_feat_lbl['predict_nn_dr1'] = [predict(NNModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
DNNModel = dnn_train(dr1_dataset, iter_no)
tbl6_dr1_feat_lbl['predict_dnn_dr1'] = [predict(DNNModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
RFModel = rf_fit(dr1_dataset,trees,samples,depth)
tbl6_dr1_feat_lbl['predict_rf_dr1'] = [RFModel.predict_tbl6(torch.FloatTensor(i.numpy())) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
tbl6_dr1_feat_lbl.to_csv('../../CSV/tbl6_dr1_feat_lbl'+csv_suffix+'.csv')
tbl6_dr1_feat_lbl_uniq = tbl6_dr1_feat_lbl.drop_duplicates(subset=['3fgl','4fgldr1','4fgldr1-cls1','4fgldr1-3fgl'])
tbl6_dr1_feat_lbl_uniq.to_csv('../../CSV/tbl6_dr1_feat_lbl_uniq'+csv_suffix+'.csv')

print("########################### Predict 4FGL-DR2 ###########################")
wanted_type_col = 64  # class1
dr2_dataset = FGL4Dataset_Tbl6(wanted_type_col, flux_col, wantedtypes, wantedfeaturenames_4fgldr1_2,
                               pointsourcecatalogue_dr2, golden_df, "4fgldr2")
dr2_x, dr2_y, dr2_z = dr2_dataset.get_xyz()
dr2_data = zip(dr2_z, dr2_x, dr2_y)
dr2_feat_lbl = pd.DataFrame(dr2_data, columns = ["4fgldr2",  "4fgldr2_features", "4fgldr2_labels"])
tbl6_dr2_feat_lbl = pd.merge(tbl6_dr1_2_3_4, dr2_feat_lbl, left_on='4fgldr2', right_on='4fgldr2', how='left')

predict_feature_list = tbl6_dr2_feat_lbl['4fgldr2_features'].tolist()
predict_features_tensor = [torch.from_numpy(np.array(i)) for i in predict_feature_list]

LogRegModel = lr_train(dr2_dataset, iter_no)
tbl6_dr2_feat_lbl['predict_lr_dr2'] = [predict(LogRegModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
NNModel = nn_train(dr2_dataset, iter_no)
tbl6_dr2_feat_lbl['predict_nn_dr2'] = [predict(NNModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
DNNModel = dnn_train(dr2_dataset, iter_no)
tbl6_dr2_feat_lbl['predict_dnn_dr2'] = [predict(DNNModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
RFModel = rf_fit(dr2_dataset,trees,samples,depth)
tbl6_dr2_feat_lbl['predict_rf_dr2'] = [RFModel.predict_tbl6(torch.FloatTensor(i.numpy())) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
tbl6_dr2_feat_lbl.to_csv('../../CSV/tbl6_dr2_feat_lbl'+csv_suffix+'.csv')
tbl6_dr2_feat_lbl_uniq = tbl6_dr2_feat_lbl.drop_duplicates(subset=['3fgl','4fgldr2','4fgldr2-cls1','4fgldr2-3fgl'])
tbl6_dr2_feat_lbl_uniq.to_csv('../../CSV/tbl6_dr2_feat_lbl_uniq'+csv_suffix+'.csv')

print("########################### Predict 4FGL-DR3 ###########################")
wanted_type_col = 69  # class1
dr3_dataset = FGL4Dataset_Tbl6(wanted_type_col, flux_col, wantedtypes, wantedfeaturenames_4fgldr3_4,
                               pointsourcecatalogue_dr3, golden_df, "4fgldr3")
dr3_x, dr3_y, dr3_z = dr3_dataset.get_xyz()
dr3_data = zip(dr3_z, dr3_x, dr3_y)
dr3_feat_lbl = pd.DataFrame(dr3_data, columns = ["4fgldr3",  "4fgldr3_features", "4fgldr3_labels"])
tbl6_dr3_feat_lbl = pd.merge(tbl6_dr1_2_3_4, dr3_feat_lbl, left_on='4fgldr3', right_on='4fgldr3', how='left')

predict_feature_list = tbl6_dr3_feat_lbl['4fgldr3_features'].tolist()
predict_features_tensor = [torch.from_numpy(np.array(i)) for i in predict_feature_list]

LogRegModel = lr_train(dr3_dataset, iter_no)
tbl6_dr3_feat_lbl['predict_lr_dr3'] = [predict(LogRegModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
NNModel = nn_train(dr3_dataset, iter_no)
tbl6_dr3_feat_lbl['predict_nn_dr3'] = [predict(NNModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
DNNModel = dnn_train(dr3_dataset, iter_no)
tbl6_dr3_feat_lbl['predict_dnn_dr3'] = [predict(DNNModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
RFModel = rf_fit(dr3_dataset,trees,samples,depth)
tbl6_dr3_feat_lbl['predict_rf_dr3'] = [RFModel.predict_tbl6(torch.FloatTensor(i.numpy())) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
tbl6_dr3_feat_lbl.to_csv('../../CSV/tbl6_dr3_feat_lbl'+csv_suffix+'.csv')
tbl6_dr3_feat_lbl_uniq = tbl6_dr3_feat_lbl.drop_duplicates(subset=['3fgl','4fgldr3','4fgldr3-cls1','4fgldr3-3fgl'])
tbl6_dr3_feat_lbl_uniq.to_csv('../../CSV/tbl6_dr3_feat_lbl_uniq'+csv_suffix+'.csv')

print("########################### Predict 4FGL-DR4 ###########################")
wanted_type_col = 69  # class1
dr4_dataset = FGL4Dataset_Tbl6(wanted_type_col, flux_col, wantedtypes, wantedfeaturenames_4fgldr3_4,
                               pointsourcecatalogue_dr4, golden_df, "4fgldr4")
dr4_x, dr4_y, dr4_z = dr4_dataset.get_xyz()
dr4_data = zip(dr4_z, dr4_x, dr4_y)
dr4_feat_lbl = pd.DataFrame(dr4_data, columns = ["4fgldr4",  "4fgldr4_features", "4fgldr4_labels"])
tbl6_dr4_feat_lbl = pd.merge(tbl6_dr1_2_3_4, dr4_feat_lbl, left_on='4fgldr4', right_on='4fgldr4', how='left')

predict_feature_list = tbl6_dr4_feat_lbl['4fgldr4_features'].tolist()
predict_features_tensor = [torch.from_numpy(np.array(i)) for i in predict_feature_list]

LogRegModel = lr_train(dr4_dataset, iter_no)
tbl6_dr4_feat_lbl['predict_lr_dr4'] = [predict(LogRegModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
NNModel = nn_train(dr4_dataset, iter_no)
tbl6_dr4_feat_lbl['predict_nn_dr4'] = [predict(NNModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
DNNModel = dnn_train(dr4_dataset, iter_no)
tbl6_dr4_feat_lbl['predict_dnn_dr4'] = [predict(DNNModel(i)) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
RFModel = rf_fit(dr4_dataset,trees,samples,depth)
tbl6_dr4_feat_lbl['predict_rf_dr4'] = [RFModel.predict_tbl6(torch.FloatTensor(i.numpy())) if (len(list(i.shape)) != 0) else 'NA' for i in predict_features_tensor]
tbl6_dr4_feat_lbl.to_csv('../../CSV/tbl6_dr4_feat_lbl'+csv_suffix+'.csv')
tbl6_dr4_feat_lbl_uniq = tbl6_dr4_feat_lbl.drop_duplicates(subset=['3fgl','4fgldr4','4fgldr4-cls1','4fgldr4-3fgl'])
tbl6_dr4_feat_lbl_uniq.to_csv('../../CSV/tbl6_dr4_feat_lbl_uniq'+csv_suffix+'.csv')

print("################################ Consolidate predicted datasets ##################################")
tbl6_dr1_predict = tbl6_dr1_feat_lbl_uniq[['3fgl','4fgldr1','4fgldr1-cls1','predict_lr_dr1','predict_nn_dr1','predict_dnn_dr1','predict_rf_dr1']]
tbl6_dr2_predict = tbl6_dr2_feat_lbl_uniq[['3fgl','4fgldr2','4fgldr2-cls1','predict_lr_dr2','predict_nn_dr2','predict_dnn_dr2','predict_rf_dr2']]
tbl6_dr3_predict = tbl6_dr3_feat_lbl_uniq[['3fgl','4fgldr3','4fgldr3-cls1','predict_lr_dr3','predict_nn_dr3','predict_dnn_dr3','predict_rf_dr3']]
tbl6_dr4_predict = tbl6_dr4_feat_lbl_uniq[['3fgl','4fgldr4','4fgldr4-cls1','predict_lr_dr4','predict_nn_dr4','predict_dnn_dr4','predict_rf_dr4']]

tbl6all_dr1_predict = pd.merge(tbl6_df, tbl6_dr1_predict, left_on='3fgl', right_on='3fgl', how='left')
tbl6all_dr1_predict.to_csv('../../CSV/tbl6all_dr1_predict'+csv_suffix+'.csv')

tbl6all_dr12_predict = pd.merge(tbl6all_dr1_predict, tbl6_dr2_predict, left_on='3fgl', right_on='3fgl', how='left')
tbl6all_dr12_predict.to_csv('../../CSV/tbl6all_dr12_predict'+csv_suffix+'.csv')

tbl6all_dr123_predict = pd.merge(tbl6all_dr12_predict, tbl6_dr3_predict, left_on='3fgl', right_on='3fgl', how='left')
tbl6all_dr123_predict.to_csv('../../CSV/tbl6all_dr123_predict'+csv_suffix+'.csv')

tbl6all_dr1234_predict = pd.merge(tbl6all_dr123_predict, tbl6_dr4_predict, left_on='3fgl', right_on='3fgl', how='left')
tbl6all_dr1234_predict.to_csv('../../CSV/tbl6all_dr1234_predict'+csv_suffix+'.csv')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:15:48 2023

@author: pablo
"""

import os
from astropy.io import fits
import matplotlib.pyplot as plt
from Utils.FGL4Dataset_Tbl1 import FGL4Dataset_Tbl1


path = os.path.join(os.getcwd(), "../../FITS/gll_psc_v33_4FGL_DR4.fit")
mainfile = fits.open(path)
point_source_catalogue = mainfile[1]
columns = 0
keymap = {}
mapkey = {}
for key in list(point_source_catalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = point_source_catalogue.header[key]
        mapkey[point_source_catalogue.header[key]] = int(key[5:len(key)])-1
        print(columns, key, point_source_catalogue.header[key])
        columns += 1

print("########################### Predict 4FGL-DR4 ###########################")
wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "msp", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg",
               "NLSY1", "nlsy1", "AGN", "agn", "ssrq", "sey"]
wantedfeaturenames_4fgldr3_4 = ["PL_Index", "LP_Index", "PLEC_IndexS", "Variability_Index", "Unc_PL_Flux_Density",
                                "Unc_LP_Flux_Density", "Unc_Energy_Flux100", "Unc_PLEC_Flux_Density",
                                "Unc_Flux1000", "LP_beta", "Frac_Variability", "LP_SigCurv", "Signif_Avg"]
flux_col = "Flux_Band"
wanted_type_col = 69  # class1
dr4_dataset = FGL4Dataset_Tbl1(wanted_type_col, flux_col, wantedtypes, wantedfeaturenames_4fgldr3_4,
                               point_source_catalogue)
signif_index,psr_signif,agn_signif,unassociated_signf = dr4_dataset.get_signif()
psr_fraction = [x/sum(psr_signif) for x in psr_signif]
agn_fraction = [x/sum(agn_signif) for x in agn_signif]
unassociated_fraction = [x/sum(unassociated_signf) for x in unassociated_signf]
print(psr_signif)
print(agn_signif)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(signif_index, psr_fraction, label='PSR', linestyle='solid')
plt.plot(signif_index, agn_fraction, label='AGN', linestyle='dashed')
plt.plot(signif_index, unassociated_fraction, label='Unassociated', linestyle='dotted')
ax.set_xlabel('Significance')
ax.set_ylabel('Fraction of Sources')
plt.legend(loc='upper right')
plt.title('Fraction of 4FGL-DR4 Sources as a function of Signif-Avg')
plt.show()
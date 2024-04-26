import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class FGL4Dataset_Reporting(Dataset):
    def __init__(self, point_source_catalogue, wanted_type_col, all_types, psr_types, agn_types):
        # Create x/input data
        # Select sources that are a part of the wanted types
        all_cnt = null_cnt = agn_cnt = AGN_cnt = psr_cnt = PSR_cnt = 0
        msp_cnt = MSP_cnt = fsrq_cnt = FSRQ_cnt = bll_cnt = BLL_cnt = bcu_cnt = BCU_cnt = 0
        RDG_cnt = rdg_cnt = NLSY1_cnt = nlsy1_cnt = YNG_cnt = yng_cnt = ssrq_cnt = sey_cnt = 0
        unk_cnt = bin_cnt = glc_cnt = sbg_cnt = clust_cnt = spp_cnt = snr_cnt = SNR_cnt = 0
        gal_cnt = HMB_cnt = pwn_cnt = GAL_cnt = PWN_cnt = BIN_cnt = css_cnt = SFR_cnt = NOV_cnt = 0
        bzb_cnt = bzq_cnt = agu_cnt = BZQ_cnt = BZB_cnt = SEY_cnt = HXB_cnt = MQO_cnt = 0
        glb_cnt = hxb_cnt = bzu_cnt = 0
        for source in point_source_catalogue.data:
            # for feature_names in wanted_feature_names:
            #     print("{0} = {1}".format(feature_names, source[feature_names]))
            # print("source[flux_col] = {0}".format(flux_list))
            # print("source[wanted_type_col] = ", source[wanted_type_col])
            # if source[wanted_type_col] in all_types:
            if 1 == 1:
                all_cnt += 1
                if source[wanted_type_col] == '' or source[wanted_type_col] is None:
                    null_cnt += 1
                elif source[wanted_type_col] == 'AGN':
                    AGN_cnt += 1
                elif source[wanted_type_col] == 'agn':
                    agn_cnt += 1
                elif source[wanted_type_col] == 'FSRQ':
                    FSRQ_cnt += 1
                elif source[wanted_type_col] == 'fsrq':
                    fsrq_cnt += 1
                elif source[wanted_type_col] == 'BLL':
                    BLL_cnt += 1
                elif source[wanted_type_col] == 'bll':
                    bll_cnt += 1
                elif source[wanted_type_col] == 'BCU':
                    BCU_cnt += 1
                elif source[wanted_type_col] == 'bcu':
                    bll_cnt += 1
                elif source[wanted_type_col] == 'RDG':
                    RDG_cnt += 1
                elif source[wanted_type_col] == 'rdg':
                    rdg_cnt += 1
                elif source[wanted_type_col] == 'NLSY1':
                    NLSY1_cnt += 1
                elif source[wanted_type_col] == 'nlsy1':
                    nlsy1_cnt += 1
                elif source[wanted_type_col] == 'YNG':
                    YNG_cnt += 1
                elif source[wanted_type_col] == 'yng':
                    yng_cnt += 1
                elif source[wanted_type_col] == 'ssrq':
                    ssrq_cnt += 1
                elif source[wanted_type_col] == 'sey':
                    sey_cnt += 1
                # ["PSR", "psr", "MSP", "msp"]
                elif source[wanted_type_col] == 'PSR':
                    PSR_cnt += 1
                elif source[wanted_type_col] == 'psr':
                    psr_cnt += 1
                elif source[wanted_type_col] == 'MSP':
                    MSP_cnt += 1
                elif source[wanted_type_col] == 'msp':
                    msp_cnt += 1
                elif source[wanted_type_col] == 'unk':
                    unk_cnt += 1
                elif source[wanted_type_col] == 'bin':
                    bin_cnt += 1
                elif source[wanted_type_col] == 'glc':
                    glc_cnt += 1
                elif source[wanted_type_col] == 'sbg':
                    sbg_cnt += 1
                elif source[wanted_type_col] == 'clust':
                    clust_cnt += 1
                elif source[wanted_type_col] == 'spp':
                    spp_cnt += 1
                elif source[wanted_type_col] == 'snr':
                    snr_cnt += 1
                elif source[wanted_type_col] == 'SNR':
                    SNR_cnt += 1
                elif source[wanted_type_col] == 'gal':
                    gal_cnt += 1
                elif source[wanted_type_col] == 'GAL':
                    GAL_cnt += 1
                elif source[wanted_type_col] == 'HMB':
                    HMB_cnt += 1
                elif source[wanted_type_col] == 'pwn':
                    pwn_cnt += 1
                elif source[wanted_type_col] == 'PWN':
                    PWN_cnt += 1
                elif source[wanted_type_col] == 'BIN':
                    BIN_cnt += 1
                elif source[wanted_type_col] == 'css':
                    css_cnt += 1
                elif source[wanted_type_col] == 'SFR':
                    SFR_cnt += 1
                elif source[wanted_type_col] == 'NOV':
                    NOV_cnt += 1
                elif source[wanted_type_col] == 'bzb':
                    bzb_cnt += 1
                elif source[wanted_type_col] == 'bzq':
                    bzq_cnt += 1
                elif source[wanted_type_col] == 'agu':
                    agu_cnt += 1
                elif source[wanted_type_col] == 'BZQ':
                    BZQ_cnt += 1
                elif source[wanted_type_col] == 'BZB':
                    BZB_cnt += 1
                elif source[wanted_type_col] == 'SEY':
                    SEY_cnt += 1
                elif source[wanted_type_col] == 'HXB':
                    HXB_cnt += 1
                elif source[wanted_type_col] == 'MQO':
                    MQO_cnt += 1
                elif source[wanted_type_col] == 'hxb':
                    hxb_cnt += 1
                elif source[wanted_type_col] == 'bzu':
                    bzu_cnt += 1
                elif source[wanted_type_col] == 'glb':
                    glb_cnt += 1
                else:
                    print("missing = ", source[wanted_type_col])
        print("all_cnt = ", all_cnt)
        print("PSR_cnt = ", PSR_cnt)
        print("psr_cnt = ", psr_cnt)
        print("MSP_cnt = ", MSP_cnt)
        print("msp_cnt = ", msp_cnt)
        print("AGN_cnt = ", AGN_cnt)
        print("agn_cnt = ", agn_cnt)
        print("FSRQ_cnt = ", FSRQ_cnt)
        print("fsrq_cnt = ", fsrq_cnt)
        print("BLL_cnt = ", BLL_cnt)
        print("bll_cnt = ", bll_cnt)
        print("BCU_cnt = ", BCU_cnt)
        print("bcu_cnt = ", bcu_cnt)
        print("RDG_cnt = ", RDG_cnt)
        print("rdg_cnt = ", rdg_cnt)
        print("NLSY1_cnt = ", NLSY1_cnt)
        print("nlsy1_cnt = ", nlsy1_cnt)
        print("YNG_cnt = ", YNG_cnt)
        print("yng_cnt = ", yng_cnt)
        print("ssrq_cnt = ", ssrq_cnt)
        print("sey_cnt = ", sey_cnt)
        print("unk_cnt = ", unk_cnt)
        print("bin_cnt = ", bin_cnt)
        print("null_cnt = ", null_cnt)
        print("glc_cnt = ", glc_cnt)
        print("sbg_cnt = ", sbg_cnt)
        print("clust_cnt = ", clust_cnt)
        print("spp_cnt = ", spp_cnt)
        print("snr_cnt = ", snr_cnt)
        print("SNR_cnt = ", SNR_cnt)
        print("gal_cnt = ", gal_cnt)
        print("GAL_cnt = ", GAL_cnt)
        print("HMB_cnt = ", HMB_cnt)
        print("pwn_cnt = ", pwn_cnt)
        print("PWN_cnt = ", PWN_cnt)
        print("BIN_cnt = ", BIN_cnt)
        print("css_cnt = ", css_cnt)
        print("SFR_cnt = ", SFR_cnt)
        print("NOV_cnt = ", NOV_cnt)
        print("bzb_cnt = ", bzb_cnt)
        print("bzq_cnt = ", bzq_cnt)
        print("agu_cnt = ", agu_cnt)
        print("BZQ_cnt = ", BZQ_cnt)
        print("HXB = ", HXB_cnt)
        print("MQO_cnt = ", MQO_cnt)
        print("glb_cnt = ", glb_cnt)
        print("bzu_cnt = ", bzu_cnt)
        print("hxb_cnt = ", hxb_cnt)

        # create data
        df1 = pd.DataFrame([
            ['AGN', AGN_cnt, agn_cnt, FSRQ_cnt, fsrq_cnt, BLL_cnt, bll_cnt, BCU_cnt, bcu_cnt, RDG_cnt, rdg_cnt, NLSY1_cnt, nlsy1_cnt, YNG_cnt, yng_cnt, ssrq_cnt, sey_cnt]],
            columns=[
                'AGN count', 'AGN', 'agn', 'FSRQ', 'fsrq', 'BLL', 'bll', 'BCU', 'bcu', 'RDG', 'rdg', 'NLSY1', 'nlsy1', 'YNG', 'yng', 'ssrq', 'sey'])

        df2 = pd.DataFrame([
            ['PSR', PSR_cnt, psr_cnt, MSP_cnt, msp_cnt]],
            columns=[
                'PSR count', 'PSR', 'psr', 'MSP', 'msp'])

        df3 = pd.DataFrame([
            ['Total', all_cnt, null_cnt, psr_cnt+msp_cnt+PSR_cnt+MSP_cnt,
             AGN_cnt+agn_cnt+FSRQ_cnt+fsrq_cnt+BLL_cnt+bll_cnt+BCU_cnt+bcu_cnt+RDG_cnt+rdg_cnt+NLSY1_cnt+nlsy1_cnt+YNG_cnt+yng_cnt+ssrq_cnt+sey_cnt,
             unk_cnt, bin_cnt, glc_cnt, sbg_cnt, clust_cnt, spp_cnt, snr_cnt, SNR_cnt, gal_cnt, GAL_cnt, HMB_cnt,
             pwn_cnt, PWN_cnt, BIN_cnt, css_cnt, SFR_cnt, NOV_cnt, bzq_cnt, bzb_cnt, agu_cnt, BZQ_cnt, HXB_cnt, MQO_cnt,
             glb_cnt, bzu_cnt, hxb_cnt,
             ]],
            columns=[
                'Total', 'All', 'Null count', 'PSR', 'AGN', 'UNK', 'BIN', 'GLC', 'SBG', 'CLUST', 'spp', 'snr', 'SNR',
            'gal', 'GAL', 'HMB', 'pwn', 'PWN', 'BIN', 'css', 'SFR', 'NOV', 'bzb', 'bzq', 'agu', 'BZQ', 'HXB', 'MQO',
            'glb', 'bzu', 'hxb'])

        # plot grouped bar chart
        ax1 = df1.plot(
            x='AGN count',
            kind='bar',
            stacked=False,
            title='AGN count')
        for container in ax1.containers:
            ax1.bar_label(container)

        ax2 = df2.plot(
            x='PSR count',
            kind='bar',
            stacked=False,
            title='PSR count')
        for container in ax2.containers:
            ax2.bar_label(container)

        ax3 = df3.plot(
            x='Total',
            kind='bar',
            stacked=False,
            title='All count')
        for container in ax3.containers:
            ax3.bar_label(container)

        plt.show()

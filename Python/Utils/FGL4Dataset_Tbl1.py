import math
import torch
import numpy as np
from torch.utils.data import Dataset


class FGL4Dataset_Tbl1(Dataset):
    def __init__(self, wanted_type_col, flux_col, wanted_types, wanted_feature_names,
                 point_source_catalogue):
        # Create x/input data
        # self.xdata = []
        self.signif_index = [0,10,20,30,40,50,60,80,100,125,150]
        self.psr_signif = [0,0,0,0,0,0,0,0,0,0,0]
        self.agn_signif = [0,0,0,0,0,0,0,0,0,0,0]
        self.unassociated_signif = [0,0,0,0,0,0,0,0,0,0,0]
        self.ydata = []
        # Select sources that are a part of the wanted types
        for source in point_source_catalogue.data:
            signif_avg = source['Signif_Avg']
            if signif_avg < 10:
                ind = 0
            elif signif_avg >= 10 and signif_avg < 20:
                ind = 1
            elif signif_avg >= 20 and signif_avg < 30:
                ind = 2
            elif signif_avg >= 30 and signif_avg < 40:
                ind = 3
            elif signif_avg >= 40 and signif_avg < 50:
                ind = 4
            elif signif_avg >= 50 and signif_avg < 60:
                ind = 5
            elif signif_avg >= 60 and signif_avg < 80:
                ind = 6
            elif signif_avg >= 80 and signif_avg < 100:
                ind = 7
            elif signif_avg >= 100 and signif_avg < 125:
                ind = 8
            else:
                ind = 9
            if source[wanted_type_col] in wanted_types:
                if source[wanted_type_col] in ("PSR", "psr", "MSP", "msp"):
                    self.ydata.append([0])  # 0 is for pulsar
                    self.psr_signif[ind] = self.psr_signif[ind]+1
                else:
                    self.ydata.append([1])  # 1 is for AGN
                    self.agn_signif[ind] = self.agn_signif[ind]+1
            elif source[wanted_type_col] == '':
                self.ydata.append([2])
                self.unassociated_signif[ind] = self.unassociated_signif[ind]+1
        self.y = torch.from_numpy(np.array(self.ydata, float))
        self.size = len(self.ydata)

    def __getitem__(self, index):
        return self.y[index]

    def __len__(self):
        return self.size

    def get_signif(self):
        return self.signif_index,self.psr_signif,self.agn_signif,self.unassociated_signif

import math
import torch
import numpy as np
from torch.utils.data import Dataset


class FGL3Dataset_Tbl6(Dataset):
    def __init__(self, wanted_type_col, fluxlevels_3fgl, wanted_types,
                 wantedfeaturenames, point_source_catalogue):
        # Create x/input data
        self.xdata = []
        self.ydata = []
        self.zdata = []
        # Select sources that are a part of the wanted types
        for source in point_source_catalogue.data:
            # for feature_names in wanted_feature_names:
            #     print("{0} = {1}".format(feature_names, source[feature_names]))
            # print("(type(source[flux_col]) = ", type(source[flux_col]))
            # flux_list = res = source[flux_col].tolist()
            # print("source[flux_col] = {0}".format(flux_list))
            if source[wanted_type_col].strip() == '':
                unit = []
                # Append wanted params. Applied a log transformation to paras with highly skewed distributions.
                unit.append(float(source[wantedfeaturenames[0]]))
                unit.append(math.log(source[wantedfeaturenames[1]])
                            if (source[wantedfeaturenames[1]] > 0) else float(source[wantedfeaturenames[1]]))
                unit.append(math.log(source[wantedfeaturenames[2]])
                            if (source[wantedfeaturenames[2]] > 0) else float(source[wantedfeaturenames[2]]))
                unit.append(math.log(source[wantedfeaturenames[3]])
                            if (source[wantedfeaturenames[3]] > 0) else float(source[wantedfeaturenames[3]]))
                unit.append(math.log(source[wantedfeaturenames[4]])
                            if (source[wantedfeaturenames[4]] > 0) else float(source[wantedfeaturenames[4]]))
                # Calculate hardness ratios (also considered in the paper)
                # for lvl_index in range(1, len(fluxlevels_3fgl)):
                #     unit.append((source[fluxlevels_3fgl[lvl_index]] - source[fluxlevels_3fgl[lvl_index - 1]] + 0.0) / (
                #                 source[fluxlevels_3fgl[lvl_index]] + source[fluxlevels_3fgl[lvl_index - 1]] + 0.0))
                sum_flux = 0
                count = 0
                for ind_outer in range(0, len(fluxlevels_3fgl)-1):
                    for ind_inner in range(ind_outer+1, len(fluxlevels_3fgl)):
                        count += 1
                        sum_flux += (source[fluxlevels_3fgl[ind_inner]] - source[fluxlevels_3fgl[ind_outer]] + 0.0) / (
                                     source[fluxlevels_3fgl[ind_inner]] + source[fluxlevels_3fgl[ind_outer]] + 0.0)
                unit.append(sum_flux/count)
                self.xdata.append(unit)
                self.zdata.append(source['Source_Name'].split(' ')[1].strip())
                if source[wanted_type_col] in ("PSR", "psr", "MSP", "msp"):
                    self.ydata.append([0])  # 0 is for pulsar
                else:
                    self.ydata.append([1])  # 1 is for AGN
        self.x = torch.from_numpy(np.array(self.xdata, float))
        self.y = torch.from_numpy(np.array(self.ydata, float))
        self.size = len(self.xdata)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size

    def get_xyz(self):
        return self.xdata, self.ydata, self.zdata

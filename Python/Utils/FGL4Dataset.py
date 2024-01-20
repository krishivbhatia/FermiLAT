import math
import torch
import numpy as np
from torch.utils.data import Dataset


class FGL4Dataset(Dataset):
    def __init__(self, wanted_type_col, flux_col, wanted_types, wanted_feature_names, point_source_catalogue):
        # Create x/input data
        xdata = []
        ydata = []
        # Select sources that are a part of the wanted types
        for source in point_source_catalogue.data:
            # for feature_names in wanted_feature_names:
            #     print("{0} = {1}".format(feature_names, source[feature_names]))
            # print("(type(source[flux_col]) = ", type(source[flux_col]))
            flux_list = res = source[flux_col].tolist()
            # print("source[flux_col] = {0}".format(flux_list))
            if source[wanted_type_col] in wanted_types:
                unit = []
                # Append wanted params. Applied a log transformation to paras with highly skewed distributions.
                unit.append(float(source[wanted_feature_names[0]]))
                unit.append(float(source[wanted_feature_names[1]]))
                unit.append(float(source[wanted_feature_names[2]]))
                unit.append(float(source[wanted_feature_names[3]]))
                unit.append(math.log(source[wanted_feature_names[4]])
                            if (source[wanted_feature_names[4]] > 0) else float(source[wanted_feature_names[4]]))
                unit.append(math.log(source[wanted_feature_names[5]])
                            if (source[wanted_feature_names[5]] > 0) else float(source[wanted_feature_names[5]]))
                unit.append(math.log(source[wanted_feature_names[6]])
                            if (source[wanted_feature_names[6]] > 0) else float(source[wanted_feature_names[6]]))
                unit.append(math.log(source[wanted_feature_names[7]])
                            if (source[wanted_feature_names[7]] > 0) else float(source[wanted_feature_names[7]]))
                unit.append(math.log(source[wanted_feature_names[8]])
                            if (source[wanted_feature_names[8]] > 0) else float(source[wanted_feature_names[8]]))
                # unit.append(math.log(source[wanted_feature_names[9]])
                #             if (source[wanted_feature_names[9]] > 0) else float(source[wanted_feature_names[9]]))
                unit.append(float(source[wanted_feature_names[9]]))
                unit.append(float(source[wanted_feature_names[10]]))
                unit.append(float(source[wanted_feature_names[11]]))
                unit.append(float(source[wanted_feature_names[12]]))
                # Calculate hardness ratios (also considered in the paper)
                for index in range(1, len(flux_list)):
                    unit.append((flux_list[index] - flux_list[index-1] + 0.0) /
                                (flux_list[index] + flux_list[index-1] + 0.0))
                xdata.append(unit)
                if source[wanted_type_col] in ("PSR", "psr"):
                    ydata.append([0])  # 0 is for pulsar
                else:
                    ydata.append([1])  # 1 is for AGN
        self.x = torch.from_numpy(np.array(xdata, float))
        self.y = torch.from_numpy(np.array(ydata, float))
        self.size = len(xdata)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size

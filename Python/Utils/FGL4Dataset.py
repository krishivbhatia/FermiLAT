import math
import torch
import numpy as np
from torch.utils.data import Dataset


class FGL4Dataset(Dataset):
    def __init__(self, wanted_type_col, fluxindices, wantedtypes, wantedfeaturenames, pointsourcecatalogue):
        # Create x/input data
        xdata = []
        ydata = []
        # Select sources that are a part of the wanted types
        for source in pointsourcecatalogue.data:
            # for feature_names in wantedfeaturenames:
            #     print("{0} = {1}".format(feature_names, source[feature_names]))
            if source[wanted_type_col] in wantedtypes:
                unit = []
                # Append wanted params. Applied a log transformation to paras with highly skewed distributions.
                unit.append(float(source[wantedfeaturenames[0]]))
                unit.append(float(source[wantedfeaturenames[1]]))
                unit.append(float(source[wantedfeaturenames[2]]))
                unit.append(float(source[wantedfeaturenames[3]]))
                unit.append(math.log(source[wantedfeaturenames[4]])
                            if (source[wantedfeaturenames[4]] > 0) else float(source[wantedfeaturenames[4]]))
                unit.append(math.log(source[wantedfeaturenames[5]])
                            if (source[wantedfeaturenames[5]] > 0) else float(source[wantedfeaturenames[5]]))
                unit.append(math.log(source[wantedfeaturenames[6]])
                            if (source[wantedfeaturenames[6]] > 0) else float(source[wantedfeaturenames[6]]))
                unit.append(math.log(source[wantedfeaturenames[7]])
                            if (source[wantedfeaturenames[7]] > 0) else float(source[wantedfeaturenames[7]]))
                unit.append(math.log(source[wantedfeaturenames[8]])
                            if (source[wantedfeaturenames[8]] > 0) else float(source[wantedfeaturenames[8]]))
                # unit.append(math.log(source[wantedfeaturenames[9]])
                #             if (source[wantedfeaturenames[9]] > 0) else float(source[wantedfeaturenames[9]]))
                unit.append(float(source[wantedfeaturenames[9]]))
                unit.append(float(source[wantedfeaturenames[10]]))
                unit.append(float(source[wantedfeaturenames[11]]))
                unit.append(float(source[wantedfeaturenames[12]]))
                # Calculate hardness ratios (also considered in the paper)
                for index in range(1, len(fluxindices)):
                    unit.append((source[fluxindices[index]] - source[fluxindices[index-1]] + 0.0) /
                                (source[fluxindices[index]] + source[fluxindices[index-1]] + 0.0))
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

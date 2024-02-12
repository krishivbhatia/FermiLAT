import torch
import math
import numpy as np
from torch.utils.data import Dataset


class FGL3Dataset(Dataset):
    def __init__(self, wanted_type_col, fluxindices, wantedtypes, wantedfeaturenames, pointsourcecatalogue):
        # Create x/input data
        xdata = []
        ydata = []
        # Select sources that are a part of the wanted types
        for source in pointsourcecatalogue.data:
            # for feature_names in wantedfeaturenames:
            #     print("{0} = {1}".format(feature_names, source[feature_names]))
            if source[wanted_type_col] in wantedtypes:
                # print("{}".format(source[wanted_type_col]))
                unit = []
                # Append wanted parameters. We then applied a log transformation to some parameters that displayed
                # highly skewed distributions.
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
                for index in range(1, len(fluxindices)):
                    unit.append((source[fluxindices[index]] - source[fluxindices[index-1]] + 0.0) / (source[fluxindices[index]] + source[fluxindices[index-1]] + 0.0))
                if source[wanted_type_col] in ("PSR", "psr", "MSP", "msp"):
                    xdata.append(unit)
                    ydata.append([0])  # 0 is for pulsar
                elif source[wanted_type_col]:
                    xdata.append(unit)
                    ydata.append([1])  # 1 is for AGN
        self.x = torch.from_numpy(np.array(xdata, float))
        self.y = torch.from_numpy(np.array(ydata, float))
        self.size = len(xdata)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size

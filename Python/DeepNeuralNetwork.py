import math
import torch
import torch.nn as nn


# Logistic Regression Model
#   https://www.youtube.com/watch?v=OGpQxIkR4ao&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=8
# However I used 2 layers of neurons instead of 1 as it yields better results.
# 1 layers of neurons would just be 1/(1 + e^-(input1 * weight1 + input2 * weight2 + input3 * weight3 ... + bias))
class DeepNeuralNetworkModel(nn.Module):
    def __init__(self, input_features):
        print("(len(input_features[0][0]) = ", len(input_features[0][0]))
        print("input_features[0][0] = ", input_features[0][0])
        super(DeepNeuralNetworkModel, self).__init__()
        self.model = nn.Sequential(nn.Linear(len(input_features[0][0]), 250),
                                   nn.ReLU(),
                                   nn.Linear(250, 40),
                                   nn.ReLU(),
                                   nn.Linear(40, 1))
        print()
        print(self.model)
        print()

    def forward(self, x):
        y_predicted = torch.sigmoid(self.model(x))
        return y_predicted

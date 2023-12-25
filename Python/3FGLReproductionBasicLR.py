#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:15:48 2023

@author: pablo
"""
import torch as pt
import numpy as np
from astropy.io import fits
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from statistics import *

#https://www.youtube.com/watch?v=qx6y1OX4S6A Astropy for opening fits files

#Use OS to find path https://www.pythoncheatsheet.org/cheatsheet/file-directory-path
path = os.path.join(os.getcwd(), "../FITS/gll_psc_v16.fit")
print(path)
mainfile = fits.open(path)
mainfile.info()
pointsourcecatalogue = mainfile[1]
#print(pointsourcecatalogue.info())

#print(pointsourcecatalogue.header)
#print(list(pointsourcecatalogue.header.keys()))

columns = 0
keymap = {}
mapkey = {}
for key in list(pointsourcecatalogue.header.keys()):
    if "TTYPE" in key:
        keymap[int(key[5:len(key)])-1] = pointsourcecatalogue.header[key]
        mapkey[pointsourcecatalogue.header[key]] = int(key[5:len(key)])-1
        #print(pointsourcecatalogue.header[key])
        columns += 1
#print(columns)
keymap = OrderedDict(sorted(keymap.items()))
mapkey = OrderedDict(sorted(mapkey.items()))
print(keymap)
print(mapkey)
#This keymap contains all the names of the corresponding column, keymap[n] contains the name of the n+1th column

#pointsourcecatalogue.data[a][b] corresponds with the a+1th row and b+1th column in the catalogue
#It is recommended you open the fit file to help easily find corresponding values
print(pointsourcecatalogue.data[1][1])

#Creating Training and Test Datasets
#The paper selects these classes of objects to be apart of the dataset
wantedtypes = ["PSR", "psr", "YNG", "yng", "MSP", "FSRQ", "fsrq", "BLL", "bll", "BCU", "bcu", "RDG", "rdg", "NLSY1", "nlsy1", "agn", "ssrq", "sey"]
#The paper selects these columns/features to be the inputs that will be taken into account
wantedfeaturenames = ["Spectral_Index", "Variability_Index", "Flux_Density", "Unc_Energy_Flux100", "Signif_Curve"]
wantedfeatureindices = [mapkey[x] for x in wantedfeaturenames]
#The paper wants hardness ratios of fluxes
fluxlevels = ["Flux100_300", "Flux300_1000", "Flux1000_3000", "Flux3000_10000", "Flux10000_100000"]
fluxindices = [mapkey[x] for x in fluxlevels]
#Pytorch Datasets and ML Models are constructed in modular class/OOP style

sc = StandardScaler()
class FGL3Dataset(Dataset):
    def __init__(self):
        #Create x/input data
        xdata = []
        ydata = []
        #Scratch this nevermind agncounter = 0 #Due to the overwhelming amount of AGNs compared with PSRs, the machine always determines
        #Select sources that are apart of the wanted types
        for source in pointsourcecatalogue.data:
            if (source[73] in wantedtypes):
                unit = []
                #Append wanted parameters. We then applied a log transformation to some parameters that displayed highly skewed distributions.
                #I noticed that logging the skewed parameters made the model converge much slower
                unit.append(source["Spectral_Index"])
                unit.append(math.log(source["Variability_Index"]) if (source["Variability_Index"] > 0) else source["Variability_Index"])
                unit.append(math.log(source["Flux_Density"]) if (source["Flux_Density"] > 0) else source["Flux_Density"])
                unit.append(math.log(source["Unc_Energy_Flux100"]) if (source["Unc_Energy_Flux100"] > 0) else source["Unc_Energy_Flux100"])
                unit.append(math.log(source["Signif_Curve"]) if (source["Signif_Curve"] > 0) else source["Signif_Curve"])
                #Calculate hardness ratios (also considered in the paper)
                unit.append((source[fluxindices[1]] - source[fluxindices[0]] + 0.0) / (source[fluxindices[1]] + source[fluxindices[0]] + 0.0)) #HR12
                unit.append((source[fluxindices[2]] - source[fluxindices[1]] + 0.0) / (source[fluxindices[2]] + source[fluxindices[1]] + 0.0)) #HR23
                unit.append((source[fluxindices[3]] - source[fluxindices[2]] + 0.0) / (source[fluxindices[3]] + source[fluxindices[2]] + 0.0)) #HR34
                unit.append((source[fluxindices[4]] - source[fluxindices[3]] + 0.0) / (source[fluxindices[4]] + source[fluxindices[3]] + 0.0)) #HR45
                #Each unit/case should consist of (in this order) ["Spectral_Index", "Variability_Index", "Flux_Density", "Unc_Energy_Flux100", "Signif_Curve",
                #"HR12", "HR23", "HR34", "HR45"]
                if (source[73] == "PSR" or source[73] == "psr"):
                    xdata.append(unit)
                    ydata.append([0]) # 0 is for pulsar
                else: #elif (agncounter <= 300):
                    xdata.append(unit)
                    ydata.append([1]) # 1 is for AGN
        self.x = torch.from_numpy(np.array(xdata, float))
        self.y = torch.from_numpy(np.array(ydata, float))
        self.size = len(xdata)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.size

#Now we can make our dataset and dataloader
#More on basics of dataloaders here: https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
dataset = FGL3Dataset()
#print(len(dataset))
torch.manual_seed(42) #Set shuffle seed to a certain value for reproducibility
dataloader = DataLoader(dataset=dataset, shuffle=True)
#Splitting dataloader into train/dev/test sets
train_size = int(0.7 * len(dataloader.dataset)) #You did a 70%:30% train:test split
test_size = len(dataloader.dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, test_size])
print(train_size)
print(test_size)
print(len(dataloader))
print(len(dataloader.dataset))
print(len(train_dataset))
print(len(test_dataset))
print(len(train_dataset.dataset))


"""
dataiter = iter(dataloader)
data = next(dataiter)
features, label = data
print(features, label)
"""

#Logistic Regression Model https://www.youtube.com/watch?v=OGpQxIkR4ao&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=8
#However I used 2 layers of neurons instead of 1 as it yields better results.
#1 layers of neurons would just be 1/(1 + e^-(input1 * weight1 + input2 * weight2 + input3 * weight3 ... + bias))
class LogisticRegression(nn.Module):
    def __init__(self, input_features):
        #print(len(input_features[0][0]))
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(len(input_features[0][0]), 1)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        #print(type(y_predicted))
        #print(y_predicted.size())
        return y_predicted


#Training Logistic Regression Model/Model Building Procedure

#num_epochs = 40 #40 epochs, anymore might result in overfitting in certain seeds
#In 10 epochs, 1 epoch will use a different final subset from one of the 10 subsets for testing
best_p_values = []
for iteration in range(0, 10):
    total_samples = len(train_dataset)
    dev_predictions = []
    dev_inputs = []
    dev_labels = []
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(42)
    LogRegModel = LogisticRegression(train_dataset) #Start from new model every epoch
    criterion = nn.BCELoss() #Binary Cross Entropy Loss which is the loss method that would most likely be used in this scenario
    optimizer = torch.optim.SGD(LogRegModel.parameters(), lr=0.0001) #Stochastic Gradient Descent. I could use Adams but Adams is worse than regular SGD unless you
    #spend ages finetuning it which in that case it performs better but I have 0 time to spare
    #Train on 9 subsets
    for i, (inputs, labels) in enumerate(train_dataset):
        """
        print(type(inputs))
        print(inputs)
        print(type(labels))
        print(labels)
        print(labels.size())
        """
        #"We used nine subsets to build a model and apply the fitted model to test on the remaining subset. We then repeated this procedure for all 10
        #subsets until all the subsets were tested.
        if (i < (iteration % 10) * math.ceil(total_samples / 10) or i >= (iteration % 10 + 1) * math.ceil(total_samples / 10)):
            y_predicted = LogRegModel(inputs) #Insert Model here
            loss = criterion(y_predicted, labels)
            #Pytorch Gradient Descent Procedure
            loss.backward() #Backwards pass in back propagation
            optimizer.step() #Update weights based on gradients
            optimizer.zero_grad() #Reset gradients to 0
            #Print results every 10 training iterations
        else:
            dev_inputs.append(inputs)
            dev_labels.append(labels.item())
    #Train for more epochs, 1 ain't enough LOL, you need at least 20-30 for good performance (I might be wrong here, please tell me a good amount of epochs)
    for epoch in range(0, 29):
        for i, (inputs, labels) in enumerate(train_dataset):
            """
            print(type(inputs))
            print(inputs)
            print(type(labels))
            print(labels)
            print(labels.size())
            """
            #"We used nine subsets to build a model and apply the fitted model to test on the remaining subset. We then repeated this procedure for all 10
            #subsets until all the subsets were tested.
            if (i < (iteration % 10) * math.ceil(total_samples / 10) or i >= (iteration % 10 + 1) * math.ceil(total_samples / 10)):
                y_predicted = LogRegModel(inputs) #Insert Model here
                loss = criterion(y_predicted, labels)
                #Pytorch Gradient Descent Procedure
                loss.backward() #Backwards pass in back propagation
                optimizer.step() #Update weights based on gradients
                optimizer.zero_grad() #Reset gradients to 0
                #Print results every 10 training iterations
    #Test on final subset (different every time) to figure out P threshold value
    with torch.no_grad():
        for i in range(0, len(dev_inputs)):
            dev_prediction = LogRegModel(dev_inputs[i])
            dev_predictions.append(dev_prediction)
        xpoints = []
        ypoints = []
        bestp = 0
        bestscore = 0
        bestpsrscore = 0
        bestagnscore = 0
        bestaccuracy = 0
        n_samples = len(dev_inputs)
        for pp in range(0, 1001):
            p = pp * 0.001
            psrcount = 0
            agncount = 0
            truepositive = 0 #True Positive Sensitivity
            truenegative = 0 #True Negative Specificity
            correct = 0 #Total Correct
            #True Negative = Pulsar Successfully Identified
            #True Positive = AGN Successfully Identified
            for i in range(0, len(dev_inputs)):
                if (dev_labels[i] == 1):
                    agncount += 1
                else:
                    psrcount += 1
                if (dev_predictions[i] >= p and dev_labels[i] == 1):
                    truepositive += 1
                    correct += 1
                if (dev_predictions[i] < p and dev_labels[i]== 0):
                    truenegative += 1
                    correct += 1
            xpoints.append(1 if (agncount == 0) else (truepositive / (agncount + 0.0))) #AGN Score/True Positive/Sensitivity
            ypoints.append(1 if (psrcount == 0) else (truenegative / (psrcount + 0.0))) #Pulsar Score/True Negative/Specificity
            score = (1 if (agncount == 0) else (truepositive / (agncount + 0.0))) + (1 if (psrcount == 0) else (truenegative / (psrcount + 0.0)))
            #The paper calculates the score as sensitivity + specificity
            if (score > bestscore):
                bestscore = score
                bestp = p
                bestagnscore = 1 if (agncount == 0) else (truepositive / (agncount + 0.0))
                bestpsrscore = 1 if (psrcount == 0) else (truenegative / (psrcount + 0.0))
                bestaccuracy = (correct / (n_samples + 0.0))
        #print(n_correct / (n_samples + 0.0))
        #
        plt.plot(xpoints, ypoints)
        print(f"Best P Threshold Value : {bestp}")
        print(f"Best Score : {bestscore}")
        print(f"Best Pulsar/True Negative/Specificity : {bestpsrscore}")
        print(f"Best AGN/True Positive/Sensitivity : {bestagnscore}")
        print(f"Accuracy of Best P Fit : {bestaccuracy}")
        plt.plot(bestagnscore, bestpsrscore, marker="x", markersize=10, markeredgecolor="red")
        plt.xlabel = "AGN Score/True Positive/Sensitivity"
        plt.ylabel = "Pulsar Score/True Negative/Specificity"
        plt.show()
        best_p_values.append(bestp)
print(best_p_values)
#Generating Sensitivity vs Specificity Graph
#We now have a list of best p-values, but I am not sure on how to generate a best-p-value from these values so I will just take the average of them

print(best_p_values)
optimal_p = mean(best_p_values)
print(optimal_p)
torch.manual_seed(42)
LogRegModel = LogisticRegression(train_dataset) #Now we found the optimal p value, we are ready to actually start training and testing the model
criterion = nn.BCELoss() #Binary Cross Entropy Loss which is the loss method that would most likely be used in this scenario
optimizer = torch.optim.SGD(LogRegModel.parameters(), lr=0.0001) #Stochastic Gradient Descent. I could use Adams but Adams is worse than regular SGD unless you
# spend ages finetuning it which in that case it performs better, but I have 0 time to spare
for epoch in range(0, 30):
    for i, (inputs, labels) in enumerate(train_dataset):
        y_predicted = LogRegModel(inputs) #Insert/fit Model for prediction here
        loss = criterion(y_predicted, labels)
        # Pytorch Gradient Descent Procedure
        loss.backward() #Backwards pass in back propagation
        optimizer.step() #Update weights based on gradients
        optimizer.zero_grad() #Reset gradients to 0
    if ((epoch + 1) % 5 == 0):
        with torch.no_grad():
            correct = 0
            optimal_p = 0
            total = len(test_dataset)
            for i, (inputs, labels) in enumerate(test_dataset):
                y_predicted = LogRegModel(inputs) #Insert/fit Model for prediction here
                if (y_predicted < optimal_p and labels.item() == 0 or y_predicted > optimal_p and labels.item() == 1):
                    correct += 1
            print(f"Accuracy: {correct / (total + 0.0)}")
    with torch.no_grad():
        for i in range(0, len(dev_inputs)):
            dev_prediction = LogRegModel(dev_inputs[i])
            dev_predictions.append(dev_prediction)
        xpoints = []
        ypoints = []
        bestp = 0
        bestscore = 0
        bestpsrscore = 0
        bestagnscore = 0
        bestaccuracy = 0
        n_samples = len(dev_inputs)
        for pp in range(0, 1001):
            p = pp * 0.001
            psrcount = 0
            agncount = 0
            truepositive = 0 #True Positive Sensitivity
            truenegative = 0 #True Negative Specificity
            correct = 0 #Total Correct
            #True Negative = Pulsar Successfully Identified
            #True Positive = AGN Successfully Identified
            for i in range(0, len(dev_inputs)):
                if (dev_labels[i] == 1):
                    agncount += 1
                else:
                    psrcount += 1
                if (dev_predictions[i] >= p and dev_labels[i] == 1):
                    truepositive += 1
                    correct += 1
                if (dev_predictions[i] < p and dev_labels[i]== 0):
                    truenegative += 1
                    correct += 1
            xpoints.append(1 if (agncount == 0) else (truepositive / (agncount + 0.0))) #AGN Score/True Positive/Sensitivity
            ypoints.append(1 if (psrcount == 0) else (truenegative / (psrcount + 0.0))) #Pulsar Score/True Negative/Specificity
            score = (1 if (agncount == 0) else (truepositive / (agncount + 0.0))) + (1 if (psrcount == 0) else (truenegative / (psrcount + 0.0)))
            #The paper calculates the score as sensitivity + specificity
            if (score > bestscore):
                bestscore = score
                bestp = p
                bestagnscore = 1 if (agncount == 0) else (truepositive / (agncount + 0.0))
                bestpsrscore = 1 if (psrcount == 0) else (truenegative / (psrcount + 0.0))
                bestaccuracy = (correct / (n_samples + 0.0))
        #print(n_correct / (n_samples + 0.0))
        #
        plt.plot(xpoints, ypoints)
        print(f"Best P Threshold Value : {bestp}")
        print(f"Best Score : {bestscore}")
        print(f"Best Pulsar/True Negative/Specificity : {bestpsrscore}")
        print(f"Best AGN/True Positive/Sensitivity : {bestagnscore}")
        print(f"Accuracy of Best P Fit : {bestaccuracy}")
        plt.plot(bestagnscore, bestpsrscore, marker="x", markersize=10, markeredgecolor="red")
        plt.xlabel = "AGN Score/True Positive/Sensitivity"
        plt.ylabel = "Pulsar Score/True Negative/Specificity"
        plt.show()
        best_p_values.append(bestp)
#Problem: The values recorded for the best p values are actually very sub-par. The model's recorded accuracy got worse right after the first 5 epochs. Using a lower p value as a threshold yields much better results. I literally set p to 0.8 (most of the recorded best values were around 0.9) and this worked a lot better, the accuracy increased steadily for 30-50 epochs
#Update to Problem: I tried a different seed, and the best-p-values (still around 0.9) calculated from the 10-fold cross validation method seemed to work a lot better, but it still converged too quickly, as its accuracy started dropping after 15 epochs. Meanwhile p = 0.8 still worked just as excellently. But maybe setting p ~ 0.9 makes it perform better on pulsars, after all your paper based the score on pulsar accuracy and agn accuracy, not the overall accuracy which is heavily skewed by agn accuracy as an overwhelming majority of the sources are agn. Or maybe 30 epochs is just way too much.
#IGNORE THE COMMENTED OUT CODE BELOW
"""
with torch.no_grad():
    for i in range(0, len(test_dataset)):
        dev_prediction = LogRegModel(test_dataset[i])
        dev_predictions.append(dev_prediction)
    xpoints = []
    ypoints = []
    bestp = 0
    bestscore = 0
    bestpsrscore = 0
    bestagnscore = 0
    bestaccuracy = 0
    n_samples = len(test_dataset)
    for pp in range(0, 1001):
        p = pp * 0.001
        psrcount = 0
        agncount = 0
        truepositive = 0 #True Positive Sensitivity
        truenegative = 0 #True Negative Specificity
        correct = 0 #Total Correct
        #True Negative = Pulsar Successfully Identified
        #True Positive = AGN Successfully Identified
        for i in range(0, len(test_dataset)):
            if (dev_labels[i] == 1):
                agncount += 1
            else:
                psrcount += 1
            if (dev_predictions[i] >= p and dev_labels[i] == 1):
                truepositive += 1
                correct += 1
            if (dev_predictions[i] < p and dev_labels[i]== 0):
                truenegative += 1
                correct += 1
        xpoints.append(1 if (agncount == 0) else (truepositive / (agncount + 0.0))) #AGN Score/True Positive/Sensitivity
        ypoints.append(1 if (psrcount == 0) else (truenegative / (psrcount + 0.0))) #Pulsar Score/True Negative/Specificity
        score = (1 if (agncount == 0) else (truepositive / (agncount + 0.0))) + (1 if (psrcount == 0) else (truenegative / (psrcount + 0.0)))
        #The paper calculates the score as sensitivity + specificity
        if (score > bestscore):
            bestscore = score
            bestp = p
            bestagnscore = 1 if (agncount == 0) else (truepositive / (agncount + 0.0))
            bestpsrscore = 1 if (psrcount == 0) else (truenegative / (psrcount + 0.0))
            bestaccuracy = (correct / (n_samples + 0.0))
    #print(n_correct / (n_samples + 0.0))
    plt.plot(xpoints, ypoints)
    print(f"Best P Threshold Value : {bestp}")
    print(f"Best Score : {bestscore}")
    print(f"Best Pulsar/True Negative/Specificity : {bestpsrscore}")
    print(f"Best AGN/True Positive/Sensitivity : {bestagnscore}")
    print(f"Accuracy of Best P Fit : {bestaccuracy}")
    plt.plot(bestagnscore, bestpsrscore, marker="x", markersize=10, markeredgecolor="red")
    plt.xlabel = "AGN Score/True Positive/Sensitivity"
    plt.ylabel = "Pulsar Score/True Negative/Specificity"
    plt.show()
    best_p_values.append(bestp)
"""
#Another problem: I do not know how to implement random forest and boosted LR the way you did it in your paper and I therefore greatly need assistance in implementing the two. Please direct me to sources that can help me implement these in python and/or teach me the gist on how to implement these 2 in general.

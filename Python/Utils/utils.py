import math
import torch
import random
import torch.nn as nn
from math import log, sqrt


def run_iteration(LogRegModel, iteration, train_dataset, total_samples, criterion,
                  optimizer, dev_inputs, dev_labels, append_dev_input_label):
    for i, (inputs, labels) in enumerate(train_dataset):
        """
        print(type(inputs))
        print(inputs)
        print(type(labels))
        print(labels)
        print(labels.size())
        """
        # We used nine subsets to build a model and apply the fitted model to test on the remaining subset.
        # We then repeated this procedure for all 10
        # subsets until all the subsets were tested.
        if i < (iteration % 10) * math.ceil(total_samples / 10) \
                or i >= (iteration % 10 + 1) * math.ceil(total_samples / 10):
            y_predicted = LogRegModel(inputs)  # Insert Model here
            loss = criterion(y_predicted, labels)
            # Pytorch Gradient Descent Procedure
            loss.backward()  # Backwards pass in back propagation
            optimizer.step()  # Update weights based on gradients
            optimizer.zero_grad()  # Reset gradients to 0
            # Print results every 10 training iterations
        elif append_dev_input_label:
            dev_inputs.append(inputs)
            dev_labels.append(labels.item())
    return dev_inputs, dev_labels

def unique_counts(labels):
    """
    Unique count function used to count labels.
    """
    results = {}
    for label in labels:
        value = label.item()
        if value not in results.keys():
            results[value] = 0
        results[value] += 1
    return results


def divide_set(vectors, labels, column, value):
    """
    Divide the sets into two different sets along a specific dimension and value.
    """
    set_1 = [(vector, label) for vector, label in zip(vectors, labels) if split_function(vector, column, value)]
    set_2 = [(vector, label) for vector, label in zip(vectors, labels) if not split_function(vector, column, value)]

    vectors_set_1 = [element[0] for element in set_1]
    vectors_set_2 = [element[0] for element in set_2]
    label_set_1 = [element[1] for element in set_1]
    label_set_2 = [element[1] for element in set_2]

    return vectors_set_1, label_set_1, vectors_set_2, label_set_2


def split_function(vector, column, value):
    """
    Split function
    """
    return vector[column] >= value


def log2(x):
    """
    Log2 function
    """
    return log(x) / log(2)


def sample_vectors(vectors, labels, nb_samples):
    """
    Sample vectors and labels uniformly.
    """
    sampled_indices = torch.LongTensor(random.sample(range(len(vectors)), nb_samples))
    # print("sampled_indices = ", sampled_indices)
    sampled_vectors = torch.index_select(vectors,0, sampled_indices)
    # print("sampled_vectors = ", sampled_vectors)
    sampled_labels = torch.index_select(labels,0, sampled_indices)
    # print("sampled_labels = ", sampled_labels)

    return sampled_vectors, sampled_labels


def sample_dimensions(vectors):
    """
    Sample vectors along dimension uniformly.
    """
    sample_dimension = torch.LongTensor(random.sample(range(len(vectors[0])), int(sqrt(len(vectors[0])))))
    return sample_dimension


def entropy(labels):
    """
    Entropy function.
    """
    results = unique_counts(labels)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(labels)
        ent = ent - p * log2(p)
    return ent


def variance(values):
    """
    Variance function.
    """
    mean_value = mean(values)
    var = 0.0
    for value in values:
        var = var + torch.sum(torch.sqrt(torch.pow(value-mean_value,2))).item()/len(values)
    return var


def mean(values):
    """
    Mean function.
    """
    m = 0.0
    for value in values:
        m = m + value/len(values)
    return m


def dataset_to_features_labels(dataset):
    # Krishiv Bhatia: train or test dataset is of the form:
    #    tensor array(train/test features), tensor array (output label)
    # or like this:
    #    tensor array(feature 0, feature 1, ... feature 12), tensor array(output label)
    # You can print it like this:
    #     for i in train_dataset:
    #         print(i)
    # An example:
    #    (tensor([  2.1602,   1.8308,   1.6083,  13.2215, -32.9276, -32.4171, -29.4589,
    #             -32.3813, -24.3150,   0.4812,   0.0000,   2.7429,   5.4350]), tensor([1.]))
    #
    # Convert tensor feature array in dataset to numpy feature array
    train_features = ([((i[0]).detach().numpy()) for i in dataset])
    # print("train_features = ", train_features)
    # Convert tensor label array in dataset to regular array
    train_labels = ([torch.LongTensor.item(i[1]) for i in dataset])
    # print("train_labels = ", train_labels)
    return train_features, train_labels
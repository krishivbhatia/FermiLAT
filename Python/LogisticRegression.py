import math
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_features):
        # print(len(input_features[0][0]))
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(len(input_features[0][0]), 1)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        # print(type(y_predicted))
        # print(y_predicted.size())
        return y_predicted


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

# -*- coding: utf-8 -*-
import torch

from torch.utils.data import DataLoader
from Utils.utils import dataset_to_features_labels
from Utils.utils import sample_vectors, sample_dimensions
from DecisionTrees.decision_tree import TorchDecisionTreeClassifier, TorchDecisionTreeRegressor


class TorchRandomForestClassifier(torch.nn.Module):
    """
    Torch random forest object used to solve classification problem. This object implements the fitting and prediction
    function which can be used with torch tensors. The random forest is based on
    :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier` which are built during the :func:`fit` and called
    recursively during the :func:`predict`.

    Args:
        nb_trees (:class:`int`): Number of :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier` used to fit the
            classification problem.
        nb_samples (:class:`int`): Number of vector samples used to fit each
            :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier`.
        max_depth (:class:`int`): The maximum depth which corresponds to the maximum successive number of
            :class:`DecisionNode`.
        bootstrap (:class:`bool`): If set to true, a sample of the dimensions of the input vectors are made during the
            fitting and the prediction.

    """
    def __init__(self, nb_trees, nb_samples, max_depth=-1, bootstrap=True):
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def fit(self, vectors, labels):
        """
        Function which must be used after the initialisation to fit the random forest and build the successive
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier` to solve a specific classification problem.

        Args:
            vectors(:class:`torch.FloatTensor`): Vectors tensor used to fit the random forest. It represents the data
                and must correspond to the following shape (num_vectors, num_dimensions).
            labels (:class:`torch.LongTensor`): Labels tensor used to fit the decision tree. It represents the labels
                associated to each vectors and must correspond to the following shape (num_vectors).

        """
        print("In Random Forest: fit()")
        for i, _ in enumerate(range(self.nb_trees)):
            print("i = ", i)
            tree = TorchDecisionTreeClassifier(self.max_depth)
            list_features = sample_dimensions(vectors)
            self.trees_features.append(list_features)
            if self.bootstrap:
                sampled_vectors, sample_labels = sample_vectors(vectors, labels, self.nb_samples)
                sampled_featured_vectors = torch.index_select(sampled_vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, sample_labels)
            else:
                sampled_featured_vectors = torch.index_select(vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, labels)
            self.trees.append(tree)

    def predict(self, vector):
        """
        Function which must be used after the fitting of the random forest. It calls recursively the different
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeClassifier` to classify the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be classified. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.LongTensor`: Tensor which corresponds to the label predicted by the random forest.

        """
        predictions = []
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = torch.index_select(vector, 0, index_features)
            predictions.append(tree.predict(sampled_vector))
        print("predictions=", predictions)
        print("sum(predictions)={}, len(self.trees)={}, sum(predictions)/len(self.trees)={}",
              sum(predictions), len(self.trees), sum(predictions)/len(self.trees))
        return sum(predictions)/len(self.trees)
        # return max(set(predictions), key=predictions.count)

    def predict_tbl6(self, vector):
        predicted_value = self.predict(vector)
        return 'AGN' if predicted_value >= 0.5 else 'PSR'


class TorchRandomForestRegressor(torch.nn.Module):
    """
    Torch random forest object used to solve regression problem. This object implements the fitting and prediction
    function which can be used with torch tensors. The random forest is based on
    :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` which are built during the :func:`fit` and called
    recursively during the :func:`predict`.

    Args:
        nb_trees (:class:`int`): Number of :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` used to fit the
            classification problem.
        nb_samples (:class:`int`): Number of vector samples used to fit each
            :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor`.
        max_depth (:class:`int`): The maximum depth which corresponds to the maximum successive number of
            :class:`Sklearn_PyTorch.decision_node.DecisionNode`.
        bootstrap (:class:`bool`): If set to true, a sample of the dimensions of the input vectors are made during the
            fitting and the prediction.

    """
    def __init__(self,  nb_trees, nb_samples, max_depth=-1, bootstrap=True):
        self.trees = []
        self.trees_features = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
        self.bootstrap = bootstrap

    def fit(self, vectors, values):
        """
        Function which must be used after the initialisation to fit the random forest and build the successive
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` to solve a specific classification problem.

        Args:
            vectors(:class:`torch.FloatTensor`): Vectors tensor used to fit the decision tree. It represents the data
                and must correspond to the following shape (num_vectors, num_dimensions_vectors).
            values(:class:`torch.FloatTensor`): Values tensor used to fit the decision tree. It represents the values
                associated to each vectors and must correspond to the following shape (num_vectors,
                num_dimensions_values).

        """
        for _ in range(self.nb_trees):
            tree = TorchDecisionTreeRegressor(self.max_depth)
            list_features = sample_dimensions(vectors)
            self.trees_features.append(list_features)
            if self.bootstrap:
                sampled_vectors, sample_labels = sample_vectors(vectors, values, self.nb_samples)
                sampled_featured_vectors = torch.index_select(sampled_vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, sample_labels)
            else:
                sampled_featured_vectors = torch.index_select(vectors, 1, list_features)
                tree.fit(sampled_featured_vectors, values)
            self.trees.append(tree)

    def predict(self, vector):
        """
        Function which must be used after the the fitting of the random forest. It calls recursively the different
        :class:`Sklearn_PyTorch.binary_tree.TorchDecisionTreeRegressor` to regress the vector.

        Args:
            vector(:class:`torch.FloatTensor`): Vectors tensor which must be regressed. It represents the data
                and must correspond to the following shape (num_dimensions).

        Returns:
            :class:`torch.FloatTensor`: Tensor which corresponds to the value regressed by the random forest.

        """
        predictions_sum = 0
        for tree, index_features in zip(self.trees, self.trees_features):
            sampled_vector = torch.index_select(vector, 0, index_features)
            predictions_sum += tree.predict(sampled_vector)

        return predictions_sum/len(self.trees)


def rf_fit(dataset, trees, samples, depth):
    torch.manual_seed(42)  # Set shuffle seed to a certain value for reproducibility
    dataloader = DataLoader(dataset=dataset, shuffle=True)
    # Splitting dataloader into train/dev/test sets
    train_size = int(1.0 * len(dataloader.dataset))  # You did a 70%:30% train:test split
    test_size = len(dataloader.dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, test_size])
    print("train_size = ", train_size)
    print("test_size = ", test_size)
    print("len(dataloader) = ", len(dataloader))
    print("len(dataloader.dataset)) = ", len(dataloader.dataset))
    print("len(train_dataset) = ", len(train_dataset))
    print("len(test_dataset) = ", len(test_dataset))
    print("len(train_dataset.dataset) = ", len(train_dataset.dataset))
    total_samples = len(train_dataset)
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(42)
    # Krishiv Bhatia: Invoke TorchRandomForestClassifier
    random_forest = TorchRandomForestClassifier(trees, samples, depth)
    print("random forest = ", random_forest.nb_trees, random_forest.nb_samples, random_forest.max_depth)
    # Krishiv Bhatia: split train dataset into features and labels
    train_features, train_labels = dataset_to_features_labels(train_dataset)
    random_forest.fit(torch.FloatTensor(train_features), torch.LongTensor(train_labels))
    return random_forest
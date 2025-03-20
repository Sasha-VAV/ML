import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    p = np.sum(y, axis=0) / y.shape[0]
    # YOUR CODE HERE

    return -np.sum(p * np.log2(p + EPS))


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    p = np.sum(y, axis=0) / y.shape[0]

    # YOUR CODE HERE

    return 1 - np.sum(p**2)


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    # YOUR CODE HERE

    return np.sum((y - y.mean()) ** 2) / y.shape[0]


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE

    return np.sum(np.abs(y - np.median(y))) / y.shape[0]


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.0
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        "gini": (gini, True),  # (criterion, classification flag)
        "entropy": (entropy, True),
        "variance": (variance, False),
        "mad_median": (mad_median, False),
    }

    def __init__(
        self,
        n_classes=None,
        max_depth=np.inf,
        min_samples_split=2,
        criterion_name="gini",
        debug=False,
    ):

        assert (
            criterion_name in self.all_criterions.keys()
        ), "Criterion name must be on of the following: {}".format(
            self.all_criterions.keys()
        )

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided breeds subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        left_indices = X_subset[:, feature_index] < threshold
        X_left = X_subset[left_indices]
        y_left = y_subset[left_indices]

        right_indices = left_indices == 0
        X_right = X_subset[right_indices]
        y_right = y_subset[right_indices]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        (_, y_left), (_, y_right) = self.make_split(
            feature_index, threshold, X_subset, y_subset
        )

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        threshold = None
        feature_index = None
        min_criterion = None
        for i in range(0, X_subset.shape[1]):
            threshold_arr = X_subset[:, i].copy()
            for j in range(0, X_subset.shape[0]):
                y_left, y_right = self.make_split_only_y(
                    i, threshold_arr[j], X_subset, y_subset
                )
                if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                    continue
                criterion = y_left.shape[0] * self.criterion(y_left) + y_right.shape[
                    0
                ] * self.criterion(y_right)
                criterion /= y_subset.shape[0]
                if min_criterion is None or criterion < min_criterion:
                    min_criterion = criterion
                    threshold = threshold_arr[j]
                    feature_index = i

        return feature_index, threshold

    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        if self.depth == self.max_depth or y_subset.shape[0] < self.min_samples_split:
            if self.classification:
                probs = np.sum(y_subset, axis=0) / y_subset.shape[0]
                return np.argmax(probs), probs
            else:
                if y_subset.shape[0] == 0:
                    raise ValueError("y_subset is empty")
                if self.criterion_name == "mad_median":
                    return np.median(y_subset)
                return np.mean(y_subset)
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)
        (X_left, y_left), (X_right, y_right) = self.make_split(
            feature_index, threshold, X_subset, y_subset
        )

        def copy():
            temp = DecisionTree(
                n_classes=self.n_classes,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion_name=self.criterion_name,
                debug=self.debug,
            )
            temp.classification = self.classification
            temp.criterion = self.criterion
            return temp

        left = copy()
        left.depth = self.depth + 1
        right = copy()
        right.depth = self.depth + 1
        new_node.left_child = left.make_tree(X_left, y_left)
        new_node.right_child = right.make_tree(X_right, y_right)
        return new_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided breeds

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the breeds to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), "Wrong y shape"
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided breeds

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the breeds the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """

        # YOUR CODE HERE
        y_predicted = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            temp = self.root
            while (
                isinstance(temp, Node)
                and temp.left_child is not None
                and temp.right_child is not None
            ):
                if X[i, temp.feature_index] < temp.value:
                    temp = temp.left_child
                else:
                    temp = temp.right_child
            if self.classification:
                y_predicted[i, 0] = temp[0]
            else:
                y_predicted[i, 0] = temp
        return y_predicted

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided breeds

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the breeds the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, "Available only for classification problem"

        # YOUR CODE HERE
        y_predicted_probs = np.zeros((X.shape[0], self.n_classes))
        for i in range(X.shape[0]):
            temp = self.root
            while (
                isinstance(temp, Node)
                and temp.left_child is not None
                and temp.right_child is not None
            ):
                if X[i, temp.feature_index] < temp.value:
                    temp = temp.left_child
                else:
                    temp = temp.right_child
            y_predicted_probs[i] = temp[1]
        return y_predicted_probs

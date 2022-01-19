import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, min_samples_split, max_depth, max_feature_per_split=None):
        self.min_samples_split = min_samples_split
        self.max_feature_per_spit = max_feature_per_split
        self.max_depth = max_depth
        self.node_size = 0
        self.tree = None
        self.n_classes = 0

    def _find_best_split(self, x, y):
        if self.max_feature_per_spit is None:
            max_feature = int(math.sqrt(x.shape[1]))
        else:
            max_feature = self.max_feature_per_spit
        m = y.size
        if m <= 1:
            return None, None
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        gini_impurity = 1 - sum((samples / m) ** 2 for samples in n_samples_per_class)
        selected_features = np.random.choice(x.shape[1], max_feature, False)
        best_feature, best_threshold, best_gain = None, None, -1
        for f in selected_features:
            threshold, classes = zip(*sorted(zip(x[:, f], y)))
            n_left = [0] * self.n_classes
            n_right = n_samples_per_class.copy()
            for i in range(1, m):
                c = classes[i - 1]
                n_left[c] += 1
                n_right[c] -= 1
                gini_impurity_left = 1 - sum((n_left[j] / i) ** 2 for j in range(self.n_classes))
                gini_impurity_right = 1 - sum((n_right[j] / (m - i) ** 2 for j in range(self.n_classes)))
                gain = gini_impurity - (gini_impurity_left * i + gini_impurity_right * (m - i)) / m

                if threshold[i] == threshold[i - 1]:
                    continue

                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = (threshold[i] + threshold[i - 1]) / 2

        return best_feature, best_threshold, best_gain

    def _grow_tree(self, x, y, depth=0):
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(n_samples_per_class)
        node = Node(predicted_class)
        if depth < self.max_depth and x.shape[0] >= self.min_samples_split:
            index, threshold, gain = self._find_best_split(x, y)
            if gain > 0:
                indexes_left = x[:, index] < threshold
                x_left, y_left = x[indexes_left], y[indexes_left]
                x_right, y_right = x[~indexes_left], y[~indexes_left]
                node.feature_index = index
                node.threshold = threshold
                node.left = self._grow_tree(x_left, y_left, depth + 1)
                node.right = self._grow_tree(x_right, y_right, depth + 1)
        return node

    def fit(self, x, y, n_classes):
        self.n_classes = n_classes
        self.tree = self._grow_tree(x, y)

    def predict(self, x):
        if len(x.shape) == 1:
            return self._predict(x)

    def _predict(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


class RandomForest:
    def __init__(self, n_trees, min_samples_split, max_depth, random_state=0):
        np.random.seed(random_state)
        self.min_samples_split = min_samples_split
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.forest = []

    def _sample(self, x, y):
        rows = x.shape[0]
        samples = np.random.choice(rows, rows, True)
        return x[samples], y[samples]

    def fit(self, x, y, n_classes):
        if len(self.forest) > 0:
            self.forest = []

        for n_built in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split, self.max_depth)
            _x, _y = self._sample(x, y)
            tree.fit(_x, _y, n_classes)
            self.forest.append(tree)

    def predict(self, x):
        y = []
        for x_ in x:
            prediction = []
            for tree in self.forest:
                prediction.append(tree.predict(x_))
            y.append(max(set(prediction), key=prediction.count))
        return y


def test(f, n_trees, min_samples_split, max_depth, random_state, repetitions):
    if f == 'bank.csv':
        header = 0
        sep = ';'
    else:
        header = None
        sep = ','
    df = pd.read_csv(f, header=header, sep=sep)
    if f == 'wbcd.csv':
        df.set_index(df.columns[0], inplace=True)
    df.replace('?', pd.NaT, inplace=True)
    df.dropna(inplace=True)
    le = preprocessing.LabelEncoder()
    if f == 'wine.csv':
        correct_cols_order = [i for i in range(1, df.shape[1])]
        correct_cols_order.append(0)
        df = df[correct_cols_order]
    dataset = df.to_numpy()
    for col in range(dataset.shape[1] - 1):
        if not isinstance(dataset[0, col], (int, float, complex)):
            dataset[:, col] = le.fit_transform(dataset[:, col])

    dataset[:, -1] = le.fit_transform(dataset[:, -1])
    print('----', f, n_trees, '----')
    my_results, sk_results = [], []
    for rep in range(repetitions):
        x, x_test, y, y_test = train_test_split(dataset[:, :-1], dataset[:, -1], test_size=0.33,
                                                random_state=random_state + rep)

        rf = RandomForest(n_trees=n_trees, min_samples_split=min_samples_split, max_depth=max_depth,
                          random_state=random_state + rep)
        rf.fit(x, y.astype('int'), len(set(dataset[:, -1])))
        rf_preds = rf.predict(x_test)

        sk_model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=random_state + rep,
                                          min_samples_split=min_samples_split)
        sk_model.fit(x, y.astype('int'))
        sk_preds = sk_model.predict(x_test)

        my_results.append(accuracy_score(y_test.astype('int'), rf_preds))
        sk_results.append(accuracy_score(y_test.astype('int'), sk_preds))
    return my_results, sk_results


if __name__ == '__main__':
    filename = ['wine.csv', 'bank.csv', 'wbcd.csv']
    for f in filename:
        m_r, s_r = test(f, n_trees=200, min_samples_split=2, max_depth=40, random_state=0, repetitions=10)
        print('Mean accuracy: %.4f' % np.mean(m_r), ' %.4f' % np.mean(s_r))

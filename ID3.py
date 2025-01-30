from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from collections import Counter
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import pandas as pd
import numpy as np


class DecisionTree:
    def __init__(self):
        self.tree = None
        self._most_common_class = ""  # Najczęściej występująca klasa w zbiorze danych

    def create_tree(self, X_train, y_train):
        self._most_common_class = self._get_most_common_class(y_train)
        data = X_train.copy()
        data['Class'] = y_train
        attributes = list(X_train.columns)
        self.tree = self._build_tree(data, attributes, 'Class')

    def _build_tree(self, data, attributes, target_col):
        if len(set(data[target_col])) == 1:
            return data[target_col].iloc[0]

        # Jeżeli brak atrybutów do podziału, zwracam najczęściej występującą klasę
        if len(attributes) == 0:
            return data[target_col].mode()[0]

        best_attr = self._best_attribute(data, attributes, target_col)
        tree = {best_attr: {}}

        remaining_attributes = [attr for attr in attributes if attr != best_attr]

        for value in data[best_attr].unique():
            subset = data[data[best_attr] == value]
            tree[best_attr][value] = self._build_tree(subset, remaining_attributes, target_col)

        return tree

    def _best_attribute(self, dataOfParent, attributes, target_col):
        best_gain = -np.inf
        best_attr = None

        for attribute in attributes:
            gain = information_gain(dataOfParent, attribute, target_col)
            if gain > best_gain:
                best_gain = gain
                best_attr = attribute

        return best_attr

    def predict(self, X_test):
        predictions = []
        for _, row in X_test.iterrows():
            prediction = self._predict_single(row)
            predictions.append(prediction)

        return pd.Series(predictions)

    def _predict_single(self, sample):
        return self._prediction(self.tree, sample)

    def _prediction(self, tree, sample):
        if not isinstance(tree, dict):
            return tree

        attribute = list(tree.keys())[0]
        attribute_value = sample[attribute]

        # Jeżeli wartość atrybutu nie występuje w drzewie, zwracamy najczęściej występującą klasę
        if attribute_value not in tree[attribute]:
            return self._most_common_class

        return self._prediction(tree[attribute][attribute_value], sample)

    def _get_most_common_class(self, target_data):
        most_common_class = target_data.mode()[0]
        return most_common_class


def information_gain(data, attribute, target_col):
    total_entropy = entropy(data[target_col])  # Liczy entropie przed podziałem (entropia rodzica)
    values = data[attribute].unique()
    children_entropy = 0

    # Obliczanie entropii po podziale
    for value in values:
        subset = data[data[attribute] == value]  # Podzbiór danych dla wartości atrybutu
        children_entropy += (len(subset) / len(data)) * entropy(subset[target_col])

    return total_entropy - children_entropy


def entropy(attributes):
    classes_counter = Counter(atr for atr in attributes)
    proportions_of_classes = (i / len(attributes) for i in classes_counter.values())
    return -sum(p * log2(p) for p in proportions_of_classes)


def preprocess_data(data, target_data, state):
    X_train, X_test, y_train, y_test = train_test_split(data, target_data, test_size=0.4, random_state=state)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    return X_train, X_test, y_train, y_test


def main():
    dataframe = fetch_ucirepo(id=19)
    # data (as pandas dataframes)
    X = dataframe.data.features
    y = dataframe.data.targets
    results = np.array([])
    for i in range(10):
        X_train, X_test, y_train, y_test = preprocess_data(X, y, i)
        tree = DecisionTree()
        tree.create_tree(X_train, y_train)
        predictions = tree.predict(X_test)
        conMatrix = confusion_matrix(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        results = np.append(results, accuracy)

    print(np.mean(results))
    print(conMatrix)

if __name__ == "__main__":
    main()
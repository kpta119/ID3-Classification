from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from collections import Counter
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class DecisionTree:
    def __init__(self):
        self.tree = None  # Drzewo decyzyjne, początkowo None
        self._most_common_class = ""  # Najczęściej występująca klasa

    def fit(self, X_train, y_train):
        # Łączenie cech (X_train) i etykiety (y_train) w jeden DataFrame
        self._most_common_class = self._get_most_common_class(y_train)
        data = X_train.copy()
        data['Class'] = y_train  # Łączymy cechy i etykiety w jeden zbiór
        attributes = list(X_train.columns)  # Lista dostępnych atrybutów
        self.tree = self._build_tree(data, attributes, 'Class')

    def _build_tree(self, data, attributes, target_col):
        # Jeżeli wszystkie dane mają tę samą klasę, kończymy
        if len(set(data[target_col])) == 1:
            return data[target_col].iloc[0]

        # Jeżeli brak atrybutów do podziału, przypisujemy większość klas
        if len(attributes) == 0:
            return data[target_col].mode()[0]

        # Wybieramy najlepszy atrybut
        best_attr = self._best_attribute(data, attributes, target_col)

        # Tworzymy węzeł
        tree = {best_attr: {}}

        # Usuwamy najlepszy atrybut z listy atrybutów
        remaining_attributes = [attr for attr in attributes if attr != best_attr]

        # Dzielimy dane według wartości najlepszego atrybutu
        for value in data[best_attr].unique():
            subset = data[data[best_attr] == value]
            tree[best_attr][value] = self._build_tree(subset, remaining_attributes, target_col)

        return tree

    def _best_attribute(self, data, attributes, target_col):
        """
        Wybiera najlepszy atrybut do podziału na podstawie największego informacyjnego zysku.

        :param data: Dane wejściowe (cechy + etykieta)
        :param attributes: Lista dostępnych atrybutów
        :param target_col: Kolumna docelowa (klasa)
        :return: Najlepszy atrybut
        """
        best_gain = -float("inf")
        best_attr = None

        for attribute in attributes:
            gain = information_gain(data, attribute, target_col)
            if gain > best_gain:
                best_gain = gain
                best_attr = attribute

        return best_attr

    def predict(self, X_test):
        predictions = []  # Lista, która przechowa przewidywane klasy

        # Iterujemy po każdym wierszu w X_test
        for _, row in X_test.iterrows():
            prediction = self._predict_single(row)  # Wywołujemy metodę _predict_single dla każdego wiersza
            predictions.append(prediction)  # Dodajemy przewidywaną klasę do listy

        return pd.Series(predictions)  # Zwracamy przewidywania jako Series

    def _predict_single(self, sample):
        return self._prediction(self.tree, sample)

    def _prediction(self, tree, sample):
        if not isinstance(tree, dict):  # Jeżeli węzeł jest liściem, zwróć klasę
            return tree

        # Pobieramy atrybut testowany w bieżącym węźle
        attribute = list(tree.keys())[0]
        attribute_value = sample[attribute]

        # Jeżeli wartość atrybutu nie występuje w drzewie, zwracamy najczęściej występującą klasę
        if attribute_value not in tree[attribute]:
            return self._most_common_class
        # Przechodzimy do odpowiedniego poddrzewa
        return self._prediction(tree[attribute][attribute_value], sample)

    def _get_most_common_class(self, target_data):
        most_common_class = target_data.mode()[0]
        return most_common_class



def information_gain(data, attribute, target_col):
    """Oblicza informacyjny zysk dla danego atrybutu."""
    total_entropy = entropy(data[target_col])  # Entropia przed podziałem
    values = data[attribute].unique()  # Unikalne wartości atrybutu
    weighted_entropy = 0

    # Obliczanie entropii po podziale
    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target_col])

    return total_entropy - weighted_entropy  # Zysk informacyjny

def entropy(atrributes) -> float:
    #Calculates the entropy of a dataset.

    classes_counter = Counter(atr for atr in atrributes)
    proportions_of_classes = (i / len(atrributes) for i in classes_counter.values())
    return -sum(ep * log2(ep) for ep in proportions_of_classes)




def preprocess_data(data, target_data):

    X_train, X_test, y_train, y_test = train_test_split(data, target_data, test_size=0.4)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    return X_train, X_test, y_train, y_test

def main():
    breast_cancer = fetch_ucirepo(id=14)
    mushroom = fetch_ucirepo(id=73)
    # data (as pandas dataframes)
    X = mushroom.data.features
    y = mushroom.data.targets
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    tree = DecisionTree()
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    # Ocena modelu
    cm = confusion_matrix(y_test, predictions)
    print(f"Macierz pomyłek:\n{cm}")
    print(f"Dokładność: {accuracy_score(y_test, predictions)}")

if __name__ == "__main__":
    main()
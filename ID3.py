from ucimlrepo import fetch_ucirepo
import numpy as np
from collections import Counter
from math import log2

breast_cancer = fetch_ucirepo(id=14)

# data (as pandas dataframes)
X = breast_cancer.data.features
y = breast_cancer.data.targets
y= np.array(y)
if len(y.shape) > 1:
    y = y.ravel()


def entropy(pairs) -> float:
    """Calculates the entropy of a dataset.

    The argument can be either a Counter instance for the classifications
    in a particular set; or the 'raw' set itself (in this case Counter object
    is created automatically).
    """
    classes_counter = Counter(str(pair) for pair in pairs)
    proportions_of_classes = (i / len(pairs) for i in classes_counter.values())
    return -sum(ep * log2(ep) for ep in proportions_of_classes)


print("Entropia:", entropy(y))

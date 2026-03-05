# models/classifiers.py
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_classifier(model_type: str, random_state: int = 42):
    """Return a classifier instance given a model name."""
    model_type = model_type.lower()

    classifiers = {
        "svm": SVC(
            kernel="linear",
            C=1,
            class_weight="balanced",
            gamma="auto",
            tol=1e-3,
            random_state=random_state,
        ),
        "svm_rf": SVC(
            kernel="rbf",
            C=1,
            gamma="auto",
            tol=1e-3,
            random_state=random_state,
        ),
        "logreg": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "lda": LinearDiscriminantAnalysis(),
        "ridge": RidgeClassifier(random_state=random_state),
        "perceptron": Perceptron(max_iter=1000, random_state=random_state),
        "qda": QuadraticDiscriminantAnalysis(),
        "gnb": GaussianNB(),
        "dtree": DecisionTreeClassifier(random_state=random_state),
        "rf": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    if model_type not in classifiers:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(classifiers.keys())}"
        )

    return classifiers[model_type]

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


class _StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    scaler = _StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    Xt = scaler.transform(X_test)

    n, d = Xs.shape
    weights = np.zeros(d)
    bias = 0.0
    lr = data.learning_rate
    loss_history = []
    for _ in range(data.epochs):
        pred = Xs @ weights + bias
        error = pred - y_train
        loss_history.append(float(np.mean(error ** 2)))
        grad_w = (2.0 / n) * (Xs.T @ error)
        grad_b = (2.0 / n) * error.sum()
        weights -= lr * grad_w
        bias -= lr * grad_b

    predictions = Xt @ weights + bias
    return RegressionResult(weights=weights, bias=float(bias), loss_history=loss_history, predictions=predictions)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    scaler = _StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    Xt = scaler.transform(X_test)

    n, d = Xs.shape
    weights = np.zeros(d)
    bias = 0.0
    lr = data.learning_rate
    eps = 1e-12
    loss_history = []
    for _ in range(data.epochs):
        z = Xs @ weights + bias
        pred = 1.0 / (1.0 + np.exp(-z))
        pred = np.clip(pred, eps, 1 - eps)
        loss_history.append(float(-np.mean(y_train * np.log(pred) + (1 - y_train) * np.log(1 - pred))))
        error = pred - y_train
        grad_w = (1.0 / n) * (Xs.T @ error)
        grad_b = (1.0 / n) * error.sum()
        weights -= lr * grad_w
        bias -= lr * grad_b

    zt = Xt @ weights + bias
    probabilities = 1.0 / (1.0 + np.exp(-zt))
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def _metrics(y_true, prob, pred) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, prob)),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred)),
        recall=float(recall_score(y_true, pred)),
        f1=float(f1_score(y_true, pred)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        results[name] = _metrics(y_test, prob, pred)

    return BaselineComparisonResult(**results)

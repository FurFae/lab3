"""Проверка lab3: подготовка данных, запуск контрактных функций, метрики."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models

# --- Линейная регрессия: Student_Performance.csv ---
sp = pd.read_csv("Student_Performance.csv")
print(sp.columns.tolist())
y = sp["Performance Index"].to_numpy(float)
X = sp.drop(columns=["Performance Index"]).copy()
X["Extracurricular Activities"] = (X["Extracurricular Activities"] == "Yes").astype(float)
X = X.to_numpy(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

res = train_linear_regression(RegressionInput(Xtr, ytr, Xte, yte))
mse_test = mean_squared_error(yte, res.predictions)
print("final train loss:", res.loss_history[-1], "| test MSE:", mse_test, "| bias:", res.bias)

# --- Логистическая регрессия: german.csv ---
g = pd.read_csv("german.csv", sep=";")
if g.shape[1] == 1:  # нет заголовка — читаем заново
    g = pd.read_csv("german.csv", sep=";", header=None)
print("german shape:", g.shape)
y = g.iloc[:, 0].to_numpy(int)
X = g.iloc[:, 1:].to_numpy(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

cres = train_logistic_classifier(ClassificationInput(Xtr, ytr, Xte, yte))
print("final logloss:", cres.loss_history[-1])
own = {
    "roc_auc": roc_auc_score(yte, cres.probabilities),
    "accuracy": accuracy_score(yte, cres.predictions),
    "precision": precision_score(yte, cres.predictions),
    "recall": recall_score(yte, cres.predictions),
    "f1": f1_score(yte, cres.predictions),
}
print("own logistic:", own)

comp = compare_classical_models(ClassificationInput(Xtr, ytr, Xte, yte))
for name in ["logistic_regression", "decision_tree", "knn"]:
    m = getattr(comp, name)
    print(name, m)

results = {
    "linear_regression": {"test_mse": mse_test},
    "classification": {
        "own_logistic": own,
        "logistic_regression": comp.logistic_regression.__dict__,
        "decision_tree": comp.decision_tree.__dict__,
        "knn": comp.knn.__dict__,
    },
}
with open("experiment_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("LAB3 OK")

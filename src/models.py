"""
models.py
=========
Entrenamiento y evaluación de modelos de clasificación supervisada.

Implementa cuatro algoritmos:
  - Random Forest
  - Regresión Logística
  - Naive Bayes (Gaussiano)
  - Árbol de Decisión

Validación con TimeSeriesSplit (respeta el orden temporal).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)


# Catálogo de modelos disponibles en la app
MODEL_REGISTRY = {
    'Random Forest': lambda params: RandomForestClassifier(
        n_estimators=params.get('n_estimators', 200),
        max_depth=params.get('max_depth', 8),
        min_samples_leaf=params.get('min_samples_leaf', 5),
        random_state=42,
        n_jobs=-1
    ),
    'Regresión Logística': lambda params: LogisticRegression(
        C=params.get('C', 1.0),
        max_iter=1000,
        random_state=42
    ),
    'Naive Bayes': lambda params: GaussianNB(),
    'Árbol de Decisión': lambda params: DecisionTreeClassifier(
        max_depth=params.get('max_depth', 6),
        min_samples_leaf=params.get('min_samples_leaf', 10),
        random_state=42
    ),
}


def time_split(X, y, train_size=0.8):
    """Split temporal simple: primer N% para train, resto para test."""
    n = len(X)
    split = int(n * train_size)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train_and_evaluate(model_name, params, X_train, X_test, y_train, y_test, standardize=False):
    """
    Entrena un modelo, predice en train y test, devuelve un dict con métricas
    y artefactos (matriz de confusión, curva ROC, importancia, predicciones).
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconocido: {model_name}")

    # Estandarización opcional (recomendable para LogReg)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train_proc = pd.DataFrame(scaler.fit_transform(X_train),
                                    index=X_train.index, columns=X_train.columns)
        X_test_proc = pd.DataFrame(scaler.transform(X_test),
                                   index=X_test.index, columns=X_test.columns)
    else:
        X_train_proc, X_test_proc = X_train, X_test

    model = MODEL_REGISTRY[model_name](params)
    model.fit(X_train_proc, y_train)

    # Predicciones
    y_pred_train = model.predict(X_train_proc)
    y_pred_test = model.predict(X_test_proc)
    y_proba_test = model.predict_proba(X_test_proc)[:, 1]
    y_proba_train = model.predict_proba(X_train_proc)[:, 1]

    # Métricas
    metrics_train = _compute_metrics(y_train, y_pred_train, y_proba_train)
    metrics_test = _compute_metrics(y_test, y_pred_test, y_proba_test)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)

    # Importancia/coeficientes
    importance = _get_importance(model, X_train.columns.tolist())

    return {
        'model': model,
        'scaler': scaler,
        'metrics_train': metrics_train,
        'metrics_test': metrics_test,
        'y_pred_test': pd.Series(y_pred_test, index=y_test.index),
        'y_proba_test': pd.Series(y_proba_test, index=y_test.index),
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr),
        'importance': importance,
        'features': X_train.columns.tolist(),
    }


def _compute_metrics(y_true, y_pred, y_proba):
    """Calcula el set estándar de métricas de clasificación."""
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1':        f1_score(y_true, y_pred, zero_division=0),
        'AUC':       roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan,
    }


def _get_importance(model, feature_names):
    """Extrae importancia (RF/DT) o coeficientes absolutos (LogReg). NB no tiene."""
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    elif hasattr(model, 'coef_'):
        return pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)
    else:
        return None


def cross_validate_temporal(model_name, params, X, y, n_splits=5, standardize=False):
    """
    Validación cruzada con TimeSeriesSplit.
    Devuelve la media y desviación estándar de cada métrica a lo largo de los folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if standardize:
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
            X_te = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)

        model = MODEL_REGISTRY[model_name](params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        fold_metrics.append(_compute_metrics(y_te, y_pred, y_proba))

    df_metrics = pd.DataFrame(fold_metrics)
    return df_metrics.mean(), df_metrics.std()


def compare_all_models(X_train, X_test, y_train, y_test, standardize_logreg=True):
    """
    Entrena los 4 modelos con configuración por defecto, devuelve un DataFrame
    comparativo de métricas en test y el resultado completo de cada uno.
    """
    results = {}
    for model_name in MODEL_REGISTRY.keys():
        # LogReg y Naive Bayes se benefician de estandarización
        std = standardize_logreg and model_name in ('Regresión Logística', 'Naive Bayes')
        result = train_and_evaluate(model_name, {}, X_train, X_test, y_train, y_test, standardize=std)
        results[model_name] = result

    # Tabla comparativa (métricas en test)
    df_comp = pd.DataFrame({
        name: r['metrics_test'] for name, r in results.items()
    }).T

    return df_comp, results


def pick_best_model(df_comp, criterion='F1'):
    """Devuelve el nombre del mejor modelo según el criterio."""
    return df_comp[criterion].idxmax()

# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
import gzip
import json
import os
import pickle
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# Paso 1. Limpieza de datos
# ---------------------------------------------------------------------------
def _load_and_clean(zip_path):
    """Carga un CSV comprimido en zip y aplica el proceso de limpieza."""
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    # Renombrar columna objetivo
    df = df.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    df = df.drop(columns=["ID"])

    # Eliminar registros con informacion no disponible (EDUCATION=0, MARRIAGE=0)
    df = df[df["EDUCATION"] != 0]
    df = df[df["MARRIAGE"] != 0]

    # Agrupar EDUCATION > 4 en la categoria "others" (4)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


# ---------------------------------------------------------------------------
# Paso 2. Division de datasets
# ---------------------------------------------------------------------------
def _split_features_target(df):
    """Separa features y target."""
    x = df.drop(columns=["default"])
    y = df["default"]
    return x, y


# ---------------------------------------------------------------------------
# Pasos 3-4. Pipeline + GridSearchCV
# ---------------------------------------------------------------------------
def _build_model():
    """Construye el pipeline y el GridSearchCV."""
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [200],
        "classifier__max_depth": [28],
        "classifier__min_samples_leaf": [2],
    }

    model = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    return model


# ---------------------------------------------------------------------------
# Paso 5. Guardar modelo
# ---------------------------------------------------------------------------
def _save_model(model):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)


# ---------------------------------------------------------------------------
# Pasos 6-7. Calcular metricas y guardar
# ---------------------------------------------------------------------------
def _compute_and_save_metrics(model, x_train, y_train, x_test, y_test):
    os.makedirs("files/output", exist_ok=True)

    metrics = []

    for dataset_name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        y_pred = model.predict(x)

        metrics.append(
            {
                "type": "metrics",
                "dataset": dataset_name,
                "precision": precision_score(y, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1_score": f1_score(y, y_pred, zero_division=0),
            }
        )

    for dataset_name, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)

        metrics.append(
            {
                "type": "cm_matrix",
                "dataset": dataset_name,
                "true_0": {
                    "predicted_0": int(cm[0, 0]),
                    "predicted_1": int(cm[0, 1]),
                },
                "true_1": {
                    "predicted_0": int(cm[1, 0]),
                    "predicted_1": int(cm[1, 1]),
                },
            }
        )

    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for entry in metrics:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Ejecucion principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Paso 1-2: Cargar, limpiar y dividir
    train_df = _load_and_clean("files/input/train_data.csv.zip")
    test_df = _load_and_clean("files/input/test_data.csv.zip")

    x_train, y_train = _split_features_target(train_df)
    x_test, y_test = _split_features_target(test_df)

    # Pasos 3-4: Construir y entrenar modelo con GridSearchCV
    model = _build_model()
    model.fit(x_train, y_train)

    # Paso 5: Guardar modelo
    _save_model(model)

    # Pasos 6-7: Calcular y guardar metricas
    _compute_and_save_metrics(model, x_train, y_train, x_test, y_test)

    print("Done. Model saved to files/models/model.pkl.gz")
    print("Metrics saved to files/output/metrics.json")
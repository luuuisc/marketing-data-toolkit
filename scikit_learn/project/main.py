#!/usr/bin/env python3
# main.py
# Requisitos: pandas, numpy, scikit-learn, joblib

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_and_train(df, test_size, model_out, metrics_out):
    # Separar features y etiqueta
    X = df.drop(columns=['customer_id', 'label'])
    y = df['label']

    # Columnas numéricas y categóricas
    num_features = ['recency', 'frequency', 'monetary', 'open_rate', 'ctr', 'age']
    cat_features = ['gender', 'region']

    # Pipeline de preprocesamiento
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

    # Pipeline completo
    model = Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(solver='liblinear'))
    ])

    # Dividir train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    # Entrenamiento
    model.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['roc_auc'] = ''
    report_df.loc['overall', 'roc_auc'] = roc_auc_score(y_test, y_proba)

    # Guardar métricas y modelo
    report_df.to_csv(metrics_out, index=True)
    joblib.dump(model, model_out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   required=True,
                        help="Ruta al CSV de historial de campaña")
    parser.add_argument('--test_size',   type=float, default=0.2,
                        help="Proporción del set de prueba")
    parser.add_argument('--model_out',   required=True,
                        help="Ruta donde guardar el modelo (.pkl)")
    parser.add_argument('--metrics_out', required=True,
                        help="Ruta donde guardar las métricas (.csv)")
    args = parser.parse_args()

    # Crear carpetas de salida si no existen
    os.makedirs(os.path.dirname(args.model_out),   exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    df = load_data(args.data_path)
    preprocess_and_train(df, args.test_size,
                         args.model_out, args.metrics_out)
    print("Entrenamiento completado. Modelo y métricas guardados.")

if __name__ == '__main__':
    main()
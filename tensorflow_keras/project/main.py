#!/usr/bin/env python3
# main.py
# Requisitos: tensorflow, pandas, numpy, scikit-learn, matplotlib

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf

def load_data(path):
    return pd.read_csv(path)

def build_preprocessor():
    num_features = ['recency', 'frequency', 'monetary', 'open_rate', 'ctr', 'age']
    cat_features = ['gender', 'region']
    return ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

def build_model(input_dim, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_and_evaluate(df, args):
    # separar X, y
    X = df.drop(columns=['customer_id', 'label'])
    y = df['label'].values

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42)

    # preprocesamiento
    preproc = build_preprocessor()
    X_train_proc = preproc.fit_transform(X_train)
    X_test_proc  = preproc.transform(X_test)

    # crear y entrenar modelo
    model = build_model(X_train_proc.shape[1], args.lr)
    history = model.fit(
        X_train_proc, y_train,
        validation_data=(X_test_proc, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2
    )

    # guardar modelo
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)

    # curvas de entrenamiento
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.history_out), exist_ok=True)
    plt.savefig(args.history_out)
    plt.close()

    # evaluación final
    y_proba = model.predict(X_test_proc).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['roc_auc'] = ''
    report_df.loc['1', 'roc_auc'] = roc_auc_score(y_test, y_proba)

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    report_df.to_csv(args.metrics_out, index=True)

    print("Modelo, métricas y gráficas de historial guardados.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',    required=True,
                        help="CSV con datos de campaña")
    parser.add_argument('--test_size',    type=float, default=0.2,
                        help="Tamaño del set de prueba")
    parser.add_argument('--model_out',    required=True,
                        help="Ruta para guardar el modelo (.h5)")
    parser.add_argument('--metrics_out',  required=True,
                        help="CSV con métricas de evaluación")
    parser.add_argument('--history_out',  required=True,
                        help="PNG con curvas de entrenamiento")
    parser.add_argument('--epochs',       type=int, default=20,
                        help="Número de épocas")
    parser.add_argument('--batch_size',   type=int, default=32,
                        help="Tamaño de batch")
    parser.add_argument('--lr',           type=float, default=1e-3,
                        help="Learning rate")
    args = parser.parse_args()

    df = load_data(args.data_path)
    train_and_evaluate(df, args)

if __name__ == '__main__':
    main()
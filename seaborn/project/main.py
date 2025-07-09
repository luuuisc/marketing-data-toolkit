#!/usr/bin/env python3
# main.py
# Requisitos: pandas, seaborn, matplotlib

import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path)

def plot_histogram(df, out_dir):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['open_rate'], bins=20, kde=True)
    plt.title('Distribución de Open Rate')
    plt.xlabel('Open Rate')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dist_open_rate.png'))
    plt.close()

def plot_box_ctr_by_region(df, out_dir):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='region', y='ctr', data=df)
    plt.title('CTR por Región')
    plt.xlabel('Región')
    plt.ylabel('CTR')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'box_ctr_by_region.png'))
    plt.close()

def plot_heatmap(df, out_dir):
    numeric = df[['recency', 'frequency', 'monetary', 'open_rate', 'ctr', 'age']]
    corr = numeric.corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Mapa de Correlaciones')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'heatmap_correlation.png'))
    plt.close()

def plot_pairplot(df, out_dir):
    sns.pairplot(
        df[['recency','frequency','monetary','open_rate','ctr','age','label']],
        hue='label',
        diag_kind='hist',
        corner=True
    )
    plt.suptitle('Pairplot de Variables Numéricas por Conversión', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pairplot_features.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True,
                        help='Ruta al CSV de datos')
    parser.add_argument('--out_dir',   required=True,
                        help='Carpeta de salida para las gráficas')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_data(args.data_path)

    plot_histogram(df, args.out_dir)
    plot_box_ctr_by_region(df, args.out_dir)
    plot_heatmap(df, args.out_dir)
    plot_pairplot(df, args.out_dir)

    print(f'Visualizaciones guardadas en {args.out_dir}')

if __name__ == '__main__':
    main()
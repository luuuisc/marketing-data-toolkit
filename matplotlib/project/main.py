#!/usr/bin/env python3
# main.py
# Requisitos: pandas, matplotlib

import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Carga el CSV y calcula las métricas derivadas.
    """
    df = pd.read_csv(csv_path, parse_dates=['week'] + [], dtype={
        'emails_sent': int,
        'opens': int,
        'clicks': int,
        'conversions': int
    })
    
    # Asegurarnos de ordenar por fecha
    df = df.sort_values('week')
    
    # Calcular métricas clave
    df['open_rate'] = df['opens'] / df['emails_sent']
    df['ctr'] = df['clicks'] / df['opens']
    df['conversion_rate'] = df['conversions'] / df['clicks']
    
    return df

def plot_campaign_performance(df: pd.DataFrame, output_path: str):
    """
    Genera un gráfico combinado de barras y líneas con las métricas de la campaña,
    ajustando las etiquetas del eje X para que no se amontonen.
    """
    # 1) Prepara las etiquetas de semana
    weeks = df['week'].dt.strftime('%Y-%W')

    # 2) Crea la figura y el primer eje
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 3) Dibuja las barras de emails enviados
    ax1.bar(weeks, df['emails_sent'], alpha=0.3, label='Emails enviados')
    ax1.set_xlabel('Semana')
    ax1.set_ylabel('Emails enviados')

    # 4) Ajusta las etiquetas del eje X
    ax1.set_xticks(range(len(weeks)))
    ax1.set_xticklabels(weeks, rotation=45, ha='right', fontsize=8)

    # 5) Dibuja el segundo eje con las tasas
    ax2 = ax1.twinx()
    ax2.plot(weeks, df['open_rate'],        marker='o', label='Tasa apertura')
    ax2.plot(weeks, df['ctr'],              marker='o', label='CTR')
    ax2.plot(weeks, df['conversion_rate'],  marker='o', label='Tasa conversión')
    ax2.set_ylabel('Tasas (%)')

    # 6) Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # 7) Ajuste de márgenes y guardado
    plt.title('Rendimiento de campaña de Email Marketing por semana')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f'Gráfico guardado en {output_path}')

def main():
    # Ruta al CSV de datos
    data_file = os.path.join(os.path.dirname(__file__), 'data.csv')
    if not os.path.exists(data_file):
        print(f'ERROR: no encontré el archivo {data_file}')
        return
    
    # Cargar y procesar datos
    df = load_data(data_file)
    
    # Directorio de salida
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    # Generar gráfico
    plot_campaign_performance(df, os.path.join(out_dir, 'campaign_performance.png'))

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# segmentation.py
# Requisitos: pandas

import os
import pandas as pd
from datetime import datetime

def load_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV de transacciones y convierte la fecha.
    """
    df = pd.read_csv(path, parse_dates=['transaction_date'])
    return df

def compute_rfm(df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
    """
    Calcula Recencia (días desde última compra), Frecuencia (número de compras)
    y Monetario (gasto total) por cliente.
    """
    # Última compra → recencia
    recency = df.groupby('customer_id')['transaction_date'] \
                .max() \
                .apply(lambda d: (reference_date - d).days)
    # Frecuencia → contador de transacciones
    frequency = df.groupby('customer_id').size()
    # Monetario → suma de amounts
    monetary = df.groupby('customer_id')['amount'].sum()

    rfm = pd.DataFrame({
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary
    })
    return rfm

def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna puntajes 1-4 a cada métrica usando cuartiles:
    - R: los clientes más recientes → 4, los más antiguos → 1
    - F y M: los de mayor frecuencia/monetario → 4, los menores → 1
    """
    # Recency: invertimos las etiquetas para que menor recency → mayor score
    rfm['R_score'] = pd.qcut(rfm['recency'], 4, labels=[4, 3, 2, 1]).astype(int)
    # Frequency y Monetary: mayor valor → mayor score
    rfm['F_score'] = pd.qcut(rfm['frequency'], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'], 4, labels=[1, 2, 3, 4]).astype(int)

    # Combinamos en un solo RFM Score tipo "4-3-2"
    rfm['RFM_Score'] = (
        rfm['R_score'].astype(str) + '-' +
        rfm['F_score'].astype(str) + '-' +
        rfm['M_score'].astype(str)
    )
    return rfm

def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica clientes en segmentos básicos de negocio según sus RFM Scores.
    """
    segments = []
    for _, row in rfm.iterrows():
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        if r == 4 and f == 4 and m == 4:
            seg = 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            seg = 'Loyal Customers'
        elif r == 2 and f >= 2:
            seg = 'Potential Loyalist'
        elif r == 1:
            seg = 'At Risk'
        else:
            seg = 'Others'
        segments.append(seg)
    rfm['segment'] = segments
    return rfm

def export_results(rfm: pd.DataFrame, out_dir: str):
    """
    Guarda la tabla RFM completa y el resumen por segmento en CSVs.
    """
    # Tabla detallada
    rfm_table = rfm.reset_index()
    rfm_table.to_csv(os.path.join(out_dir, 'rfm_table.csv'), index=False)

    # Resumen por segmento
    summary = rfm_table.groupby('segment').agg(
        customers=('customer_id', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean')
    ).reset_index()
    summary.to_csv(os.path.join(out_dir, 'segment_summary.csv'), index=False)

def main():
    # Ruta a datos y directorio de salida
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, 'data', 'data.csv')
    out_dir   = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Cargar datos
    df = load_data(data_path)

    # Fecha de referencia (por defecto hoy)
    ref_date = datetime.today()

    # Pipeline RFM
    rfm       = compute_rfm(df, ref_date)
    rfm_scored = score_rfm(rfm)
    rfm_seg    = assign_segments(rfm_scored)

    # Guardar resultados
    export_results(rfm_seg, out_dir)

    print(f'RFM analysis complete. Outputs in {out_dir}')

if __name__ == '__main__':
    main()
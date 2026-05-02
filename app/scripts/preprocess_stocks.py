import pandas as pd
import numpy as np
from pathlib import Path

def load_raw_data(file_path):
    """Carga los datos crudos desde un archivo Parquet."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    print(f"Cargando datos desde {file_path}...")
    df = pd.read_parquet(file_path)
    return df

def basic_info(df):
    """Muestra información básica del DataFrame."""
    print("\n--- Información del Dataset ---")
    print(f"Número de filas: {len(df)}")
    print(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    print(f"Tickers únicos: {df['ticker'].nunique()}")
    print("\nConteo de datos por ticker (primeros 5):")
    print(df['ticker'].value_counts().head())
    
    # Verificar si hay nulos
    nulos = df.isnull().sum().sum()
    print(f"\nValores nulos totales: {nulos}")

def calculate_spread(df):
    """
    Calcula un proxy del bid-ask spread.
    Utiliza el High-Low spread normalizado por el precio de cierre.
    """
    # Spread absoluto: High - Low
    df['abs_spread'] = df['high'] - df['low']
    
    # Spread relativo: (High - Low) / Close
    df['rel_spread'] = (df['high'] - df['low']) / df['close']
    
    return df

def calculate_trading_costs(df):
    """
    Calcula los costos de transacción estimados.
    Se asume que el costo de entrar o salir de una posición (one-way) 
    es aproximadamente la mitad del spread relativo.
    """
    df['trading_costs'] = df['rel_spread'] / 2
    return df

def calculate_log_returns(df):
    """
    Calcula los retornos logarítmicos agrupando por ticker.
    log_return = log(Price_t / Price_t-1)
    """
    df['log_return'] = df.groupby('ticker')['close'].transform(lambda x: np.log(x / x.shift(1)))
    return df

def preprocess(df):
    """
    Espacio para aplicar transformaciones:
    - Cálculo de retornos
    - Manejo de outliers
    - Normalización
    """
    print("\nIniciando preprocesamiento...")
    
    # 1. Formateo básico y Ordenamiento (Crítico para retornos y tiempo)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    
    # 2. Cálculo de Spread
    df = calculate_spread(df)
    
    # 3. Cálculo de Costos de Trading
    df = calculate_trading_costs(df)
    
    # 4. Cálculo de Retornos Logarítmicos
    df = calculate_log_returns(df)
    
    # 5. Limpiar nulos resultantes del shift (el primer día de cada ticker no tendrá retorno)
    df = df.dropna(subset=['log_return'])
    
    return df




if __name__ == "__main__":
    raw_path = "app/data/raw/stocks.parquet"
    
    try:
        # 1. Cargar
        df_raw = load_raw_data(raw_path)
        
        # 2. Analizar
        basic_info(df_raw)
        
        # 3. Preprocesar
        df_clean = preprocess(df_raw)
        
        # 4. Mostrar resultado
        print("\nPreprocesamiento completado.")
        print(f"Filas finales (sin nulos de retornos): {len(df_clean)}")
        print(df_clean.head())
        
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")

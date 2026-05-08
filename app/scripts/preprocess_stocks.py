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

def calculate_volume_log_returns(df):
    """
    Calcula los retornos logarítmicos del volumen.
    volume_prc = log(Volume_t / Volume_t-1)
    """
    df['volume_prc'] = df.groupby('ticker')['volume'].transform(lambda x: np.log(x / x.shift(1)))
    return df

def process_vix(df):
    """
    Calcula los retornos logarítmicos del VIX y los añade como columna.
    Luego elimina el ticker ^VIX del dataset original.
    """
    print("Procesando VIX...")
    # 1. Extraer datos del VIX
    vix_data = df[df['ticker'] == '^VIX'].copy()
    vix_data = vix_data.sort_values('date')
    
    # 2. Calcular log retorno del VIX: ln(VIX_t / VIX_{t-1})
    vix_data['vix_log_return'] = np.log(vix_data['close'] / vix_data['close'].shift(1))
    
    # 3. Quedarse solo con date y el nuevo retorno para el merge
    vix_to_merge = vix_data[['date', 'vix_log_return']]
    
    # 4. Unir con el dataframe original
    df = df.merge(vix_to_merge, on='date', how='left')
    
    # 5. Eliminar el ticker ^VIX del dataframe principal
    df = df[df['ticker'] != '^VIX']
    
    return df

def process_macro_data(df_main, macro_path):
    """
    Carga datos macro, aplica un retenedor de orden cero (ZOH),
    calcula variaciones y curva de tipos, y los une a la tabla principal.
    """
    print("Procesando datos macro (FED)...")
    if not Path(macro_path).exists():
        print(f"Advertencia: No se encontró el archivo macro en {macro_path}")
        return df_main
        
    df_macro = pd.read_parquet(macro_path)
    df_macro['date'] = pd.to_datetime(df_macro['date'])
    df_macro = df_macro.sort_values('date')
    
    # Aplicar retenedor de orden cero (Zero-Order Hold) a los datos crudos
    # Esto asegura que huecos en las tasas originales se llenen antes de calcular deltas
    cols_macro = ['fed_rate', 'yield10k', 'yield2k']
    df_macro[cols_macro] = df_macro[cols_macro].ffill()
    
    # 1. Delta Fed Rate: r_t - r_{t-1}
    df_macro['delta_fed_rate'] = df_macro['fed_rate'] - df_macro['fed_rate'].shift(1)
    
    # 2. Curva de tipos: Yield 10Y - Yield 2Y
    df_macro['yield_curve'] = df_macro['yield10k'] - df_macro['yield2k']
    
    # 3. Seleccionar columnas relevantes para el join
    macro_to_merge = df_macro[['date', 'delta_fed_rate', 'yield_curve', 'fed_rate']]
    
    # 4. Merge con la tabla principal
    df_main = df_main.merge(macro_to_merge, on='date', how='left')
    
    return df_main

def preprocess(df, macro_path="app/data/raw/fed_macro.parquet"):
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
    
    # 2. Procesar VIX antes de calcular otros retornos o filtrar
    df = process_vix(df)
    
    # 3. Procesar datos macro (FED)
    df = process_macro_data(df, macro_path)
    
    # 4. Cálculo de Spread
    df = calculate_spread(df)
    
    # 5. Cálculo de Costos de Trading
    df = calculate_trading_costs(df)
    
    # 6. Cálculo de Retornos Logarítmicos para el resto de tickers
    df = calculate_log_returns(df)
    
    # 7. Cálculo de Log Retornos del Volumen
    df = calculate_volume_log_returns(df)
    
    # 8. Limpiar nulos resultantes del shift e infinitos (por log(0))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['log_return', 'vix_log_return', 'volume_prc', 'delta_fed_rate', 'yield_curve'])
    
    # 9. Renombrar yield_curve a yield_spread
    df = df.rename(columns={'yield_curve': 'yield_spread'})
    
    # 10. Seleccionar columnas finales
    final_cols = [
        'date', 'ticker', 'sector', 'log_return', 
        'volume_prc', 'delta_fed_rate', 'yield_spread', 'vix_log_return'
    ]
    df = df[final_cols]
    
    return df




if __name__ == "__main__":
    raw_path = "app/data/raw/stocks.parquet"
    macro_path = "app/data/raw/fed_macro.parquet"
    
    try:
        # 1. Cargar
        df_raw = load_raw_data(raw_path)
        
        # 2. Analizar
        basic_info(df_raw)
        
        # 3. Preprocesar
        df_clean = preprocess(df_raw, macro_path)
        
        # 4. Guardar resultado
        output_dir = Path("app/data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "data.parquet"
        
        df_clean.to_parquet(output_file, index=False)
        print(f"\nPreprocesamiento completado. Datos guardados en {output_file}")
        print(f"Filas finales (sin nulos): {len(df_clean)}")
        print(df_clean.head())
        
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")

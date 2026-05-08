import pandas as pd
from fredapi import Fred
from pathlib import Path
from datetime import datetime

def download_fed_data_fred(api_key):
    """
    Descarga datos macro desde FRED usando fredapi:
    - DFF: Effective Federal Funds Rate (fed_rate)
    - DGS10: 10-Year Treasury Constant Maturity Rate (yield10k)
    - DGS2: 2-Year Treasury Constant Maturity Rate (yield2k)
    """
    fred = Fred(api_key=api_key)
    start_date = '2000-01-01'
    
    print(f"Descargando datos desde FRED (Series: DFF, DGS10, DGS2) desde {start_date}...")
    
    try:
        # Descargar series individuales
        fed_rate = fred.get_series('DFF', observation_start=start_date)
        yield10k = fred.get_series('DGS10', observation_start=start_date)
        yield2k = fred.get_series('DGS2', observation_start=start_date)
        
        # Combinar en un DataFrame
        df = pd.DataFrame({
            'fed_rate': fed_rate,
            'yield10k': yield10k,
            'yield2k': yield2k
        })
        
        # Resetear índice para tener 'date' como columna
        df.index.name = 'date'
        df = df.reset_index()
        
        # Limpieza básica
        # FRED a veces tiene puntos o NaNs en días festivos para tasas de mercado
        df = df.sort_values('date')
        df = df.ffill() # Llenar huecos de fines de semana/festivos
        
        return df
        
    except Exception as e:
        print(f"Error descargando datos de FRED: {e}")
        return None

def save_macro_data(df):
    """Guarda los datos en un archivo Parquet."""
    if df is None or df.empty:
        print("No hay datos para guardar.")
        return
        
    output_path = Path("app/data/raw/fed_macro.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"Datos macro de FRED guardados exitosamente en {output_path}")
    print(df.tail())

if __name__ == "__main__":
    # API Key proporcionada por el usuario
    API_KEY = "4936e38eef5ebd557cc230069a72ff69"
    
    df_macro = download_fed_data_fred(API_KEY)
    save_macro_data(df_macro)

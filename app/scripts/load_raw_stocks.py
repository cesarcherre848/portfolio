import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from pathlib import Path

def scraping_profesional_sp100():
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        tabla = soup.find('table', {'class': 'wikitable sortable'})
        
        if not tabla:
            raise ValueError("No se encontró la tabla de S&P 100.")
            
        filas = tabla.find_all('tr')
        datos = []
        for fila in filas[1:]:
            celdas = fila.find_all('td')
            if len(celdas) >= 3:
                symbol = celdas[0].text.strip().replace('.', '-')
                datos.append(symbol)
        return datos
    except Exception as e:
        print(f"Error en el scraping: {e}")
        return None

# 1. Obtener lista de activos
tickers = scraping_profesional_sp100()

if tickers:
    print(f"Descargando datos para {len(tickers)} tickers...")
    
    # 2. Descargar datos
    # Definimos el rango de tiempo: desde el 2000 hasta hoy
    from datetime import datetime
    hoy = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Descargando datos desde 2000-01-01 hasta {hoy}...")
    data = yf.download(
        tickers=tickers + ["^IRX"], 
        start="2000-01-01",
        end=hoy,
        interval="1d",
        auto_adjust=True
    )

    # 3. Filtrado estricto: Asegurar datos completos en todo el rango
    # Para un rango tan largo, es muy probable que muchos tickers no existieran en el 2000.
    # Eliminamos tickers que tengan nulos para asegurar series temporales perfectas.
    columnas_con_nan = data.columns[data.isna().any()]
    tickers_incompletos = columnas_con_nan.get_level_values(1).unique()
    
    if len(tickers_incompletos) > 0:
        print(f"Tickers eliminados por datos incompletos ({len(tickers_incompletos)}): {list(tickers_incompletos)}")
    
    # Mantenemos solo los tickers que tienen la serie completa
    data_clean = data.drop(columns=tickers_incompletos, level=1)
    
    # 4. Procesamiento para almacenamiento
    # Pasamos de formato ancho (MultiIndex) a formato largo (Long format)
    df_db = data_clean.stack(level=1).reset_index()
    
    # Normalizar nombres de columnas a minúsculas
    df_db.columns = [col.lower() if isinstance(col, str) else col for col in df_db.columns]
    if 'level_1' in df_db.columns:
        df_db = df_db.rename(columns={'level_1': 'ticker'})

    # 5. Guardar en Parquet
    output_path = Path("./app/data/raw/stocks.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        num_tickers_final = len(df_db['ticker'].unique())
        print(f"Guardando {num_tickers_final} tickers en {output_path}...")
        df_db.to_parquet(output_path, index=False)
        print("¡Datos crudos guardados exitosamente!")
    except Exception as e:
        print(f"Error al guardar el archivo parquet: {e}")
else:
    print("No se pudo obtener la lista de activos.")

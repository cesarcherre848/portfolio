import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf

def scraping_profesional_sp100():
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # 1. Petición HTTP
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # 2. Parseo del DOM con BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 3. Localizar la tabla exacta (Wikipedia usa la clase 'wikitable')
        # Podemos ser aún más estrictos buscando el ID o el contenido previo
        tabla = soup.find('table', {'class': 'wikitable sortable'})
        
        if not tabla:
            raise ValueError("No se encontró la tabla con la clase 'wikitable sortable'.")
            
        # 4. Extracción manual de las filas (Node traversal)
        filas = tabla.find_all('tr')
        datos = []
        
        # Iteramos desde 1 para saltarnos la cabecera (<th>)
        for fila in filas[1:]:
            celdas = fila.find_all('td')
            
            # Verificamos que la fila tenga celdas (evita errores con filas vacías o malformadas)
            if len(celdas) >= 3:
                # Extraemos el texto y limpiamos espacios o saltos de línea (\n)
                symbol = celdas[0].text.strip().replace('.', '-')
                name = celdas[1].text.strip()
                sector = celdas[2].text.strip()
                
                datos.append([symbol, name, sector])
                
        # 5. Construcción del DataFrame
        df = pd.DataFrame(datos, columns=['Symbol', 'Name', 'Sector'])
        return df

    except Exception as e:
        print(f"Fallo en el pipeline de extracción: {e}")
        return None




df_activos = scraping_profesional_sp100()
tickers = df_activos["Symbol"].to_list()

data = yf.download(
    tickers=tickers,
    period="10y", 
    interval="1wk",
    auto_adjust=True
)


data = data.dropna(axis=1, subset=[data.index[0]])

df_db = data.stack(level=1).reset_index()

print(df_db)
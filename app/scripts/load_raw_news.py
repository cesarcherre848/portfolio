import pandas as pd
from edgar import set_identity, Company
from pathlib import Path
from tqdm import tqdm
import json
import time
from datetime import datetime

def load_tickers_from_stocks(stocks_path):
    """Lee los tickers únicos del archivo de stocks."""
    if not Path(stocks_path).exists():
        raise FileNotFoundError(f"No se encontró el archivo de stocks: {stocks_path}")
    df = pd.read_parquet(stocks_path)
    return df['ticker'].unique().tolist()

def download_full_news_json(tickers, email, base_output_dir, after_date="2000-01-01"):
    """
    Descarga el texto completo de los 8-K y los guarda en JSON individuales por ticker.
    """
    set_identity(email)
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    print(f"Iniciando descarga de 8-K para {len(tickers)} tickers desde {after_date}...")
    
    for ticker in tqdm(tickers):
        ticker_news = []
        sec_ticker = ticker.replace('-', '.')
        output_file = base_path / f"{ticker}_news.json"
        
        try:
            company = Company(sec_ticker)
            # Filtro de fecha en formato YYYY-MM-DD:
            filings = company.get_filings(form="8-K", filing_date=f"{after_date}:")
            
            if filings:
                for filing in filings:
                    try:
                        text_content = filing.text()
                        ticker_news.append({
                            "ticker": ticker,
                            "date": str(filing.filing_date),
                            "accession_number": filing.accession_no,
                            "content": text_content
                        })
                    except Exception:
                        continue
                
                if ticker_news:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(ticker_news, f, ensure_ascii=False, indent=4)
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError procesando {ticker}: {e}")

if __name__ == "__main__":
    STOCKS_PATH = "app/data/raw/stocks.parquet"
    NEWS_DIR = "app/data/raw/news"
    SEC_EMAIL = "admin@example.com" 

    # Configuración para procesar todos los tickers del último año
    PRUEBA_UN_TICKER = False
    
    try:
        if PRUEBA_UN_TICKER:
            tickers = ["AAPL"]
            fecha_inicio = "2023-05-02"
        else:
            tickers = load_tickers_from_stocks(STOCKS_PATH)
            # Un año atrás desde la fecha actual (2026-05-02)
            fecha_inicio = "2025-05-02"
            
        download_full_news_json(tickers, SEC_EMAIL, NEWS_DIR, after_date=fecha_inicio)
        print(f"\n¡Proceso completado! Archivos en {NEWS_DIR}")
    except Exception as e:
        print(f"Error en el pipeline de noticias: {e}")

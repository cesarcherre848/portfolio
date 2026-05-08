import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo para los gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(path):
    """Carga el dataset procesado."""
    print(f"Cargando datos desde {path}...")
    return pd.read_parquet(path)

def analyze_missing_values(df):
    """Analiza nulos y vacíos en el dataframe."""
    print("\n--- Análisis de Valores Faltantes ---")
    missing = df.isnull().sum()
    percent_missing = (missing / len(df)) * 100
    
    analysis_df = pd.DataFrame({
        'Nulos': missing,
        'Porcentaje (%)': percent_missing
    })
    print(analysis_df[analysis_df['Nulos'] > 0] if missing.sum() > 0 else "No se encontraron valores nulos.")
    
    # Verificar strings vacíos en columnas categóricas
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        empty_count = (df[col] == "").sum()
        if empty_count > 0:
            print(f"Columna '{col}' tiene {empty_count} strings vacíos.")

def descriptive_statistics(df):
    """Muestra estadísticas descriptivas de las columnas numéricas."""
    print("\n--- Estadísticas Descriptivas ---")
    print(df.describe().T)

def plot_correlations(df, output_dir):
    """Genera y guarda un mapa de calor de correlaciones."""
    print("\nGenerando matriz de correlación...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlación de Variables")
    
    plt.savefig(output_dir / "correlation_matrix.png")
    plt.close()

def plot_distributions(df, output_dir):
    """Genera y guarda histogramas de las variables principales."""
    print("Generando distribuciones...")
    cols = ['log_return', 'volume_prc', 'delta_fed_rate', 'yield_spread', 'vix_log_return']
    
    for col in cols:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col], kde=True, bins=50)
            plt.title(f"Distribución de {col}")
            plt.savefig(output_dir / f"dist_{col}.png")
            plt.close()

def plot_ticker_series(df, ticker, output_dir):
    """Plotea la serie temporal de retornos logarítmicos para un ticker específico."""
    ticker_data = df[df['ticker'] == ticker].sort_values('date')
    if ticker_data.empty:
        print(f"Ticker {ticker} no encontrado.")
        return
        
    plt.figure()
    plt.plot(ticker_data['date'], ticker_data['log_return'], label='Log Return')
    plt.title(f"Serie Temporal de Log Returns - {ticker}")
    plt.xlabel("Fecha")
    plt.ylabel("Log Return")
    plt.legend()
    
    plt.savefig(output_dir / f"series_{ticker}.png")
    plt.close()
    print(f"Gráfico para {ticker} guardado.")

def run_eda():
    """Ejecuta el flujo completo de EDA."""
    data_path = Path("app/data/processed/data.parquet")
    output_dir = Path("app/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"Error: No se encuentra el archivo {data_path}. ¿Has ejecutado el preprocesamiento?")
        return
        
    df = load_data(data_path)
    
    # 1. Info básica
    analyze_missing_values(df)
    
    # 2. Estadísticas
    descriptive_statistics(df)
    
    # 3. Visualizaciones generales
    plot_correlations(df, output_dir)
    plot_distributions(df, output_dir)
    
    # 4. Visualización por Ticker (ejemplos)
    tickers_to_plot = df['ticker'].unique()[:3] # Tomamos los 3 primeros como ejemplo
    for t in tickers_to_plot:
        plot_ticker_series(df, t, output_dir)
        
    print(f"\nEDA completado. Las figuras se han guardado en: {output_dir}")

if __name__ == "__main__":
    run_eda()

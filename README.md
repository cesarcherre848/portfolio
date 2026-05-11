# 📈 S&P 100 Log-Return Predictive Engine

Este proyecto implementa un sistema avanzado de inteligencia artificial diseñado para la predicción de retornos logarítmicos (*log-returns*) de los activos más estables del S&P 100. Utilizando una arquitectura híbrida de Deep Learning, el sistema integra factores macroeconómicos y técnicos para proyectar el comportamiento del mercado con alta fidelidad.

## 🚀 Visión General

El núcleo del sistema es un modelo de aprendizaje profundo que procesa múltiples flujos de datos simultáneamente:
*   **Rama Temporal:** Redes Neuronales Convolucionales (CNN) y Celdas de Memoria de Largo y Corto Plazo (LSTM) para capturar patrones de volatilidad y tendencias históricas.
*   **Rama Estática:** Capas de Embeddings para modelar las interrelaciones jerárquicas entre activos y sus sectores industriales.
*   **Factores Externos:** Integración de tasas de la FED y spreads de rendimiento para contextualizar las predicciones dentro del entorno macroeconómico.

## 🛠️ Arquitectura Técnica

### 1. Procesamiento de Datos (Pipeline)
*   **Ingesta:** Captura automatizada de datos de mercado (OHLCV) y fundamentales a través de `yfinance` y `fredapi`.
*   **Estructuración:** Transformación de datos tabulares a tensores de alta dimensión con ventanas deslizantes (*lags*) configurables.
*   **Horizonte Variable:** Soporte para predicción multi-paso mediante el parámetro `outputs_horizons`, permitiendo proyectar el retorno para *k* periodos futuros.
*   **Integridad de Datos:** Limpieza automática de muestras con `NaN` en el pipeline de TensorFlow para evitar la divergencia de gradientes y asegurar un entrenamiento estable.
*   **Normalización:** Cálculo de retornos logarítmicos y escalamiento robusto de variables numéricas.

### 2. Estructura del Modelo
El modelo utiliza un enfoque de **Fusión Multimodal**:
*   **Static Embeddings:** Procesa `Ticker` y `Sector`, convirtiendo identificadores categóricos en vectores densos que capturan la "personalidad" de cada empresa.
*   **Feature Extraction:** Capas 1D-CNN para identificar patrones locales en los retornos.
*   **Temporal Memory:** Capas LSTM para entender la dependencia secuencial a largo plazo.

## 📁 Estructura del Proyecto

```text
├── app/
│   ├── data/             # Almacenamiento optimizado (Parquet & TF Datasets)
│   ├── scripts/          # Pipeline: Carga -> Preprocesamiento -> Dataset -> Entrenamiento
│   └── models/           # Checkpoints y metadatos del modelo
├── requirements.txt      # Dependencias optimizadas para Python 3.13+
└── README.md
```

## ⚡ Ejecución Rápida

1.  **Preparar el entorno:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Ejecutar el pipeline de datos:**
    ```bash
    python app/scripts/preprocess_stocks.py
    python app/scripts/prepare_dataset.py
    ```
3.  **Entrenar el modelo:**
    ```bash
    python app/scripts/train_model.py
    ```

## 📊 Metas del Proyecto
*   **Precisión:** Minimizar el RMSE en la predicción de retornos a corto plazo.
*   **Escalabilidad:** Arquitectura preparada para integrarse con más activos del mercado global.
*   **Interpretabilidad:** Análisis de cómo los factores macro afectan a sectores específicos mediante el estudio de los pesos de los embeddings.

---
*Desarrollado para el análisis cuantitativo de carteras de inversión.*

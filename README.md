# Trading Algorítmico con IA — Actividad 3

Aplicación interactiva desarrollada como Actividad 3 de la asignatura **Inteligencia Artificial y Sistemas de Trading**, dentro del Máster en Mercados Financieros, Gestión de Carteras y Sistemas de Trading.

## 📋 Descripción

Esta app permite entrenar modelos de clasificación supervisada sobre activos del Ibex 35 (Santander, Telefónica y Repsol) y simular estrategias de trading algorítmico basadas en las predicciones del modelo.

El usuario puede configurar interactivamente:

- **Activo** y **periodo histórico** de análisis.
- **Variable objetivo**: horizonte de predicción (1, 3 o 5 días) y umbral de movimiento.
- **Indicadores técnicos**: SMA, EMA, RSI, MACD, Bandas de Bollinger, ATR, retornos rezagados, volumen.
- **Variables externas**: Ibex 35, EUR/USD, bonos 10A de España y Alemania.
- **Modelo de clasificación**: Random Forest, Regresión Logística, Naive Bayes o Árbol de Decisión.
- **Estrategia de trading**: solo largos o largos+cortos, umbral de probabilidad, stop-loss, take-profit, tamaño de posición.

## 🔍 Pestañas de resultados

1. **Datos** — Exploración del activo, distribución de la variable objetivo, correlaciones.
2. **Modelo** — Métricas (Accuracy, Precision, Recall, F1, AUC), matriz de confusión, curva ROC, importancia de variables.
3. **Comparativa de modelos** — Los 4 modelos enfrentados con el mismo dataset; declaración del modelo ganador.
4. **Backtest** — Equity curve vs Buy&Hold vs Ibex 35, drawdown, métricas financieras, tabla de operaciones.
5. **Comparativa de activos** — Mismo modelo aplicado a los tres activos.
6. **Conclusiones** — Resumen automático interpretativo.

## 🚀 Despliegue

La app está desplegada en Streamlit Community Cloud y es accesible públicamente.

## 📂 Estructura del repositorio

```
trading-ia-actividad3/
├── app.py                    # Aplicación Streamlit principal
├── requirements.txt          # Dependencias Python
├── README.md                 # Este archivo
├── .streamlit/
│   └── config.toml           # Tema visual de la app
├── data/
│   └── Datos_cotizaciones.xlsx
└── src/
    ├── __init__.py
    ├── data_loader.py        # Carga del Excel y descarga del Ibex 35
    ├── features.py           # Cálculo de indicadores técnicos y variable objetivo
    ├── models.py             # Entrenamiento y evaluación de modelos
    ├── backtest.py           # Motor de backtest y métricas financieras
    └── plots.py              # Gráficos Plotly reutilizables
```

## 🛠️ Stack técnico

- **Streamlit** — Framework de la app web interactiva.
- **scikit-learn** — Modelos de clasificación y validación.
- **pandas / numpy** — Manipulación de datos e indicadores técnicos.
- **Plotly** — Visualizaciones interactivas.
- **yfinance** — Descarga del Ibex 35 como benchmark.

## 📚 Asignatura

**Inteligencia Artificial y Sistemas de Trading**
Máster en Mercados Financieros, Gestión de Carteras y Sistemas de Trading.

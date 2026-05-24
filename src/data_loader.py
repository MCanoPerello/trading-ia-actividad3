"""
data_loader.py
==============
Carga las cotizaciones del Excel del máster, las limpia, ordena cronológicamente
y descarga el Ibex 35 como benchmark.

El Excel original tiene los siguientes problemas que aquí resolvemos:
  - Fechas en formato americano MM/DD/YYYY como string.
  - Datos ordenados de más reciente a más antiguo (los invertimos).
  - Precios escalados (multiplicados por factor según activo, por la conversión
    coma-decimal europea ↔ punto-decimal anglosajón que hizo Excel).
  - Volumen como texto con sufijos K/M/B.
  - Cada serie tiene festivos distintos: alineamos todo por fecha con forward-fill.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from pathlib import Path

# Ruta al Excel
DATA_PATH = Path(__file__).parent.parent / "data" / "Datos_cotizaciones.xlsx"

# Mapeo de activos seleccionables → (nombre de hoja, factor de escalado)
# El factor convierte el precio del Excel a euros reales.
ASSETS = {
    "Santander":  {"sheet": "Santander Stock Price History (", "scale": 10000},
    "Telefónica": {"sheet": "Telefonica Stock Price History ", "scale": 10000},
    "Repsol":     {"sheet": "Repsol Stock Price History",      "scale": 1000},
}

# Variables externas: hoja y factor de escalado
EXTERNALS = {
    "EUR/USD":           {"sheet": "EUR_USD Historical Data (2)",      "scale": 10000, "is_yield": False},
    "Bono España 10A":   {"sheet": "Spain 10-Year Bond Yield Histor",  "scale": 1000,  "is_yield": True},
    "Bono Alemania 10A": {"sheet": "Germany 10-Year Bond Yield Hist",  "scale": 10000, "is_yield": True},
}


def _parse_volume(vol_str):
    """Convierte '136.81K', '22.91M', '1.5B' a número."""
    if pd.isna(vol_str) or vol_str == '' or vol_str is None:
        return np.nan
    if isinstance(vol_str, (int, float)):
        return float(vol_str)
    s = str(vol_str).strip().upper().replace(',', '')
    if s == '' or s == 'NAN':
        return np.nan
    multiplier = 1
    if s.endswith('K'):
        multiplier = 1e3
        s = s[:-1]
    elif s.endswith('M'):
        multiplier = 1e6
        s = s[:-1]
    elif s.endswith('B'):
        multiplier = 1e9
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan


def _read_sheet(sheet_name, scale, has_volume=True):
    """Lee una hoja del Excel y devuelve un DataFrame limpio indexado por fecha."""
    df = pd.read_excel(DATA_PATH, sheet_name=sheet_name, header=0)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])

    # Escalar precios
    for col in ['Price', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') / scale

    # Volumen (puede no estar)
    if has_volume and 'Vol.' in df.columns:
        df['Volume'] = df['Vol.'].apply(_parse_volume)
    else:
        df['Volume'] = np.nan

    df = df.set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


@st.cache_data(show_spinner=False)
def load_asset(asset_name):
    """Carga un activo individual (Santander, Telefónica o Repsol)."""
    info = ASSETS[asset_name]
    df = _read_sheet(info['sheet'], info['scale'], has_volume=True)
    return df[['Price', 'Open', 'High', 'Low', 'Volume']].rename(columns={'Price': 'Close'})


@st.cache_data(show_spinner=False)
def load_external(name):
    """Carga una variable externa (EUR/USD, bonos)."""
    info = EXTERNALS[name]
    df = _read_sheet(info['sheet'], info['scale'], has_volume=False)
    return df[['Price']].rename(columns={'Price': name})


@st.cache_data(show_spinner=False, ttl=24*3600)
def load_ibex():
    """Descarga el Ibex 35 desde Yahoo Finance. Cacheado 24h."""
    try:
        df = yf.download('^IBEX', start='2014-12-01', end='2025-05-01',
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        # yfinance puede devolver columnas multi-nivel; aplanar
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].rename(columns={'Close': 'IBEX'})
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.warning(f"No se pudo descargar el Ibex 35 ({e}). Se trabajará sin este benchmark.")
        return None


@st.cache_data(show_spinner=False)
def build_dataset(asset_name, externals_selected, date_start, date_end):
    """
    Construye el DataFrame consolidado:
      - Precios del activo seleccionado (OHLC + Volume).
      - Variables externas seleccionadas (alineadas y rellenadas).
      - Ibex 35 como referencia (siempre incluido).
    Devuelve un DataFrame indexado por fecha, ordenado, filtrado al rango.
    """
    # Activo principal
    asset = load_asset(asset_name).copy()

    # Variables externas
    for ext_name in externals_selected:
        if ext_name in EXTERNALS:
            ext_df = load_external(ext_name)
            asset = asset.join(ext_df, how='left')

    # Ibex 35 siempre intentamos cargarlo
    ibex = load_ibex()
    if ibex is not None:
        asset = asset.join(ibex, how='left')

    # Forward-fill para festivos no comunes (máx. 5 días)
    asset = asset.ffill(limit=5)

    # Filtrar por fechas
    asset = asset.loc[date_start:date_end]

    # Eliminar filas con NaN en el Close del activo principal
    asset = asset.dropna(subset=['Close'])

    return asset


def get_date_range(asset_name):
    """Devuelve la fecha mínima y máxima disponibles para un activo."""
    df = load_asset(asset_name)
    return df.index.min().date(), df.index.max().date()

"""
features.py
===========
Cálculo de indicadores técnicos y construcción de la variable objetivo.

Todos los indicadores se implementan en pandas/numpy sin dependencias externas
para tener control total y poder explicarlos en la defensa del trabajo.
"""

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
# Indicadores técnicos                                                #
# ------------------------------------------------------------------ #

def sma(series, window):
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series, window):
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series, window=14):
    """
    Relative Strength Index.
    Mide la fuerza/debilidad del precio. RSI > 70 = sobrecompra; RSI < 30 = sobreventa.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    """
    MACD - Moving Average Convergence Divergence.
    Devuelve: línea MACD, línea de señal e histograma (diferencia).
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(series, window=20, n_std=2):
    """
    Bandas de Bollinger.
    Devuelve: banda superior, media, banda inferior, %B (posición dentro de la banda).
    """
    middle = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, middle, lower, pct_b


def atr(high, low, close, window=14):
    """
    Average True Range. Medida de volatilidad.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()


# ------------------------------------------------------------------ #
# Construcción del dataset de features                                #
# ------------------------------------------------------------------ #

def build_features(df, selected_indicators, selected_externals=None):
    """
    Construye el DataFrame de features a partir del dataset consolidado.

    Parámetros
    ----------
    df : DataFrame
        Datos consolidados con columnas: Close, Open, High, Low, Volume,
        y variables externas opcionales (EUR/USD, Bono España 10A, etc.)
    selected_indicators : list[str]
        Lista de indicadores a calcular. Valores aceptados:
        ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger', 'ATR', 'Lags', 'Volume']
    selected_externals : list[str]
        Variables externas a incluir como features (deben existir en df).

    Devuelve
    --------
    DataFrame con todas las features alineadas.
    """
    feat = pd.DataFrame(index=df.index)
    close = df['Close']

    if 'SMA' in selected_indicators:
        feat['SMA_20'] = sma(close, 20)
        feat['SMA_50'] = sma(close, 50)
        feat['SMA_ratio'] = feat['SMA_20'] / feat['SMA_50']

    if 'EMA' in selected_indicators:
        feat['EMA_20'] = ema(close, 20)
        feat['EMA_ratio'] = close / feat['EMA_20']

    if 'RSI' in selected_indicators:
        feat['RSI_14'] = rsi(close, 14)

    if 'MACD' in selected_indicators:
        macd_line, signal_line, histogram = macd(close)
        feat['MACD'] = macd_line
        feat['MACD_signal'] = signal_line
        feat['MACD_hist'] = histogram

    if 'Bollinger' in selected_indicators:
        upper, middle, lower, pct_b = bollinger(close)
        feat['BB_pct'] = pct_b
        feat['BB_width'] = (upper - lower) / middle

    if 'ATR' in selected_indicators and 'High' in df.columns and 'Low' in df.columns:
        feat['ATR_14'] = atr(df['High'], df['Low'], close, 14)
        feat['ATR_pct'] = feat['ATR_14'] / close  # normalizado

    if 'Lags' in selected_indicators:
        ret = close.pct_change()
        for lag in [1, 2, 3, 5]:
            feat[f'Ret_lag_{lag}'] = ret.shift(lag)

    if 'Volume' in selected_indicators and 'Volume' in df.columns:
        vol = df['Volume']
        feat['Volume_norm'] = vol / vol.rolling(20, min_periods=20).mean()

    # Variables externas (como retornos para que sean estacionarias)
    if selected_externals:
        for ext in selected_externals:
            if ext in df.columns:
                # Para EUR/USD usamos retorno; para bonos usamos diferencia (puntos básicos)
                if 'Bono' in ext:
                    feat[f'{ext}_chg'] = df[ext].diff()
                else:
                    feat[f'{ext}_ret'] = df[ext].pct_change()

    # Ibex 35 como variable externa siempre que esté disponible
    if 'IBEX' in df.columns:
        feat['IBEX_ret'] = df['IBEX'].pct_change()

    return feat


def build_target(df, horizon=1, threshold=0.0):
    """
    Construye la variable objetivo binaria.
    y(t) = 1 si Close(t+horizon) > Close(t) * (1 + threshold), 0 en caso contrario.

    Parámetros
    ----------
    horizon : int
        Días hacia delante para mirar el movimiento.
    threshold : float
        Umbral mínimo de subida (en tanto por uno). 0 = cualquier subida cuenta.
    """
    close = df['Close']
    future_return = close.shift(-horizon) / close - 1
    target = (future_return > threshold).astype(int)
    # Las últimas `horizon` filas no tienen futuro: descartarlas como NaN
    target.iloc[-horizon:] = np.nan
    return target


def prepare_xy(df, indicators, externals, horizon, threshold):
    """
    Función conjunta: construye features y target, alinea, elimina NaNs.
    Devuelve X, y y el índice de fechas resultante.
    """
    X = build_features(df, indicators, externals)
    y = build_target(df, horizon, threshold)

    # Concatenar y limpiar NaNs (vienen de los indicadores que requieren ventana inicial
    # y del horizonte que descarta las últimas filas)
    combined = X.copy()
    combined['_target'] = y
    combined = combined.dropna()

    y_clean = combined['_target'].astype(int)
    X_clean = combined.drop(columns='_target')

    return X_clean, y_clean

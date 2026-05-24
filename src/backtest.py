"""
backtest.py
===========
Motor de backtest sencillo y transparente.

Lógica:
  - Cada día decide si tener posición (largo/corto/fuera) según la predicción del modelo.
  - Aplica stop-loss y take-profit sobre el precio de entrada.
  - Calcula equity curve, drawdown, número de operaciones y métricas financieras.

Sin costes de transacción (decisión explícita del enunciado de la actividad).
"""

import numpy as np
import pandas as pd


def run_backtest(prices, signals, initial_capital=10000.0,
                 allow_short=False, prob_threshold=0.5, probabilities=None,
                 stop_loss_pct=0.0, take_profit_pct=0.0,
                 position_size_pct=1.0):
    """
    Ejecuta el backtest.

    Parámetros
    ----------
    prices : Series de precios de cierre del activo (indexada por fecha)
    signals : Series binarias 0/1 alineadas con prices (1 = el modelo predice subida)
    initial_capital : capital inicial
    allow_short : si True, cuando predice 0 entra corto; si False, queda fuera
    prob_threshold : umbral de probabilidad para activar señal (necesita 'probabilities')
    probabilities : Series con prob de clase 1 (opcional, para aplicar threshold)
    stop_loss_pct : ej. 0.05 = 5% (0 desactiva)
    take_profit_pct : ej. 0.10 = 10% (0 desactiva)
    position_size_pct : fracción del capital a usar en cada operación (0-1)

    Devuelve un dict con:
      - 'equity': Series de equity (valor de la cuenta día a día)
      - 'buy_hold': Series con equity de un buy & hold del mismo activo
      - 'trades': DataFrame con las operaciones cerradas
      - 'positions': Series de posición diaria (-1, 0, 1)
      - 'metrics': dict con métricas financieras
    """
    # Alinear todo al índice común
    df = pd.DataFrame({'price': prices})
    df['signal'] = signals.reindex(df.index)
    if probabilities is not None:
        df['proba'] = probabilities.reindex(df.index)
    df = df.dropna()

    # Convertir señal a posición deseada: 1 = largo, -1 = corto, 0 = fuera
    if probabilities is not None:
        # Largo cuando proba >= threshold; corto (si allow_short) cuando proba <= 1-threshold
        df['desired'] = 0
        df.loc[df['proba'] >= prob_threshold, 'desired'] = 1
        if allow_short:
            df.loc[df['proba'] <= (1 - prob_threshold), 'desired'] = -1
    else:
        df['desired'] = df['signal'].map(lambda s: 1 if s == 1 else (-1 if allow_short else 0))

    # Simulación día a día
    equity = []
    positions = []
    trades = []

    capital = initial_capital
    pos = 0          # posición actual: -1, 0, 1
    entry_price = None
    pos_capital = 0  # capital comprometido en la posición
    entry_date = None

    prices_arr = df['price'].values
    desired_arr = df['desired'].values
    dates = df.index

    for i in range(len(df)):
        price = prices_arr[i]
        date = dates[i]

        # Si hay posición abierta, comprobar stop-loss / take-profit
        if pos != 0:
            ret_open = (price / entry_price - 1) * pos  # retorno de la posición (corto invierte)
            exit_reason = None
            if stop_loss_pct > 0 and ret_open <= -stop_loss_pct:
                exit_reason = 'Stop-Loss'
            elif take_profit_pct > 0 and ret_open >= take_profit_pct:
                exit_reason = 'Take-Profit'

            if exit_reason is not None:
                # Cerrar posición a precio actual
                pnl = pos_capital * ret_open
                capital += pnl
                trades.append({
                    'entry_date': entry_date, 'exit_date': date,
                    'side': 'Largo' if pos == 1 else 'Corto',
                    'entry_price': entry_price, 'exit_price': price,
                    'return_pct': ret_open * 100, 'pnl': pnl,
                    'reason': exit_reason
                })
                pos = 0
                pos_capital = 0
                entry_price = None
                entry_date = None

        # Cambio de señal: cerrar (si hay) y abrir nueva (si la señal lo pide)
        desired = desired_arr[i]
        if desired != pos:
            # Cerrar posición previa si la había
            if pos != 0:
                ret_close = (price / entry_price - 1) * pos
                pnl = pos_capital * ret_close
                capital += pnl
                trades.append({
                    'entry_date': entry_date, 'exit_date': date,
                    'side': 'Largo' if pos == 1 else 'Corto',
                    'entry_price': entry_price, 'exit_price': price,
                    'return_pct': ret_close * 100, 'pnl': pnl,
                    'reason': 'Señal'
                })
                pos = 0
                pos_capital = 0

            # Abrir nueva si la señal pide
            if desired != 0:
                pos = desired
                entry_price = price
                pos_capital = capital * position_size_pct
                entry_date = date

        # Calcular equity actual (capital + valor de la posición abierta)
        if pos != 0:
            ret_open = (price / entry_price - 1) * pos
            current_value = capital + pos_capital * ret_open - pos_capital + pos_capital
            # Reescritura más clara:
            unrealized_pnl = pos_capital * ret_open
            current_value = capital + unrealized_pnl
        else:
            current_value = capital

        equity.append(current_value)
        positions.append(pos)

    equity_s = pd.Series(equity, index=df.index, name='Equity')
    positions_s = pd.Series(positions, index=df.index, name='Position')

    # Buy & Hold del propio activo (mismo capital inicial)
    bh = initial_capital * df['price'] / df['price'].iloc[0]
    bh.name = 'Buy & Hold'

    trades_df = pd.DataFrame(trades)

    metrics = compute_metrics(equity_s, bh, trades_df, initial_capital)

    return {
        'equity': equity_s,
        'buy_hold': bh,
        'trades': trades_df,
        'positions': positions_s,
        'metrics': metrics,
    }


def compute_metrics(equity, buy_hold, trades, initial_capital):
    """Calcula las métricas financieras estándar para la estrategia y el B&H."""
    def stats(serie, name):
        rets = serie.pct_change().dropna()
        n_years = max((serie.index[-1] - serie.index[0]).days / 365.25, 0.01)
        total_ret = (serie.iloc[-1] / serie.iloc[0]) - 1
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        ann_vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        downside = rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else 0
        sortino = ann_ret / downside if downside > 0 else 0
        # Max drawdown
        cummax = serie.cummax()
        dd = (serie / cummax - 1)
        max_dd = dd.min()
        return {
            'name': name,
            'final_value': serie.iloc[-1],
            'total_return_pct': total_ret * 100,
            'annual_return_pct': ann_ret * 100,
            'annual_vol_pct': ann_vol * 100,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown_pct': max_dd * 100,
        }

    strat = stats(equity, 'Estrategia IA')
    bh = stats(buy_hold, 'Buy & Hold')

    # Métricas específicas de la estrategia: nº operaciones, win rate, profit factor
    n_trades = len(trades)
    if n_trades > 0:
        wins = trades[trades['return_pct'] > 0]
        losses = trades[trades['return_pct'] <= 0]
        win_rate = len(wins) / n_trades * 100
        gross_win = wins['pnl'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf
    else:
        win_rate = 0
        profit_factor = 0

    strat['n_trades'] = n_trades
    strat['win_rate_pct'] = win_rate
    strat['profit_factor'] = profit_factor

    return {'strategy': strat, 'buy_hold': bh}


def benchmark_equity(prices_benchmark, initial_capital=10000.0):
    """Construye la equity de un benchmark (ej. Ibex) con el mismo capital inicial."""
    if prices_benchmark is None or len(prices_benchmark.dropna()) < 2:
        return None
    p = prices_benchmark.dropna()
    return initial_capital * p / p.iloc[0]


def drawdown_series(equity):
    """Devuelve la serie de drawdown (%) día a día."""
    cummax = equity.cummax()
    return (equity / cummax - 1) * 100

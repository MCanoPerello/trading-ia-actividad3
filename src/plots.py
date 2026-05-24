"""
plots.py
========
Funciones de gráficos Plotly reutilizables.

Paleta de colores consistente y profesional, estilo financiero.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# Paleta corporativa - tonos azules + acento dorado/cian
COLOR_PRIMARY = '#0A66C2'      # azul corporativo
COLOR_SECONDARY = '#00B4D8'    # cian
COLOR_ACCENT = '#D4A017'       # dorado financiero
COLOR_GREEN = '#2E8B57'        # subidas
COLOR_RED = '#C0392B'          # bajadas
COLOR_GRAY = '#5A6B7B'         # neutro
COLOR_BG_LIGHT = '#F4F6FA'

PLOTLY_LAYOUT = dict(
    template='plotly_white',
    font=dict(family='Inter, Helvetica, Arial, sans-serif', size=12, color='#0E1A2B'),
    paper_bgcolor='white',
    plot_bgcolor='white',
    hovermode='x unified',
    margin=dict(l=40, r=20, t=50, b=40),
)


def fig_price(df, train_end_date=None, title="Precio del activo"):
    """Gráfico de precio con división train/test si se proporciona."""
    fig = go.Figure()
    if train_end_date is not None:
        train = df[df.index <= train_end_date]
        test = df[df.index > train_end_date]
        fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train',
                                 line=dict(color=COLOR_PRIMARY, width=2)))
        fig.add_trace(go.Scatter(x=test.index, y=test['Close'], name='Test',
                                 line=dict(color=COLOR_ACCENT, width=2)))
        # add_vline requiere la fecha como string para evitar problemas con Timestamp
        vline_x = pd.Timestamp(train_end_date).strftime('%Y-%m-%d')
        fig.add_shape(type='line', x0=vline_x, x1=vline_x, xref='x',
                      y0=0, y1=1, yref='paper',
                      line=dict(color=COLOR_GRAY, dash='dash', width=1.5))
        fig.add_annotation(x=vline_x, y=1, xref='x', yref='paper',
                           text='Inicio test', showarrow=False, yshift=10,
                           font=dict(color=COLOR_GRAY, size=11))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Precio',
                                 line=dict(color=COLOR_PRIMARY, width=2)))
    fig.update_layout(title=title, xaxis_title='Fecha', yaxis_title='Precio (€)',
                      **PLOTLY_LAYOUT)
    return fig


def fig_target_distribution(y_train, y_test):
    """Histograma de la distribución de la variable objetivo en train y test."""
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    labels = ['Baja (0)', 'Sube (1)']
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Train', x=labels,
                         y=[train_counts.get(0, 0), train_counts.get(1, 0)],
                         marker_color=COLOR_PRIMARY))
    fig.add_trace(go.Bar(name='Test', x=labels,
                         y=[test_counts.get(0, 0), test_counts.get(1, 0)],
                         marker_color=COLOR_ACCENT))
    fig.update_layout(title='Distribución de la variable objetivo', barmode='group',
                      yaxis_title='Nº observaciones', **PLOTLY_LAYOUT)
    return fig


def fig_returns_hist(prices):
    """Histograma de retornos diarios."""
    rets = prices.pct_change().dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets, nbinsx=60, marker_color=COLOR_PRIMARY,
                                marker_line_color='white', marker_line_width=0.5))
    fig.update_layout(title='Distribución de retornos diarios (%)',
                      xaxis_title='Retorno diario (%)', yaxis_title='Frecuencia',
                      **PLOTLY_LAYOUT)
    return fig


def fig_correlation(X):
    """Heatmap de correlaciones entre features."""
    corr = X.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu', zmin=-1, zmax=1, zmid=0,
        colorbar=dict(title='Corr.')
    ))
    fig.update_layout(title='Matriz de correlaciones entre features',
                      **{**PLOTLY_LAYOUT, 'height': 600})
    return fig


def fig_confusion_matrix(cm, labels=('Baja (0)', 'Sube (1)')):
    """Matriz de confusión como heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=list(labels), y=list(labels),
        colorscale='Blues',
        text=cm, texttemplate='%{text}', textfont=dict(size=16, color='white')
    ))
    fig.update_layout(title='Matriz de confusión',
                      xaxis_title='Predicho', yaxis_title='Real',
                      yaxis=dict(autorange='reversed'),
                      **{**PLOTLY_LAYOUT, 'height': 400})
    return fig


def fig_roc(fpr, tpr, auc_score):
    """Curva ROC."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc_score:.3f})',
                             line=dict(color=COLOR_PRIMARY, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Azar',
                             line=dict(color=COLOR_GRAY, width=1, dash='dash')))
    fig.update_layout(title='Curva ROC',
                      xaxis_title='Tasa de falsos positivos',
                      yaxis_title='Tasa de verdaderos positivos',
                      **{**PLOTLY_LAYOUT, 'height': 400})
    return fig


def fig_importance(importance, top_n=15):
    """Gráfico de barras horizontal de importancia de variables."""
    if importance is None or len(importance) == 0:
        return None
    imp = importance.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation='h',
        marker_color=COLOR_PRIMARY
    ))
    fig.update_layout(title=f'Top {top_n} variables más importantes',
                      xaxis_title='Importancia', yaxis_title='',
                      **{**PLOTLY_LAYOUT, 'height': 450})
    return fig


def fig_equity(equity, buy_hold, benchmark=None, benchmark_name='Ibex 35'):
    """Equity curve: estrategia vs Buy&Hold vs benchmark opcional."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name='Estrategia IA',
                             line=dict(color=COLOR_PRIMARY, width=2.5)))
    fig.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold.values, name='Buy & Hold',
                             line=dict(color=COLOR_GRAY, width=2)))
    if benchmark is not None:
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name=benchmark_name,
                                 line=dict(color=COLOR_ACCENT, width=2, dash='dot')))
    fig.update_layout(title='Equity curve - Estrategia vs Benchmarks',
                      xaxis_title='Fecha', yaxis_title='Valor de la cuenta (€)',
                      **PLOTLY_LAYOUT)
    return fig


def fig_drawdown(drawdown):
    """Gráfico de área de drawdown."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                             fill='tozeroy', mode='lines',
                             line=dict(color=COLOR_RED, width=1),
                             fillcolor='rgba(192, 57, 43, 0.3)',
                             name='Drawdown'))
    fig.update_layout(title='Drawdown de la estrategia (%)',
                      xaxis_title='Fecha', yaxis_title='Drawdown (%)',
                      **{**PLOTLY_LAYOUT, 'height': 300})
    return fig


def fig_models_comparison(df_comp):
    """Gráfico comparativo de barras agrupadas con todas las métricas por modelo."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    fig = go.Figure()
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_GREEN, COLOR_GRAY]
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(name=m, x=df_comp.index, y=df_comp[m],
                             marker_color=colors[i % len(colors)]))
    fig.update_layout(title='Comparativa de modelos en test',
                      yaxis_title='Valor de la métrica',
                      yaxis_range=[0, 1], barmode='group',
                      **PLOTLY_LAYOUT)
    return fig


def fig_assets_comparison(results_dict, initial_capital=10000):
    """Comparativa de equity entre los 3 activos."""
    fig = go.Figure()
    colors = {'Santander': COLOR_PRIMARY, 'Telefónica': COLOR_ACCENT, 'Repsol': COLOR_GREEN}
    for asset, data in results_dict.items():
        fig.add_trace(go.Scatter(x=data['equity'].index, y=data['equity'].values,
                                 name=asset,
                                 line=dict(color=colors.get(asset, COLOR_GRAY), width=2.5)))
    # Línea horizontal del capital inicial
    fig.add_shape(type='line', xref='paper', x0=0, x1=1,
                  yref='y', y0=initial_capital, y1=initial_capital,
                  line=dict(color=COLOR_GRAY, dash='dot', width=1.5))
    fig.add_annotation(xref='paper', x=0.01, yref='y', y=initial_capital,
                       text='Capital inicial', showarrow=False,
                       yshift=10, xanchor='left',
                       font=dict(color=COLOR_GRAY, size=11))
    fig.update_layout(title='Equity de la estrategia en los 3 activos',
                      xaxis_title='Fecha', yaxis_title='Valor de la cuenta (€)',
                      **PLOTLY_LAYOUT)
    return fig

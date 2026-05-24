"""
app.py — Trading Algorítmico con IA
=====================================
Actividad 3 - Máster en Mercados Financieros, Gestión de Carteras y Sistemas de Trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from src.data_loader import build_dataset, get_date_range, ASSETS, EXTERNALS
from src.features import prepare_xy
from src.models import (time_split, train_and_evaluate, compare_all_models,
                        pick_best_model, MODEL_REGISTRY)
from src.backtest import run_backtest, drawdown_series, benchmark_equity
from src import plots as P


# ------------------------------------------------------------------ #
# Configuración de la página                                          #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Trading IA - Actividad 3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado para look profesional
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A66C2 0%, #0E1A2B 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #D4E4F5; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .metric-card {
        background: #F4F6FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0A66C2;
    }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    h2, h3 { color: #0E1A2B; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F4F6FA;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0A66C2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Cabecera                                                            #
# ------------------------------------------------------------------ #

st.markdown("""
<div class="main-header">
    <h1>📈 Trading Algorítmico con IA — Actividad 3</h1>
    <p>Máster en Mercados Financieros, Gestión de Carteras y Sistemas de Trading · Inteligencia Artificial y Sistemas de Trading</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Sidebar - Panel de configuración                                    #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    # --- Activo y periodo ---
    with st.expander("1️⃣ Activo y periodo", expanded=True):
        asset_name = st.radio("Activo", list(ASSETS.keys()), index=0)

        min_date, max_date = get_date_range(asset_name)
        date_range = st.slider(
            "Periodo de análisis",
            min_value=min_date, max_value=max_date,
            value=(date(2020, 1, 1), max_date),
            format="DD/MM/YYYY"
        )

        train_size = st.slider("Proporción de entrenamiento", 0.60, 0.90, 0.80, 0.05)

    # --- Variable objetivo ---
    with st.expander("2️⃣ Variable objetivo"):
        horizon = st.radio("Horizonte de predicción (días)", [1, 3, 5], index=0, horizontal=True)
        threshold_pct = st.slider("Umbral de subida mínima (%)", 0.0, 1.0, 0.0, 0.1,
                                   help="0% = cualquier subida cuenta como '1'. Valores mayores filtran ruido pero descompensan clases.")
        threshold = threshold_pct / 100

    # --- Indicadores técnicos ---
    with st.expander("3️⃣ Indicadores técnicos"):
        all_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger', 'ATR', 'Lags', 'Volume']
        selected_indicators = []
        cols = st.columns(2)
        for i, ind in enumerate(all_indicators):
            with cols[i % 2]:
                if st.checkbox(ind, value=True, key=f"ind_{ind}"):
                    selected_indicators.append(ind)

    # --- Variables externas ---
    with st.expander("4️⃣ Variables externas"):
        all_externals = list(EXTERNALS.keys())
        selected_externals = []
        for ext in all_externals:
            if st.checkbox(ext, value=True, key=f"ext_{ext}"):
                selected_externals.append(ext)
        st.caption("El Ibex 35 se incorpora automáticamente como variable externa (descarga desde Yahoo Finance).")

    # --- Modelo ---
    with st.expander("5️⃣ Modelo"):
        model_name = st.radio("Algoritmo", list(MODEL_REGISTRY.keys()), index=0)
        standardize = st.checkbox("Estandarizar variables", value=(model_name in ('Regresión Logística', 'Naive Bayes')),
                                  help="Recomendable para Regresión Logística.")

        with st.expander("Parámetros avanzados"):
            params = {}
            if model_name == 'Random Forest':
                params['n_estimators'] = st.slider("Nº de árboles", 50, 500, 200, 50)
                params['max_depth'] = st.slider("Profundidad máxima", 3, 20, 8)
                params['min_samples_leaf'] = st.slider("Muestras mínimas por hoja", 1, 20, 5)
            elif model_name == 'Regresión Logística':
                params['C'] = st.select_slider("Regularización C", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
            elif model_name == 'Árbol de Decisión':
                params['max_depth'] = st.slider("Profundidad máxima", 3, 20, 6)
                params['min_samples_leaf'] = st.slider("Muestras mínimas por hoja", 1, 30, 10)

    # --- Estrategia ---
    with st.expander("6️⃣ Estrategia de trading"):
        operativa = st.radio("Tipo de operativa",
                             ["Solo largos (fuera cuando predice bajada)", "Largos y cortos"], index=0)
        allow_short = (operativa == "Largos y cortos")
        prob_threshold = st.slider("Umbral de probabilidad para entrar", 0.50, 0.70, 0.50, 0.01,
                                    help="Solo entra cuando la confianza del modelo supera este umbral.")
        initial_capital = st.number_input("Capital inicial (€)", 1000, 1000000, 10000, 1000)
        position_size_pct = st.slider("Tamaño de posición (% capital)", 10, 100, 100, 10) / 100
        stop_loss_pct = st.slider("Stop-Loss (%)", 0.0, 10.0, 0.0, 0.5,
                                   help="0 desactiva. Cierra posición si las pérdidas superan este %.") / 100
        take_profit_pct = st.slider("Take-Profit (%)", 0.0, 20.0, 0.0, 0.5,
                                     help="0 desactiva. Cierra posición si los beneficios superan este %.") / 100

    st.markdown("---")
    run = st.button("🚀 Ejecutar simulación", type="primary", use_container_width=True)


# ------------------------------------------------------------------ #
# Ejecución del pipeline                                              #
# ------------------------------------------------------------------ #

def run_pipeline(asset_name, date_range, selected_externals, selected_indicators,
                 horizon, threshold, model_name, params, standardize, train_size,
                 allow_short, prob_threshold, initial_capital,
                 position_size_pct, stop_loss_pct, take_profit_pct):
    """Ejecuta todo el pipeline y devuelve los resultados para mostrar."""
    df = build_dataset(asset_name, selected_externals,
                       pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    X, y = prepare_xy(df, selected_indicators, selected_externals, horizon, threshold)
    X_train, X_test, y_train, y_test = time_split(X, y, train_size)

    # Modelo elegido
    result = train_and_evaluate(model_name, params, X_train, X_test, y_train, y_test, standardize)

    # Comparativa de los 4 modelos
    df_comp, all_results = compare_all_models(X_train, X_test, y_train, y_test)
    best_model = pick_best_model(df_comp, 'F1')

    # Backtest
    prices_test = df.loc[X_test.index, 'Close']
    bt = run_backtest(prices_test, result['y_pred_test'],
                      initial_capital=initial_capital,
                      allow_short=allow_short,
                      probabilities=result['y_proba_test'],
                      prob_threshold=prob_threshold,
                      stop_loss_pct=stop_loss_pct,
                      take_profit_pct=take_profit_pct,
                      position_size_pct=position_size_pct)

    # Benchmark Ibex
    ibex_test = None
    if 'IBEX' in df.columns:
        ibex_test = df.loc[X_test.index, 'IBEX'].dropna()
        if len(ibex_test) >= 2:
            ibex_test = benchmark_equity(ibex_test, initial_capital)

    return {
        'df': df, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'result': result, 'df_comp': df_comp, 'all_results': all_results,
        'best_model': best_model, 'bt': bt, 'ibex_eq': ibex_test,
    }


def run_assets_comparison(date_range, selected_externals, selected_indicators,
                          horizon, threshold, model_name, params, standardize, train_size,
                          allow_short, prob_threshold, initial_capital,
                          position_size_pct, stop_loss_pct, take_profit_pct):
    """Ejecuta el mismo modelo sobre los 3 activos para la comparativa."""
    out = {}
    for asset in ASSETS.keys():
        try:
            r = run_pipeline(asset, date_range, selected_externals, selected_indicators,
                             horizon, threshold, model_name, params, standardize, train_size,
                             allow_short, prob_threshold, initial_capital,
                             position_size_pct, stop_loss_pct, take_profit_pct)
            out[asset] = {
                'equity': r['bt']['equity'],
                'metrics_strat': r['bt']['metrics']['strategy'],
                'metrics_bh': r['bt']['metrics']['buy_hold'],
                'metrics_model': r['result']['metrics_test'],
            }
        except Exception as e:
            st.warning(f"No se pudo ejecutar {asset}: {e}")
    return out


# Estado de sesión
if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.assets_comp = None

if run:
    if len(selected_indicators) == 0:
        st.error("Debes seleccionar al menos un indicador técnico.")
    else:
        with st.spinner("Cargando datos, entrenando modelos y ejecutando backtest..."):
            try:
                st.session_state.results = run_pipeline(
                    asset_name, date_range, selected_externals, selected_indicators,
                    horizon, threshold, model_name, params, standardize, train_size,
                    allow_short, prob_threshold, initial_capital,
                    position_size_pct, stop_loss_pct, take_profit_pct,
                )
            except Exception as e:
                st.error(f"Error en el pipeline: {e}")
                st.session_state.results = None

        if st.session_state.results is not None:
            with st.spinner("Ejecutando comparativa de los 3 activos..."):
                st.session_state.assets_comp = run_assets_comparison(
                    date_range, selected_externals, selected_indicators,
                    horizon, threshold, model_name, params, standardize, train_size,
                    allow_short, prob_threshold, initial_capital,
                    position_size_pct, stop_loss_pct, take_profit_pct,
                )


# ------------------------------------------------------------------ #
# Área principal - Pestañas                                           #
# ------------------------------------------------------------------ #

if st.session_state.results is None:
    st.info("👈 Configura los parámetros en el panel lateral y pulsa **Ejecutar simulación**.")
    st.markdown("""
    ### ¿Qué hace esta aplicación?

    Permite **entrenar modelos de Machine Learning** sobre datos históricos de tres activos del Ibex 35
    (Santander, Telefónica y Repsol) y **simular estrategias de trading** basadas en las predicciones.

    Sigue las cuatro fases de la Actividad 3:

    1. **Preparación de datos** — Limpieza, alineación temporal, indicadores técnicos y variables externas.
    2. **Clasificación supervisada** — Entrenamiento y comparación de 4 modelos (Random Forest, Regresión Logística, Naive Bayes, Árbol de Decisión).
    3. **Estrategia de trading** — Backtest con reglas de gestión monetaria.
    4. **Análisis** — Interpretación y comparativa frente a Buy & Hold y al Ibex 35.
    """)
    st.stop()


R = st.session_state.results

tab_data, tab_model, tab_compare_models, tab_backtest, tab_assets, tab_conclusions = st.tabs([
    "📊 Datos", "🤖 Modelo", "⚖️ Comparativa modelos",
    "💰 Backtest", "🌐 Comparativa activos", "📋 Conclusiones"
])


# ============================ TAB DATOS ============================ #

with tab_data:
    st.markdown(f"### Análisis exploratorio · {asset_name}")

    col1, col2, col3, col4 = st.columns(4)
    df = R['df']
    rets = df['Close'].pct_change().dropna()
    total_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    ann_ret = ((1 + total_ret/100) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    ann_vol = rets.std() * np.sqrt(252) * 100

    col1.metric("Rentabilidad total", f"{total_ret:.1f}%")
    col2.metric("Rent. anualizada", f"{ann_ret:.1f}%")
    col3.metric("Volatilidad anual.", f"{ann_vol:.1f}%")
    col4.metric("Observaciones", f"{len(df):,}")

    train_end = R['X_train'].index[-1]
    st.plotly_chart(P.fig_price(df, train_end_date=train_end,
                                 title=f"{asset_name} - Precio histórico"),
                    use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(P.fig_returns_hist(df['Close']), use_container_width=True)
    with col_b:
        st.plotly_chart(P.fig_target_distribution(R['y_train'], R['y_test']),
                        use_container_width=True)

    st.plotly_chart(P.fig_correlation(R['X_train']), use_container_width=True)

    st.markdown("""
    **📖 Interpretación.** El conjunto de datos se ha dividido cronológicamente respetando el orden temporal:
    el modelo se entrena con el periodo inicial (azul) y se evalúa sobre el más reciente (dorado).
    La distribución de la variable objetivo permite comprobar el balance de clases — si está muy desequilibrado,
    el modelo podría sesgarse hacia la clase mayoritaria. La matriz de correlaciones ayuda a identificar
    posibles redundancias entre features.
    """)


# ============================ TAB MODELO ============================ #

with tab_model:
    st.markdown(f"### Modelo: {model_name}")

    mt = R['result']['metrics_test']
    mtr = R['result']['metrics_train']

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{mt['Accuracy']:.3f}", f"Train: {mtr['Accuracy']:.3f}")
    col2.metric("Precision", f"{mt['Precision']:.3f}", f"Train: {mtr['Precision']:.3f}")
    col3.metric("Recall", f"{mt['Recall']:.3f}", f"Train: {mtr['Recall']:.3f}")
    col4.metric("F1-Score", f"{mt['F1']:.3f}", f"Train: {mtr['F1']:.3f}")
    col5.metric("AUC", f"{mt['AUC']:.3f}", f"Train: {mtr['AUC']:.3f}")

    # Texto interpretativo automático
    gap = mtr['F1'] - mt['F1']
    overfit_msg = ""
    if gap > 0.15:
        overfit_msg = "⚠️ Se observa **sobreajuste** significativo (gap train-test alto)."
    elif gap > 0.05:
        overfit_msg = "ℹ️ Hay un ligero sobreajuste pero dentro de lo razonable."
    else:
        overfit_msg = "✅ Ajuste estable: no se observa sobreajuste relevante."

    auc_msg = "el modelo discrimina correctamente entre clases" if mt['AUC'] > 0.55 else \
              "el modelo apenas distingue entre clases (AUC cercano al azar)"

    st.info(f"""
    **📖 Interpretación de las métricas.**
    El **F1-score** en test es **{mt['F1']:.3f}** y el **AUC** es **{mt['AUC']:.3f}**, indicando que {auc_msg}.
    El **Recall** ({mt['Recall']:.3f}) mide qué porcentaje de subidas reales detecta el modelo;
    la **Precision** ({mt['Precision']:.3f}) mide cuántas de sus predicciones de subida son correctas.
    {overfit_msg}
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(P.fig_confusion_matrix(R['result']['confusion_matrix']),
                        use_container_width=True)
        st.caption("La matriz muestra aciertos en la diagonal. Idealmente la diagonal concentra los valores más altos.")
    with col_b:
        fpr, tpr = R['result']['roc_curve']
        st.plotly_chart(P.fig_roc(fpr, tpr, mt['AUC']), use_container_width=True)
        st.caption("Cuanto más se aleja la curva ROC de la diagonal hacia la esquina superior izquierda, mejor discrimina el modelo.")

    if R['result']['importance'] is not None:
        st.plotly_chart(P.fig_importance(R['result']['importance']),
                        use_container_width=True)
        st.caption("Variables con mayor peso en las decisiones del modelo. En modelos basados en árboles se interpreta como 'importancia'; en regresión logística, como magnitud del coeficiente.")
    else:
        st.info("Este modelo (Naive Bayes) no proporciona importancia de variables directamente.")


# =================== TAB COMPARATIVA DE MODELOS =================== #

with tab_compare_models:
    st.markdown("### Comparativa de los 4 modelos sobre el mismo dataset")

    df_comp = R['df_comp']
    # Construir el dataframe formateado sin background_gradient (que requiere matplotlib).
    # Usamos column_config con ProgressColumn para una visualización elegante sin dependencias extra.
    st.dataframe(
        df_comp,
        use_container_width=True,
        column_config={
            col: st.column_config.ProgressColumn(
                col, format="%.3f", min_value=0.0, max_value=1.0,
            ) for col in df_comp.columns
        },
    )

    st.plotly_chart(P.fig_models_comparison(df_comp), use_container_width=True)

    best = R['best_model']
    best_metrics = df_comp.loc[best]
    st.success(f"""
    **🏆 Modelo ganador (por F1-Score): {best}**

    Métricas en test: Accuracy = **{best_metrics['Accuracy']:.3f}** ·
    Precision = **{best_metrics['Precision']:.3f}** ·
    Recall = **{best_metrics['Recall']:.3f}** ·
    F1 = **{best_metrics['F1']:.3f}** ·
    AUC = **{best_metrics['AUC']:.3f}**
    """)

    st.markdown(f"""
    **📖 Interpretación.** Hemos entrenado los cuatro modelos con la misma configuración de features y
    el mismo split temporal. La elección del **F1-Score** como criterio prioritario se justifica porque
    equilibra precision y recall, lo cual es relevante en trading: queremos detectar subidas (recall)
    sin generar demasiados falsos positivos (precision).

    Nota: el modelo seleccionado actualmente en el panel para el backtest es **{model_name}**.
    Si difiere del ganador ({best}), puedes seleccionarlo arriba y volver a ejecutar.
    """)


# ============================ TAB BACKTEST ============================ #

with tab_backtest:
    st.markdown(f"### Backtest sobre el periodo de test · Modelo: {model_name}")

    bt = R['bt']
    s = bt['metrics']['strategy']
    bh = bt['metrics']['buy_hold']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valor final estrategia", f"{s['final_value']:,.0f} €",
                f"{s['total_return_pct']:.1f}% total")
    col2.metric("Valor final B&H", f"{bh['final_value']:,.0f} €",
                f"{bh['total_return_pct']:.1f}% total")
    col3.metric("Sharpe estrategia", f"{s['sharpe']:.2f}",
                f"B&H: {bh['sharpe']:.2f}")
    col4.metric("Max Drawdown estrategia", f"{s['max_drawdown_pct']:.1f}%",
                f"B&H: {bh['max_drawdown_pct']:.1f}%")

    st.plotly_chart(P.fig_equity(bt['equity'], bt['buy_hold'], R['ibex_eq']),
                    use_container_width=True)

    dd = drawdown_series(bt['equity'])
    st.plotly_chart(P.fig_drawdown(dd), use_container_width=True)

    # Tabla de métricas comparativas
    st.markdown("#### Métricas financieras")
    metrics_table = pd.DataFrame([
        {'Métrica': 'Rentabilidad total (%)', 'Estrategia IA': f"{s['total_return_pct']:.2f}", 'Buy & Hold': f"{bh['total_return_pct']:.2f}"},
        {'Métrica': 'Rentabilidad anual. (%)', 'Estrategia IA': f"{s['annual_return_pct']:.2f}", 'Buy & Hold': f"{bh['annual_return_pct']:.2f}"},
        {'Métrica': 'Volatilidad anual. (%)', 'Estrategia IA': f"{s['annual_vol_pct']:.2f}", 'Buy & Hold': f"{bh['annual_vol_pct']:.2f}"},
        {'Métrica': 'Sharpe ratio', 'Estrategia IA': f"{s['sharpe']:.2f}", 'Buy & Hold': f"{bh['sharpe']:.2f}"},
        {'Métrica': 'Sortino ratio', 'Estrategia IA': f"{s['sortino']:.2f}", 'Buy & Hold': f"{bh['sortino']:.2f}"},
        {'Métrica': 'Max Drawdown (%)', 'Estrategia IA': f"{s['max_drawdown_pct']:.2f}", 'Buy & Hold': f"{bh['max_drawdown_pct']:.2f}"},
        {'Métrica': 'Nº operaciones', 'Estrategia IA': f"{s['n_trades']}", 'Buy & Hold': "1"},
        {'Métrica': 'Win rate (%)', 'Estrategia IA': f"{s['win_rate_pct']:.1f}", 'Buy & Hold': "-"},
        {'Métrica': 'Profit factor', 'Estrategia IA': f"{s['profit_factor']:.2f}", 'Buy & Hold': "-"},
    ])
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # Interpretación
    if s['total_return_pct'] > bh['total_return_pct']:
        veredict = f"✅ La estrategia **supera** al Buy & Hold en **{s['total_return_pct'] - bh['total_return_pct']:.1f} puntos porcentuales** de rentabilidad total."
    else:
        veredict = f"⚠️ La estrategia **no supera** al Buy & Hold (diferencia: {s['total_return_pct'] - bh['total_return_pct']:.1f} pp)."
    st.info(veredict)

    # Tabla de operaciones
    if len(bt['trades']) > 0:
        st.markdown("#### Tabla de operaciones")
        trades_show = bt['trades'].copy()
        trades_show['entry_date'] = pd.to_datetime(trades_show['entry_date']).dt.strftime('%Y-%m-%d')
        trades_show['exit_date'] = pd.to_datetime(trades_show['exit_date']).dt.strftime('%Y-%m-%d')
        trades_show['return_pct'] = trades_show['return_pct'].round(2)
        trades_show['pnl'] = trades_show['pnl'].round(2)
        trades_show['entry_price'] = trades_show['entry_price'].round(4)
        trades_show['exit_price'] = trades_show['exit_price'].round(4)
        trades_show.columns = ['Entrada', 'Salida', 'Tipo', 'P. entrada', 'P. salida',
                                'Retorno %', 'P&L (€)', 'Motivo cierre']
        st.dataframe(trades_show, use_container_width=True, hide_index=True, height=350)


# ====================== TAB COMPARATIVA ACTIVOS ====================== #

with tab_assets:
    st.markdown(f"### Misma estrategia aplicada a los 3 activos · Modelo: {model_name}")

    if st.session_state.assets_comp is None or len(st.session_state.assets_comp) == 0:
        st.warning("Ejecuta la simulación para ver la comparativa.")
    else:
        ac = st.session_state.assets_comp
        st.plotly_chart(P.fig_assets_comparison(ac, initial_capital), use_container_width=True)

        # Tabla comparativa
        rows = []
        for asset, data in ac.items():
            s = data['metrics_strat']
            bh = data['metrics_bh']
            mm = data['metrics_model']
            rows.append({
                'Activo': asset,
                'Rent. estrategia (%)': f"{s['total_return_pct']:.1f}",
                'Rent. B&H (%)': f"{bh['total_return_pct']:.1f}",
                'Sharpe estrat.': f"{s['sharpe']:.2f}",
                'Max DD (%)': f"{s['max_drawdown_pct']:.1f}",
                'Nº ops.': f"{s['n_trades']}",
                'Win rate (%)': f"{s['win_rate_pct']:.1f}",
                'Accuracy modelo': f"{mm['Accuracy']:.3f}",
                'F1 modelo': f"{mm['F1']:.3f}",
            })
        df_assets = pd.DataFrame(rows)
        st.dataframe(df_assets, use_container_width=True, hide_index=True)

        # Mejor activo
        best_asset = max(ac.items(), key=lambda kv: kv[1]['metrics_strat']['sharpe'])[0]
        best_sharpe = ac[best_asset]['metrics_strat']['sharpe']
        st.success(f"🏆 Con esta configuración, el activo con mejor Sharpe es **{best_asset}** ({best_sharpe:.2f}).")

        st.markdown("""
        **📖 Interpretación.** Esta tabla permite comparar cómo se comporta la misma estrategia sobre tres
        activos del Ibex 35 con perfiles muy distintos (banca, telecomunicaciones y energía). Los resultados
        dependen mucho del activo: en algunos la estrategia bate al Buy & Hold con margen, en otros no
        compensa el coste implícito de salir y entrar del mercado.
        """)


# ============================ TAB CONCLUSIONES ============================ #

with tab_conclusions:
    st.markdown("### Resumen ejecutivo")

    bt = R['bt']
    s = bt['metrics']['strategy']
    bh = bt['metrics']['buy_hold']
    best = R['best_model']
    best_metrics = R['df_comp'].loc[best]

    bate_bh = s['total_return_pct'] > bh['total_return_pct']
    icono_bh = "✅" if bate_bh else "⚠️"

    st.markdown(f"""
    #### 🎯 Configuración

    - **Activo**: {asset_name}
    - **Periodo**: {date_range[0]} a {date_range[1]}
    - **Modelo seleccionado**: {model_name}
    - **Horizonte de predicción**: {horizon} día(s)
    - **Indicadores técnicos**: {", ".join(selected_indicators)}
    - **Variables externas**: {", ".join(selected_externals) if selected_externals else "ninguna"}
    - **Tipo de operativa**: {operativa}

    #### 🤖 Mejor modelo identificado

    El modelo con mejor **F1-score** entre los cuatro probados es **{best}**, con:
    - Accuracy = **{best_metrics['Accuracy']:.3f}**
    - F1 = **{best_metrics['F1']:.3f}**
    - AUC = **{best_metrics['AUC']:.3f}**

    #### 💰 Resultado de la estrategia (modelo {model_name})

    - Rentabilidad total: **{s['total_return_pct']:.2f}%** (anualizada **{s['annual_return_pct']:.2f}%**)
    - Sharpe ratio: **{s['sharpe']:.2f}**
    - Max Drawdown: **{s['max_drawdown_pct']:.2f}%**
    - Nº operaciones: **{s['n_trades']}** · Win rate **{s['win_rate_pct']:.1f}%** · Profit Factor **{s['profit_factor']:.2f}**

    #### 📊 Comparación con Buy & Hold

    {icono_bh} La estrategia obtiene una rentabilidad de **{s['total_return_pct']:.2f}%** frente al **{bh['total_return_pct']:.2f}%**
    del Buy & Hold del mismo activo (**{'mejor' if bate_bh else 'peor'} en {abs(s['total_return_pct'] - bh['total_return_pct']):.2f} pp**).
    El ratio de Sharpe de la estrategia ({s['sharpe']:.2f}) es {'**superior**' if s['sharpe'] > bh['sharpe'] else '**inferior**'}
    al del Buy & Hold ({bh['sharpe']:.2f}).

    #### 🔬 Propuestas de mejora

    1. **Walk-forward retraining**: reentrenar el modelo periódicamente (cada N meses) en lugar de un único entrenamiento.
    2. **Optimización de hiperparámetros**: aplicar GridSearchCV con validación temporal.
    3. **Ensemble de modelos**: combinar las predicciones de los 4 modelos (voting/stacking).
    4. **Más variables exógenas**: incorporar VIX, precios del petróleo (relevante para Repsol), tipos del BCE.
    5. **Posicionamiento dinámico**: ajustar el tamaño de posición en función de la probabilidad del modelo.
    6. **Análisis de costes de transacción**: incorporar comisiones y spreads para evaluar viabilidad operativa real.
    """)

    st.markdown("---")
    st.caption("App desarrollada como Actividad 3 - Máster en Mercados Financieros, Gestión de Carteras y Sistemas de Trading.")

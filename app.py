import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

# Configuración
st.set_page_config(page_title="Dashboard de Predicción", layout="wide")
st.title("Predicción de Ventas por Categoría (6 meses)")

# Categorías top
categorias = ["Almacenamiento", "Auriculares", "Memoria RAM", "Mouse", "Teclados"]

def suavizar_predicciones(forecast, metodo='savgol', window=3):
    """Suaviza las predicciones para evitar picos abruptos"""
    forecast = forecast.copy()
    
    if metodo == 'savgol' and len(forecast) >= window:
        forecast['yhat_smooth'] = savgol_filter(forecast['yhat'], 
                                               window_length=window, 
                                               polyorder=1)
    elif metodo == 'rolling':
        forecast['yhat_smooth'] = forecast['yhat'].rolling(window=window, center=True).mean()
        forecast['yhat_smooth'].fillna(forecast['yhat'], inplace=True)
    else:
        forecast['yhat_smooth'] = forecast['yhat']
    
    return forecast

def aplicar_factores_simulacion(forecast, factores):
    """Aplica factores de simulación a las predicciones"""
    forecast = forecast.copy()
    
    
    # Factor temporada
    if factores['temporada'] == 'Alta':
        forecast['yhat_smooth'] *= 1.15
    elif factores['temporada'] == 'Baja':
        forecast['yhat_smooth'] *= 0.85
    
    # Factor económico
    forecast['yhat_smooth'] *= factores['factor_economico']
    
    # Ajustar intervalos de confianza proporcionalmente
    ratio = forecast['yhat_smooth'] / forecast['yhat']
    forecast['yhat_upper'] *= ratio
    forecast['yhat_lower'] *= ratio
    
    return forecast

def mostrar_componentes(modelo):
    import matplotlib.pyplot as plt
    fig = modelo.plot_components(modelo.predict(modelo.history))
    st.pyplot(fig)

# Cargar modelos Prophet
@st.cache_resource
def cargar_modelos():
    modelos = {}
    for cat in categorias:
        path = Path(f"models/modelo_{cat}.joblib")
        if path.exists():
            modelos[cat] = joblib.load(path)
    return modelos

modelos = cargar_modelos()

# ===== SIDEBAR CON INPUTS INTERACTIVOS =====
st.sidebar.header("🎛️ Simulador de Escenarios")
st.sidebar.markdown("Ajusta los parámetros para simular diferentes escenarios")

# Parámetros básicos
periodos = st.sidebar.slider("Meses a predecir", 1, 12, 6)
metodo_suavizado = st.sidebar.selectbox(
    "Método de suavizado", 
    ["savgol", "rolling", "ninguno"],
    help="Suaviza las predicciones para evitar picos abruptos"
)

# Factores de simulación
st.sidebar.subheader("📊 Factores de Negocio")


temporada = st.sidebar.selectbox(
    "Temporada esperada",
    ["Normal", "Alta", "Baja"],
    help="Alta: +15%, Baja: -15%"
)

factor_economico = st.sidebar.slider(
    "Factor económico", 
    0.7, 1.3, 1.0, 0.05,
    help="0.7 = Recesión, 1.0 = Normal, 1.3 = Boom económico"
)

# Categoría específica para análisis detallado
st.sidebar.subheader("🔍 Análisis Detallado")
cat_seleccionada = st.sidebar.selectbox("Selecciona una categoría:", categorias)

# Configuración de gráficos
mostrar_intervalos = st.sidebar.checkbox("Mostrar intervalos de confianza", True)
mostrar_componentes = st.sidebar.checkbox("Mostrar componentes del modelo")

# Factores de simulación
factores = {
    'temporada': temporada,
    'factor_economico': factor_economico
}

# ===== PROCESAMIENTO DE DATOS =====
# DataFrame para todas las predicciones
df_predicciones = []

# Predicción por categoría con suavizado
for cat in categorias:
    modelo = modelos.get(cat)
    if modelo:
        future = modelo.make_future_dataframe(periods=periodos, freq="M")
        forecast = modelo.predict(future)
        
        # Aplicar suavizado
        forecast = suavizar_predicciones(forecast, metodo_suavizado)
        
        # Aplicar factores de simulación
        forecast = aplicar_factores_simulacion(forecast, factores)
        
        forecast["categoria"] = cat
        df = forecast[["ds", "yhat", "yhat_smooth", "yhat_lower", "yhat_upper", "categoria"]].copy()
        df_predicciones.append(df.tail(periodos))

# Concatenar predicciones
df_pred = pd.concat(df_predicciones)

st.subheader("📉 Evolución de predicciones por categoría (Suavizado)")
fig_lineas = go.Figure()

for cat in categorias:
    datos = df_pred[df_pred["categoria"] == cat]
    fig_lineas.add_trace(go.Scatter(
        x=datos["ds"],
        y=datos["yhat_smooth"],
        mode="lines+markers",
        name=cat,
        line=dict(width=3)
    ))

fig_lineas.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Predicción de Ventas (Suavizado)",
    legend_title="Categoría",
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig_lineas, use_container_width=True)

# ===== GRÁFICO DE BARRAS POR MES COMPARANDO CATEGORÍAS =====
st.subheader("📊 Comparación mensual entre categorías")
pivot_bar = df_pred.pivot(index="ds", columns="categoria", values="yhat_smooth")
fig_barras = go.Figure()

for cat in categorias:
    fig_barras.add_trace(go.Bar(
        name=cat,
        x=pivot_bar.index,
        y=pivot_bar[cat]
    ))

fig_barras.update_layout(
    barmode="group",
    xaxis_title="Fecha",
    yaxis_title="Predicción",
    height=500
)
st.plotly_chart(fig_barras, use_container_width=True)

# ===== ANÁLISIS DETALLADO POR CATEGORÍA =====
st.subheader(f"🔍 Análisis detallado: {cat_seleccionada}")

modelo = modelos.get(cat_seleccionada)
if modelo:
    df_real = modelo.history[['ds', 'y']]
    future = modelo.make_future_dataframe(periods=periodos, freq="ME")
    forecast = modelo.predict(future)
    
    # Aplicar suavizado y factores
    forecast = suavizar_predicciones(forecast, metodo_suavizado)
    forecast = aplicar_factores_simulacion(forecast, factores)
    
    # Crear gráfico personalizado con suavizado
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=df_real['ds'],
        y=df_real['y'],
        mode='lines+markers',
        name='Datos Reales',
        line=dict(color='blue', width=2)
    ))
    
    # Predicción suavizada
    prediccion_futura = forecast[forecast['ds'] > df_real['ds'].max()]
    fig.add_trace(go.Scatter(
        x=prediccion_futura['ds'],
        y=prediccion_futura['yhat_smooth'],
        mode='lines+markers',
        name='Predicción Suavizada',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    # Intervalos de confianza
    if mostrar_intervalos:
        fig.add_trace(go.Scatter(
            x=prediccion_futura['ds'],
            y=prediccion_futura['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=prediccion_futura['ds'],
            y=prediccion_futura['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Intervalo de Confianza',
            fillcolor='rgba(255,0,0,0.2)'
        ))
    
    fig.update_layout(
        title=f"Predicción para {cat_seleccionada}",
        xaxis_title="Fecha",
        yaxis_title="Ventas",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ===== COMPONENTES DEL MODELO =====
if mostrar_componentes:
        st.subheader("🧠 Componentes del Modelo")
        
        # Tendencia
        fig_tendencia = go.Figure()
        fig_tendencia.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['trend'],
            mode='lines',
            name='Tendencia',
            line=dict(color='#2ca02c', width=3)
        ))
        fig_tendencia.update_layout(title='Tendencia', height=300)
        
        # Estacionalidad anual
        if 'yearly' in forecast:
            fig_anual = go.Figure()
            fig_anual.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yearly'],
                mode='lines',
                name='Estacionalidad Anual',
                line=dict(color='#d62728', width=2)
            ))
            fig_anual.update_layout(title='Estacionalidad Anual', height=300)
        
        # Eventos
        if 'holidays' in forecast:
            fig_eventos = go.Figure()
            fig_eventos.add_trace(go.Bar(
                x=forecast['ds'], y=forecast['holidays'],
                name='Efecto de Eventos',
                marker_color='#9467bd'
            ))
            fig_eventos.update_layout(title='Impacto de Eventos Especiales', height=300)
        
        # Mostrar componentes en columnas
        cols = st.columns(3)
        cols[0].plotly_chart(fig_tendencia, use_container_width=True)
        
        if 'yearly' in forecast:
            cols[1].plotly_chart(fig_anual, use_container_width=True)
        
        if 'holidays' in forecast:
            cols[2].plotly_chart(fig_eventos, use_container_width=True)

st.subheader("🧪 Validación del Modelo con Datos Reales")

# Ingreso de datos reales para validar predicciones
st.markdown(f"Ingrese las compras reales de los próximos {periodos} meses:")

# Lista de inputs
compras_reales = []
for i in range(periodos):
    valor = st.number_input(f"Mes {i+1}", min_value=0.0, step=1.0, key=f"real_val_{i}")
    compras_reales.append(valor)

# Procesar si hay datos ingresados
df_val = df_pred[df_pred["categoria"] == cat_seleccionada].copy().reset_index(drop=True)
df_val = df_val.tail(periodos)

if len(compras_reales) == len(df_val):
    df_val["real"] = compras_reales
    df_val["error_abs"] = abs(df_val["real"] - df_val["yhat_smooth"])
    df_val["error_pct"] = 100 * df_val["error_abs"] / df_val["real"].replace(0, np.nan)

    # Calcular métricas
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(df_val["real"], df_val["yhat_smooth"])
    rmse = mean_squared_error(df_val["real"], df_val["yhat_smooth"])
    r2 = r2_score(df_val["real"], df_val["yhat_smooth"])
    mape = np.mean(df_val["error_pct"].dropna())

    # Mostrar métricas
    st.markdown("### 📊 Métricas de Evaluación")
    st.write(f"**MAE (Error Absoluto Medio):** {mae:.2f}")
    st.write(f"**RMSE (Raíz del Error Cuadrático Medio):** {rmse:.2f}")
    st.write(f"**R² (Coeficiente de determinación):** {r2:.2f}")
    st.write(f"**MAPE (Error Porcentual Absoluto Medio):** {mape:.2f}%")

    # Comparación visual
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=df_val["ds"], y=df_val["real"], mode="lines+markers", name="Real"))
    fig_cmp.add_trace(go.Scatter(x=df_val["ds"], y=df_val["yhat_smooth"], mode="lines+markers", name="Predicción"))
    fig_cmp.update_layout(title="Comparación: Predicción vs Real", xaxis_title="Fecha", yaxis_title="Compras")
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Mostrar tabla
    st.markdown("### 📄 Detalle por mes")
    st.dataframe(df_val[["ds", "real", "yhat_smooth", "error_abs", "error_pct"]].rename(columns={
        "ds": "Fecha",
        "yhat_smooth": "Predicción",
        "real": "Real",
        "error_abs": "Error Absoluto",
        "error_pct": "Error %"
    }).round(2), use_container_width=True)



# ===== TABLA RESUMEN Y DESCARGA =====
st.subheader("📄 Tabla resumen de predicción")

# Mostrar tabla con predicciones suavizadas
df_tabla = df_pred[['ds', 'categoria', 'yhat_smooth', 'yhat_lower', 'yhat_upper']].copy()
df_tabla.columns = ['Fecha', 'Categoría', 'Predicción', 'Límite Inferior', 'Límite Superior']
df_tabla = df_tabla.round(2)

st.dataframe(df_tabla.reset_index(drop=True), use_container_width=True)

# Botón para descargar CSV
@st.cache_data
def convertir_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv_data = convertir_csv(df_tabla)
st.download_button(
    label="⬇️ Descargar predicciones en CSV",
    data=csv_data,
    file_name="predicciones_categorias_suavizadas.csv",
    mime="text/csv"
)

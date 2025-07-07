import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path
from metrics import calcular_metricas
from graphic import grafico_historico_prediccion
# Configuración
st.set_page_config(page_title="Dashboard de Predicción", layout="wide")
st.title("Predicción de Ventas por Categoría (6 meses)")

# Categorías top
categorias = ["Almacenamiento", "Auriculares", "Memoria RAM", "Mouse", "Teclados"]

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
periodos = 6

# DataFrame para todas las predicciones
df_predicciones = []

# Predicción por categoría
for cat in categorias:
    modelo = modelos.get(cat)
    if modelo:
        future = modelo.make_future_dataframe(periods=periodos, freq="M")
        forecast = modelo.predict(future)
        forecast["categoria"] = cat
        df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "categoria"]].copy()
        df_predicciones.append(df.tail(periodos))


# Concatenar predicciones
df_pred = pd.concat(df_predicciones)

# === GRÁFICO DE LÍNEA POR CATEGORÍA ===
st.subheader("📉 Evolución de predicciones por categoría")
fig_lineas = go.Figure()
for cat in categorias:
    datos = df_pred[df_pred["categoria"] == cat]
    fig_lineas.add_trace(go.Scatter(
        x=datos["ds"],
        y=datos["yhat"],
        mode="lines+markers",
        name=cat
    ))

fig_lineas.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Predicción de Ventas",
    legend_title="Categoría",
    height=500
)
st.plotly_chart(fig_lineas, use_container_width=True)

# === GRÁFICO DE BARRAS POR MES COMPARANDO CATEGORÍAS ===
st.subheader("📊 Comparación mensual entre categorías")
pivot_bar = df_pred.pivot(index="ds", columns="categoria", values="yhat")
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



st.subheader("🔍 Análisis por categoría")

# Selector de categoría
cat_seleccionada = st.selectbox("Selecciona una categoría:", categorias)

modelo = modelos.get(cat_seleccionada)
if modelo:
    df_real = modelo.history[['ds', 'y']]
    future = modelo.make_future_dataframe(periods=periodos, freq="ME")
    forecast = modelo.predict(future)

    # # Métricas
    # mae, rmse, mape, r2 = calcular_metricas(modelo)
    # st.markdown(f"""
    # ### 📦 {cat_seleccionada} — Métricas de precisión
    # - **MAE:** {mae:.2f}
    # - **RMSE:** {rmse:.2f}
    # - **MAPE:** {mape * 100:.2f}%
    # - **R²:** {r2:.2f}
    # """)

    # Gráfico completo
    fig = grafico_historico_prediccion(df_real, forecast, cat_seleccionada)
    st.plotly_chart(fig, use_container_width=True)



# === TABLA RESUMEN Y DESCARGA ===
st.subheader("📄 Tabla resumen de predicción")
st.dataframe(df_pred.reset_index(drop=True), use_container_width=True)

# Botón para descargar CSV
@st.cache_data
def convertir_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv_data = convertir_csv(df_pred)
st.download_button(
    label="⬇️ Descargar predicciones en CSV",
    data=csv_data,
    file_name="predicciones_categorias.csv",
    mime="text/csv"
)

def grafico_historico_prediccion(df_real, forecast, categoria):
    import plotly.graph_objects as go

    ultima_fecha = df_real['ds'].max()
    forecast_real = forecast[forecast['ds'] <= ultima_fecha]
    forecast_futuro = forecast[forecast['ds'] > ultima_fecha]

    fig = go.Figure()

    # 1. Real (y observado)
    fig.add_trace(go.Scatter(
        x=df_real['ds'], y=df_real['y'],
        mode='lines+markers',
        name='Datos Reales',
        line=dict(color='blue')
    ))

    # 2. Ajuste en el pasado (yhat sobre entrenamiento)
    fig.add_trace(go.Scatter(
        x=forecast_real['ds'], y=forecast_real['yhat'],
        mode='lines',
        name='Ajuste del modelo',
        line=dict(color='rgba(255,165,0,0.6)', dash='dot')
    ))

    # 3. Predicción futura
    fig.add_trace(go.Scatter(
        x=forecast_futuro['ds'], y=forecast_futuro['yhat'],
        mode='lines+markers',
        name='Predicción futura',
        line=dict(color='orange')
    ))

    # 4. Banda de confianza
    fig.add_trace(go.Scatter(
        x=forecast_futuro['ds'], y=forecast_futuro['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_futuro['ds'], y=forecast_futuro['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.2)',
        name='Intervalo de Confianza'
    ))

    fig.update_layout(
        title=f'📈 {categoria} — Real + Ajuste + Predicción',
        xaxis_title='Fecha',
        yaxis_title='Ventas',
        height=500
    )

    return fig

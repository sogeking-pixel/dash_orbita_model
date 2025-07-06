from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def calcular_metricas(modelo):
    df_train = modelo.history[['ds', 'y']]
    forecast = modelo.predict(df_train)
    y_true = df_train['y']
    y_pred = forecast['yhat']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, mape, r2

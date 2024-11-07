import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("Data.csv", parse_dates=['Fecha'])
df.head(2)


# Seleccionamos la Fecha como el índice del DataFrame y ordenamos por esta
df.set_index('Fecha', inplace=True)
df.sort_index(inplace=True)
# Establecemos la frecuencia de los datos de forma explícita
df = df.asfreq('MS')
mod = ARIMA(df['Ordenes completas'], order=(3,1,4))
res = mod.fit()
# Generamos las predicciones y su intervalo de confianza
pred = res.get_prediction(start=pd.to_datetime('2023-08-01'), end=pd.to_datetime('2024-12-31'), dynamic=False)
pred_ci = pred.conf_int()
# Creamos el gráfico
ax = df['Ordenes completas']['2023':].plot(label='Datos observados')
pred.predicted_mean.plot(ax=ax, label='Datos simulados', alpha=.7, figsize=(18, 6))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_ylim([0, df['Ordenes completas'].max() + 100]) 
ax.set_xlabel('Fecha')
ax.set_ylabel('Número de Ordenes completas')
plt.legend()
plt.show()

# Crea el DataFrame a partir de un diccionario que contiene pred.predicted_mean
df_pred = pd.DataFrame({'Ordenes completas simulados': pred.predicted_mean})
df_pred.to_excel('datos_simulados.xlsx')

pred = res.get_prediction(start=pd.to_datetime('2023-08-01'), end=pd.to_datetime('2024-12-31'), dynamic=False)
pred_ci = pred.conf_int()
# Cálculo del MSE y R2
mse = mean_absolute_error(df['Ordenes completas'][pred.predicted_mean.index[0]:pred.predicted_mean.index[-1]], pred.predicted_mean)
r2 = r2_score(df['Ordenes completas'][pred.predicted_mean.index[0]:pred.predicted_mean.index[-1]], pred.predicted_mean)
print(f"El error cuadrático medio (MSE) es: {mse}")
print(f"El coeficiente de determinación (R2) es: {r2}")
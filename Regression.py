import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("Data.csv", parse_dates=['Fecha'])
df.head(2)

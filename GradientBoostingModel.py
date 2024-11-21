# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:25:06 2024

@author: boett
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

# Die Daten von der Yfinance Api laden
company = 'PFE.DE' #Symbolkürzung GC=F SI=F PFE.DE
start = '2023-06-01' # '2023-06-01'
end = '2024-01-01' 
data = yf.download(company, start=start, end=end)

# Daten werden vorbereitet. Skalierung der Daten zu 0 und 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


#festlegung der vorherigen tage als eingabe für die Vorhersage des nächsten Tag dienen soll
prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))

# Gradient Boosting Regression erstellen und trainieren: erläuterung der weiteren Kompenten sind in der Ausarbeitung
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='squared_error') # 'ls' für least squares loss
gbr.fit(x_train, y_train)

# Testdaten vorbereiten
test_start = dt.datetime(2023, 6, 1) #'2021-01-01'  '2023-06-01'
test_end = dt.datetime(2024, 1, 1)
test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Vorhersagen mit Gradient Boosting Regression
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

predicted_prices = gbr.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

# Datumsangaben für die Testvorhersagen auswählen
test_dates = data.index[-len(predicted_prices):]

# Plot der Testvorhersagen
plt.plot(test_dates, actual_prices[-len(predicted_prices):], color="black", label=f"Aktueller {company} Preis")
plt.plot(test_dates, predicted_prices, color="green", label=f"Prognosierter {company} Preis (Gradient Boosting)")
plt.title(f"{company} Aktie")
plt.xlabel('Datum')
plt.ylabel(f'{company} Aktien Preis')
plt.legend()
plt.xticks(rotation=45)  # Rotation der Datumsangaben

plt.savefig('GBM_gold.png')

plt.show()

# Vorhersage für den nächsten Tag
real_data = model_inputs[-prediction_days:].reshape(1, -1)
prediction = gbr.predict(real_data)
prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
print(f"Vorhersage für den nächsten Tag: {prediction}")

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:13:35 2024

@author: boett
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout

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
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Azfbau des GRU-Models: Nähere Errläuterung finden Sie in der Ausarbeitung
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


#laden der Testdaten
test_start = dt.datetime(2023, 6, 1)  #'2021-01-01'  '2023-06-01'
test_end = dt.datetime(2024, 1, 1)    # '2024-01-01'
test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#schleife um die werte von model_inputs in x_test hinzuzufügen
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Datumsangaben für die Testvorhersagen auswählen
test_dates = data.index[-len(predicted_prices):]

# Plotten der Daten
plt.plot(test_dates, actual_prices[-len(predicted_prices):], color="black", label=f"Aktueller {company} Preis")
plt.plot(test_dates, predicted_prices, color="blue", label=f"Prognosierter {company} Preis (GRU)")
plt.title(f"{company} Aktie")
plt.xlabel('Datum')
plt.ylabel(f'{company} Aktien Preis')
plt.legend()
plt.xticks(rotation=45) # Rotation der Datumsangaben

plt.savefig('GRU_gold.png')  
plt.show()





# nächsten Tag vorhersagen
real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Vorhersage für den nächsten Tag: {prediction}")

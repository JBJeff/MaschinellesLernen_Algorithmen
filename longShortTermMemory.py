# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:31:21 2024

@author: boett
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.dates as mdates

# Die Daten von der Yfinance Api laden
company = 'PFE.DE' #Symbolkürzung GC=F SI=F PFE.DE
start = '2022-06-01' # '2023-06-01'
end = '2024-01-01'   
data = yf.download(company, start=start, end=end)

# Daten werden vorbereitet. Skalierung der Daten zu 0 und 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


#festlegung der vorherigen tage als eingabe für die Vorhersage des nächsten Tag dienen soll
prediction_days = 60

#speicherung der Trainingsdaten
x_train = [] #eingabewerte mit den vorhersagen getroffen werden
y_train = [] #werte die aus den eingabewerten von x_train vorherrgesagt wird

#erstellung der trainingsdaten
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

#umwandlung in ein gängiges Format für LSTM
x_train, y_train = np.array(x_train), np.array(y_train) #von python Array zu numpy array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #anzahl der features bestimmen. Simples modell. Nur Preis, daher anzahl der features=1

# Aufbau des LSTM Models: Nähere Errläuterung finden Sie in der Ausarbeitung
model = Sequential() 
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))) # erste schicht mit 50 neuronen. Umso mehr einheiten, umso komplexer -> umso höher die wahrscheinlichkeit des overfitting
model.add(Dropout(0.2)) #nimmt 0.2 * der neuronen raus, um overfitting zu vermeiden.
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Vorhersage des nächsten Schlusskurses
model.compile(optimizer='adam', loss='mean_squared_error') 
model.fit(x_train, y_train, epochs=25, batch_size=32)


#laden der Testdaten
test_start = dt.datetime(2022, 6, 1) 
test_end = dt.datetime(2024, 1, 1)    #dt.datetime.now() 
test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values #Schlusskurse im Numpy-Array laden


total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0) #zusammenlegung der Schlusskurse aus Training- und Testdaten

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values #model_inputs enthält die schlusskurse der vergangenen Tage. Nötig für die Vorhersage des nächsten Tages
model_inputs = model_inputs.reshape(-1, 1) #konvertierung in Spaltenform. jede zeile hat eine Spalte
model_inputs = scaler.transform(model_inputs) #skalierung der daten von 0 bis 1

# Vorhersage der testdaten
x_test = []

#schleife um die werte von model_inputs in x_test hinzuzufügen
for x in range(prediction_days, len(model_inputs)): 
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test) #mit der methode .predict werden Vorhersagen der x_test gemacht
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot der Daten
plt.plot(test_data.index, actual_prices, color="black", label=f"Aktueller {company} Preis")
plt.plot(test_data.index, predicted_prices, color="orange", label=f"Prognosierter {company} Preis")
plt.title(f"{company} Aktie")
plt.xlabel('Datum')
plt.ylabel(f'{company} Aktien Preis')
plt.legend()

# x-achse Datumsformat
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50))

plt.xticks(rotation=45)

plt.savefig('LSTM_gold.png')

plt.show()

# vorhersage des nächsten Tages von den man in optimal Fall wirklich nichts weiß(auf dt.datetime.now() ändern um Echtzeitverarbeitung zu ermöglichen )
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1),0 ]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Vorhersage für den nächsten Tag: {prediction}")

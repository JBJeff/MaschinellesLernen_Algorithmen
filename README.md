## Doku folgt AUSFÜHRLICHER
Studium Ausarbeitungsprojekt

Erläuterung des Projekts finden Sie in der BDA_Ausarbeitung_Böttcher.pdf ab Seite 8.
.md Dokumentation folgt
Aktienkurs-Vorhersage mit Linearer Regression

Dieses Projekt verwendet historische Aktienkurse und eine lineare Regression, um zukünftige Kursbewegungen vorherzusagen. Die Daten werden über die Yahoo Finance API geladen, und das Modell verwendet eine lineare Regression zur Vorhersage des zukünftigen Preises einer Aktie, basierend auf den letzten 60 Tagen.

Zum ausführen "Spyder" Entwicklungsumgebung empfohlen 

# Doku folgt AUSFÜHRLICHER  
**Studium Ausarbeitungsprojekt**

Erläuterung des Projekts finden Sie in der BDA_Ausarbeitung_Böttcher.pdf ab Seite 8.  
.md Dokumentation folgt

# Aktienkurs-Vorhersage mit Maschinelles Lernen Algorithmen

Dieses Projekt verwendet historische Aktienkurse und eine lineare Regression, um zukünftige Kursbewegungen vorherzusagen. Die Daten werden über die Yahoo Finance API geladen, und das Modell verwendet eine lineare Regression zur Vorhersage des zukünftigen Preises einer Aktie, basierend auf den letzten 60 Tagen.

Zum Ausführen wird die "Spyder" Entwicklungsumgebung empfohlen.

## Installation und Setup

Stelle sicher, dass die folgenden Python-Bibliotheken installiert sind:

- **pandas** – Datenmanipulation und -analyse  
- **sklearn** – maschinelles Lernen und Vorverarbeitung  
- **numpy** – numerische Berechnungen  
- **datetime** – Datum und Uhrzeit  
- **yfinance** – Daten von Yahoo Finance  
- **matplotlib** – Visualisierung der Daten  

Verwende den folgenden Befehl, um die erforderlichen Pakete zu installieren:
pip install pandas scikit-learn numpy yfinance matplotlib


# Funktionsweise

## 1. Daten laden
Die historischen Aktienkurse einer angegebenen Aktie werden mit `yfinance` heruntergeladen. Das Beispiel verwendet `PFE.DE` für Pfizer, aber du kannst das Symbol durch jedes gewünschte Wertpapier ersetzen.

## 2. Datenvorbereitung

### Daten Extraktion und Skalierung:
Der **Schlusskurs** der Aktie wird extrahiert und auf den Bereich von 0 bis 1 skaliert, um die Leistung des Modells zu verbessern. Dies hilft, die Auswirkung von Extremwerten zu minimieren und ermöglicht es dem Modell, Muster leichter zu erkennen.

### Erstellung von Trainingsdaten:
Es wird eine **Rollensplitting-Technik** verwendet, bei der die letzten 60 Tage der Daten genutzt werden, um den nächsten Tag zu prognostizieren. Das bedeutet, dass das Modell für die Vorhersage eines zukünftigen Preises immer die letzten 60 Tage als Eingabedaten verwendet.

In der Code-Implementierung wird dies durch das Erstellen von `x_train` und `y_train` erreicht:

- **x_train**: Eine Liste von 60-tägigen Datenfenstern, die als Eingabe für das Modell dienen.
- **y_train**: Die korrespondierenden tatsächlichen Preise, die das Modell lernen soll zu prognostizieren.

## 3. Modell erstellen
Ein **lineares Regressionsmodell** wird auf den vorbereiteten Daten trainiert, um eine Vorhersage für den zukünftigen Preis der Aktie zu machen.

## 4. Vorhersage
Das Modell wird mit den **Testdaten** validiert, und der vorhergesagte Kurs wird gegen den tatsächlichen Kurs der Aktie verglichen.  
Ein Plot wird erstellt, um den Vergleich zwischen tatsächlichem und prognostiziertem Kurs zu visualisieren.

## 5. Vorhersage für den nächsten Tag
Die Vorhersage für den nächsten Tag wird berechnet und ausgegeben.

# Ergebnis

- Das Skript erstellt ein Diagramm, das die tatsächlichen und prognostizierten Preise für die Aktie über den Testzeitraum anzeigt.
- Das Diagramm wird als **LR_gold.png** gespeichert und angezeigt.
- Eine Vorhersage für den nächsten Tag wird in der Konsole ausgegeben.

# Visualisierung
Am Ende des Skripts wird ein Diagramm erstellt, das den Vergleich zwischen den tatsächlichen und den prognostizierten Preisen zeigt.





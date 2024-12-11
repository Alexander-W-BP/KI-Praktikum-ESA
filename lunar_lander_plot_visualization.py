import pandas as pd
import matplotlib.pyplot as plt

# Datei mit den Daten laden
file_path = "output.txt"  # Pfad zur Datei anpassen

# Daten laden und Header verwenden
data = pd.read_csv(file_path, delimiter="\t")

# Spalten auswählen, die geplottet werden sollen (ohne "Frame")
columns_to_plot = data.columns[1:]

# Diagramm erstellen
plt.figure(figsize=(12, 8))
for column in columns_to_plot:
    plt.plot(data["Frame"], data[column], label=column)

# Diagramm beschriften
plt.title("Datenvisualisierung")
plt.xlabel("Frame")
plt.ylabel("Werte")
plt.legend()
plt.grid(True)

# Diagramm anzeigen
plt.show()

from matplotlib import animation, pyplot as plt
import numpy as np
import pandas as pd

# Datei einlesen
file_path = "output.txt"

try:
    data = pd.read_csv(file_path, delimiter="\t")
except Exception as e:
    print(f"Fehler beim Laden der Datei: {e}")
    exit()

if "X" in data.columns and "Y" in data.columns:
    x_coords = np.array(pd.to_numeric(data["X"], errors='coerce').dropna())
    y_coords = np.array(pd.to_numeric(data["Y"], errors='coerce').dropna())
else:
    print("Fehler: Spalten 'X' und 'Y' nicht gefunden.")
    exit()

# Plot initialisieren
fig, ax = plt.subplots()
scat = ax.scatter([], [], c="b", s=5, label='Trajectory')
testFrame_scat = ax.scatter([], [], c="r", s=10, label='Frame Points')  # Frame-Scatter initialisieren
ax.set(xlim=[min(x_coords) - 0.5, max(x_coords) + 0.5],
       ylim=[min(y_coords) - 0.5, max(y_coords) + 0.5],
       xlabel='X', ylabel='Y')
ax.legend()

# Aktualisierungsfunktion
def update(frame):
    # Streudiagramm aktualisieren
    x = x_coords[:frame]
    y = y_coords[:frame]
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    
    # Frame-Scatter aktualisieren (aktueller Punkt)
    x_frame = x_coords[frame]  # x-Koordinate zum aktuellen Zeitpunkt
    y_frame = y_coords[frame]  # y-Koordinate zum aktuellen Zeitpunkt
    testFrame_scat.set_offsets(np.c_[[x_frame], [y_frame]])  # Punkt aktualisieren
    
    return scat, testFrame_scat

# Animation erstellen
ani = animation.FuncAnimation(fig=fig, func=update, frames=len(x_coords), interval=30, repeat=False)
plt.show()

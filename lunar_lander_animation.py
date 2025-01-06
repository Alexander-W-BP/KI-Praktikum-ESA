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

# Überprüfen, ob benötigte Spalten vorhanden sind
required_columns = ["X", "Y", "Angle"]
if all(col in data.columns for col in required_columns):
    x_coords = np.array(pd.to_numeric(data["X"], errors='coerce').dropna())
    y_coords = np.array(pd.to_numeric(data["Y"], errors='coerce').dropna())
    angles = np.array(pd.to_numeric(data["Angle"], errors='coerce').dropna())
else:
    print(f"Fehler: Eine der benötigten Spalten {required_columns} wurde nicht gefunden.")
    exit()

# Plot initialisieren
fig, ax = plt.subplots(figsize=(10, 6))

# Landefläche zeichnen
landing_x = [-0.25, 0.25]
landing_y = [0, 0]
ax.plot(landing_x, landing_y, color='black', linewidth=2, label='Landefläche')

# Trajektorie und aktueller Punkt initialisieren
scat = ax.scatter([], [], c="b", s=5, label='Trajektorie')
current_point_scat = ax.scatter([], [], c="r", s=50, label='Aktuelle Position')  # Größerer Punkt für den aktuellen Frame

# Text für den Winkel initialisieren
angle_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Pfeil zur Anzeige des Winkels relativ zur Vertikalen initialisieren
# Wir platzieren den Pfeil in einem festen Bereich außerhalb der Hauptanimation, z.B. unten links
# Definiere die Position des Pfeils
arrow_origin_x = min(x_coords) - 0.3
arrow_origin_y = min(y_coords) - 0.3
arrow_length = 0.2  # Länge des Pfeils

# Initialisiere den Pfeil mit Richtung nach oben (0°)
angle_arrow = ax.quiver(arrow_origin_x, arrow_origin_y, 0, arrow_length,
                        angles='xy', scale_units='xy', scale=1, color='magenta', label='Winkelanzeige')

# Achsenlimits setzen
padding = 0.5
ax.set_xlim([min(x_coords) - padding, max(x_coords) + padding])
ax.set_ylim([min(y_coords) - padding, max(y_coords) + padding])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Lunar Lander Animation')
ax.legend(loc='upper right')

# Aktualisierungsfunktion
def update(frame):
    # Trajektorie bis zum aktuellen Frame aktualisieren
    x = x_coords[:frame]
    y = y_coords[:frame]
    data_traj = np.stack([x, y]).T
    scat.set_offsets(data_traj)
    
    # Aktuellen Punkt aktualisieren
    if frame < len(x_coords):
        x_frame = x_coords[frame]
        y_frame = y_coords[frame]
        current_point_scat.set_offsets(np.c_[x_frame, y_frame])
        
        # Winkel aktualisieren
        angle = angles[frame]
        angle_text.set_text(f'Winkel: {angle:.2f}°')
        
        # Pfeil zur Anzeige des Winkels relativ zur Vertikalen aktualisieren
        # Wenn Winkel = 0°, zeigt der Pfeil nach oben (0, 1)
        # Positive Winkel drehen den Pfeil im Uhrzeigersinn, negative gegen den Uhrzeigersinn
        angle_rad = np.deg2rad(angle)
        dx = arrow_length * np.sin(angle_rad)
        dy = arrow_length * np.cos(angle_rad)
        angle_arrow.set_UVC(dx, dy)  # Setzt die Richtungsvektoren des Pfeils
        
    return scat, current_point_scat, angle_text, angle_arrow

# Animation erstellen
ani = animation.FuncAnimation(fig, func=update, frames=len(x_coords),
                              interval=30, repeat=False)

plt.show()

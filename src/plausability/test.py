import matplotlib.pyplot as plt

# Daten für das Diagramm
sss_scores = {
    "Orig.": 1.0,
    "Synth. 1": 0.89,
    "Synth. 2": 0.86,
    "Synth. 3": 0.78,
    "Synth. 4": 0.71,
}

# Erstellen der Figur und der Achsen
plt.figure(figsize=(8, 5))

# Farben definieren
color_original = '#1f77b4'  # Blau für "Original"
color_synthetic = '#ff7f0e' # Orange für "Synthetisch"

# Balkendiagramm erstellen
bars = plt.bar(sss_scores.keys(), sss_scores.values(), edgecolor='black')

# Farbe für den "Original"-Balken (blau)
bars[0].set_color(color_original)

# Farbe für den ersten "Synthetisch"-Balken (dunkles Orange)
bars[1].set_color(color_synthetic)

# Farbe und Transparenz für die restlichen "Synthetisch"-Balken (blasses Orange)
for bar in bars[2:]:
    bar.set_color(color_synthetic)
    bar.set_alpha(0.6)  # Alpha-Wert für einen "blassen" Effekt

# Beschriftungen über den Balken hinzufügen
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Titel und Achsenbeschriftungen
plt.ylabel("Structure Consistency Score (SSS)", fontsize=12)
plt.title("SCS-Vergleich: Original vs. synthetische Datensätze", fontsize=14, fontweight='bold')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()

# Diagramm anzeigen
plt.show()
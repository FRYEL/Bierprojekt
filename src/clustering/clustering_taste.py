import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1) Parameter anpassen
INPUT_CSV = "original_dataset_schema.csv"      # Pfad zur CSV
NA_VALUES = -99                    # fehlende Werte
K_MIN, K_MAX = 2, 10               # Range für Elbow/Silhouette
K_OPT = 4                          # finales k (nach Auswertung)

# 2) Daten einlesen
df = pd.read_csv(INPUT_CSV, na_values=NA_VALUES)
taste_cols = [
    "geschmack_rauchig", "geschmack_hopfig", "geschmack_suesslich_malzig",
    "geschmack_fruchtig", "geschmack_kraeuter", "geschmack_gewuerze",
    "geschmack_bitter", "geschmack_saeuerlich", "geschmack_zitrus"
]
X = df[taste_cols]

# 3) Imputation + Skalierung
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# 4) Elbow- und Silhouette-Analyse
inertias = []
sil_scores = []
Ks = range(K_MIN, K_MAX + 1)

for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Plot Elbow
plt.figure()
plt.plot(Ks, inertias, marker="o")
plt.xlabel("Anzahl Cluster k")
plt.ylabel("Inertia")
plt.title("Elbow-Methode")
plt.tight_layout()
plt.show()

# Plot Silhouette
plt.figure()
plt.plot(Ks, sil_scores, marker="o")
plt.xlabel("Anzahl Cluster k")
plt.ylabel("Silhouette-Score")
plt.title("Silhouette-Analyse")
plt.tight_layout()
plt.show()

# 5) Clustering mit finalem k
kmeans = KMeans(n_clusters=K_OPT, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# 6) Cluster-Größen
counts = df["cluster"].value_counts().sort_index()
perc   = df["cluster"].value_counts(normalize=True).sort_index() * 100
cluster_sizes = pd.DataFrame({
    "Cluster": counts.index,
    "Anzahl Personen": counts.values,
    "Prozent (%)": perc.round(1).values
})

# 7) Cluster-Zentren (auf Originalskala)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=taste_cols)
centers_df.insert(0, "Cluster", centers_df.index)

# 8) Zusammenfassungstabelle
summary = pd.merge(cluster_sizes, centers_df, on="Cluster")
print("\n=== Cluster-Übersicht ===")
print(summary.to_string(index=False))

# Optional: Tabelle speichern
summary.to_csv("cluster_summary.csv", index=False)
print("\nTabelle gespeichert als 'cluster_summary.csv'.")

# 9) Heatmap der Geschmacksprofile
heatmap_data = summary.set_index("Cluster")[taste_cols].values
clusters = summary["Cluster"].astype(str).tolist()

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(heatmap_data, aspect="auto")

# Achsen-Beschriftung
ax.set_yticks(np.arange(len(clusters)))
ax.set_yticklabels(clusters)
ax.set_xticks(np.arange(len(taste_cols)))
ax.set_xticklabels(taste_cols, rotation=45, ha="right")
ax.set_xlabel("Geschmack")
ax.set_ylabel("Cluster")
ax.set_title("Cluster-Geschmacksprofile")

# Werte annotieren
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        ax.text(j, i,
                f"{heatmap_data[i, j]:.1f}",
                ha="center", va="center",
                color="white" if heatmap_data[i, j] < np.mean(heatmap_data) else "black")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Durchschnittsbewertung")

plt.tight_layout()
plt.show()

# 10) PCA-Visualisierung
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
for cl in range(K_OPT):
    mask = df["cluster"] == cl
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=f"Cluster {cl}", alpha=0.7)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA-Visualisierung")
plt.legend()
plt.tight_layout()
plt.show()

# 11) t-SNE-Visualisierung
tsne = TSNE(n_components=2, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()
for cl in range(K_OPT):
    mask = df["cluster"] == cl
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                label=f"Cluster {cl}", alpha=0.7)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE-Visualisierung")
plt.legend()
plt.tight_layout()
plt.show()

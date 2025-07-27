# Vierdimensionales Evaluationsframework für synthetische Bier-Sensordaten

**Autor\*innen:** Furkan Yel, Hosung Ryu, Abdulaziz Al-Surabi
**Stand:** 27.07.2025

---

## Zusammenfassung

Dieses Repository enthält ein Python-basiertes Framework zur Evaluierung synthetischer tabellarischer Daten aus einer Bierpräferenz-Studie. Das Framework bewertet Daten entlang von vier Dimensionen:

1. **Statistische Wiedergabetreue (Fidelity)**
2. **Strukturelle Konsistenz (Structure)**
3. **Praktische Nutzbarkeit (Usability)**
4. **Semantische Erklärbarkeit (Explainability)**



---
## Inhaltsverzeichnis
1. [Datengrundlage & Synthese](#1-datengrundlage--synthese)  
2. [Projektstruktur](#2-projektstruktur)  
3. [Installation & Setup](#3-installation--setup)  
4. [Modulbeschreibungen](#4-modulbeschreibungen)  
   * [Dimension 1: Statistische Wiedergabetreue (Fidelity)](#dimension-1-statistische-wiedergabetreue-fidelity)  
   * [Dimension 2: Strukturelle Konsistenz (Structure)](#dimension-2-strukturelle-konsistenz-structure)  
   * [Dimension 3: Praktische Nutzbarkeit (Usability)](#dimension-3-praktische-nutzbarkeit-usability)  
   * [Dimension 4: Semantische Erklärbarkeit (Explainability)](#dimension-4-semantische-erklarbarkeit-explainability)  


---

# 1. Datengrundlage & Synthese

### Originaldaten

* Online-Umfrage zu Bierpräferenzen (Gen Z) in Baden-Württemberg
* Erhebungszeitraum: 04.10.–06.12.2024
* **n = 524** bereinigte Antworten
* Demografische Angaben, Konsumverhalten, Geschmacksvorlieben (Likert 1–5), Kaufentscheidungsfaktoren
* Fehlende Werte codiert als `-99`, werden in Preprocessing-Skripten ersetzt

### Synthetische Datengenerierung

* **CTGAN** über `sdv.single_table.CTGANSynthesizer`
* Imputation: Median (numerisch) / Modalwert (kategorial)
* `SingleTableMetadata` definiert Datentypen, Kategorien, Ordnungsrelationen
* Training: 500 Epochen, Batch-Größe 500

---

# 2. Projektstruktur

```bash
EvaluationFramework/
├── data/                      # Original- & synthetische CSVs
├── src/
│   ├── data_generation/       # CTGAN-Pipeline
│   ├── statistical_analysis/  # Fidelity (sdmetrics)
│   ├── structural_consistency # Structure (PCA, t-SNE, Clustering)
│   ├── usability/             # Usability (TSTR mit XGBoost)
│   └── plausibility/          # Plausibility (Regeln & LLM)
├── requirements.txt           # Pip-Abhängigkeiten
└── documentation.md           # Code-Dokumentation
```

---

# 3. Installation & Setup

1. **Repository klonen**

```bash
git clone [https://github.com/FRYEL/Bierprojekt.git](https://github.com/FRYEL/Bierprojekt.git)
cd Bierprojekt
```

2. **Umgebung anlegen**

*Conda*
```bash
conda env create -f environment.yml
conda activate bierprojekt
```
*venv*
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


---
# 4. Modulbeschreibungen

### Dimension 1: Statistische Wiedergabetreue (Fidelity)

Datei: `src/statistical_analysis/compute_fidelity.py`

In dieser Dimension wird eine umfassende, benutzerdefinierte "Statistical Similarity Score (SSS)" berechnet, die verschiedene Submetriken für jede Feature-Spalte kombiniert.

* **Submetriken (pro Feature):**

  * **Mean & Median:** Differenz der Mittelwerte bzw. Mediane, normiert auf den Wertebereich
  * **Variance:** Differenz der Varianzen, normiert auf maximale Varianz
  * **Kolmogorov-Smirnov-Test (KS):** P-Wert aus `ks_2samp`
  * **Wasserstein-Distanz:** Umgewandelt zu Ähnlichkeit: `1/(1+distance)`
  * **Maximum Mean Discrepancy (MMD):** RBF-Kernel-basiertes MMD, umgewandelt zu Ähnlichkeit
  * **KL-Divergenz:** Histogramm-basierte KL, transformiert via `exp(-alpha * KL)`
  * **Coverage:** Anteil gemeinsamer Werte (Kategorieüberlappung)
  * **Chi-Quadrat (nur kategorial):** P-Wert aus `chisquare`

* **Gewichtung:**
  Submetriken können per `weights`-Argument gewichtet werden (Default: gleichgewichtet).

* **Berechnung:**
  `compute_subscores(original_df, synthetic_df, alpha_kl, mmd_gamma, weights)` gibt ein Dictionary mit:

  * `per_feature`: Für jede Spalte ein Dict der Subscores und `feature_score`
  * `SSS`: Durchschnittlicher Feature-Score (Overall Statistical Similarity Score)

```python
from src.statistical_analysis.compute_fidelity import compute_subscores
import pandas as pd

# Daten laden
df_real = pd.read_csv('data/real/real.csv')
df_syn = pd.read_csv('data/synthetic/synth.csv')

# Optionale Custom Weights
delta_weights = {
    'mean': 0.05, 'median': 0.05, 'variance': 0.05,
    'ks': 0.1, 'chi2': 0.1, 'wasserstein': 0.2,
    'mmd': 0.2, 'kl': 0.2, 'coverage': 0.05
}

# Subscores und Gesamt-Score berechnen
result = compute_subscores(df_real, df_syn, alpha_kl=1.0, mmd_gamma=1.0, weights=delta_weights)
print("Statistical Similarity Score (SSS):", result['SSS'])
# Details pro Feature
for feat, vals in result['per_feature'].items():
    print(f"Feature: {feat}, Score: {vals['feature_score']:.2f}")
```

* **Visuelle Analyse:**

  * Overlay-Histogramme und Dichtekurven mit `seaborn.kdeplot` zur Gegenüberstellung der Verteilungen
  * Boxplots zum Vergleich zentraler Tendenzen und Spread
  * Paarweise Scatterplots (`itertools.combinations`) zur Visualisierung von Feature-Kopplungen
  * Korrelations-Heatmaps (Seaborn `heatmap`), um Differenzen in den Korrelationsstrukturen darzustellen

---

---

### Dimension 2: Strukturelle Konsistenz (Structure)

Datei: `src/structural_consistency/consistency.py`

Diese Dimension quantifiziert, wie gut die synthetischen Daten die latenten Strukturen und Clusterkonfigurationen der Originaldaten bewahren. Sie besteht aus einem universellen Structural Consistency Score und verschiedenen visuellen Analysen.

**1. Structural Consistency Score**
Funktion: `compute_structural_similarity_score(original: DataFrame, synthetic: DataFrame, eps=0.5, min_samples=5, weights=None) -> float`

* **Subscores:**

  * **Cluster Purity (KMeans & DBSCAN):** Reinheit der Cluster im Joint-Datensatz (Original vs. Synthetic)
  * **Adjusted Rand Index (KMeans & DBSCAN):** Übereinstimmung der Clusterlabels (1 = perfekt)
  * **Cosine Similarity:** Kosinus-Ähnlichkeit der globalen Mittelvektoren beider Datensätze, skaliert auf \[0,1]

* **Gewichtung:**
  Standardgewichte (Default):

  ```yaml
  purity_km: 0.15
  purity_db: 0.15
  ARI_km:    0.15
  ARI_db:    0.15
  cosine:    0.40
  ```

  Anpassung über `weights`-Parameter möglich.

* **Rückgabe:**
  Ein einzelner Score ∈ \[0,1], wobei 1 höchste strukturelle Übereinstimmung bedeutet.

```python
from src.structural_consistency.consistency import compute_structural_similarity_score
import pandas as pd

orig = pd.read_csv('data/real/real.csv')
synth = pd.read_csv('data/synthetic/synth.csv')
score = compute_structural_similarity_score(orig, synth)
print(f"Structural Consistency Score: {score:.3f}")
```

**2. Visuelle Analysen**
Zur qualitativen Begutachtung werden mehrere Verfahren genutzt:

* **PCA-Scatterplots:**

  * PCA auf numerischen Features, Training auf Originaldaten, Transformation beider Datensätze.
  * Side-by-side-Scatterplots mit identischen Achsenlimits.

* **t-SNE-Scatterplots:**

  * Unabhängige t-SNE-Projektionen für Original und Synthetic (jeweils Perplexity=30, random\_state=42).
  * Vergleich der Punktwolken auf gemeinsamen Achsen.

* **Joint t-SNE mit Zentroiden:**

  * Gemeinsame t-SNE-Transformation beider Datensätze.
  * Darstellung von Original- vs. Synthetic-Punkten in einem Plot.
  * Markierung der Zentroiden zur Illustration der Distanz der Mittelwerte.

* **KMeans-Cluster-Visualisierung:**

  * KMeans mit k=4 auf PCA- und t-SNE-Räumen.
  * Side-by-side-Darstellung der Clusterlabels für Original und Synthetic.
  * Silhouette-Scores zur quantitativen Bewertung der Clusterqualität.

* **Agglomerative Clustering:**

  * AgglomerativeClustering(n\_clusters=2) auf dem gemeinsamen t-SNE-Embedding.
  * Farbliche Trennung von Original und Synthetic Clustern, um strukturelle Distanz zu veranschaulichen.

```python
# Beispiel: PCA-Scatter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
real_pca = pca.fit_transform(orig[num_cols])
syn_pca = pca.transform(synth[num_cols])
# Plotting wie oben beschrieben...
```

---

### Dimension 3: Praktische Nutzbarkeit (Usability)

Datei: `src/usability/tstr_test.py`

Diese Dimension prüft, wie gut ein Modell, das auf synthetischen Daten trainiert wurde, auf echten Testdaten performt – im Vergleich zu einem Modell, das auf realen Daten trainiert wurde.

**1. Datenvorbereitung:**

* Einlesen von `synthetic_train.csv`, `real_train.csv` und `real_test.csv`.
* Datenbereinigung und -formatierung mit `clean_alter_column(df)`.
* One-Hot-Encoding der kategorialen Spalten (`geschlecht`, `bundesland`, `beruf`, `konsumhaeufigkeit`) mittels `one_hot_encode_dataframe()`, wobei derselbe Encoder für Trainings- und Testdaten verwendet wird.

**2. Modellaufbau und Training:**

* Klassifikator: `XGBClassifier(objective="multi:softmax", num_class=5, use_label_encoder=False, eval_metric="mlogloss", random_state=77)`
* Multi-Label-Wrapper: `MultiOutputClassifier(XGBClassifier(...))`
* Zielspalten: Alle Spalten mit Präfix `geschmack_` (Likert-Skala 1–5).
* Zwei Trainingsläufe:

  1. **TRTR (Train-on-Real, Test-on-Real):** Modell auf `real_train` trainieren.
  2. **TSTR (Train-on-Synthetic, Test-on-Real):** Modell auf `synthetic_train` trainieren.

**3. Evaluierung per Kategorie:**

* Für jede Geschmacks-Kategorie werden mittels `classification_report` Präzision, Recall und F1-Score ausgegeben.

```python
from sklearn.metrics import classification_report
# Nach dem Trainieren und Vorhersagen:
print(classification_report(y_true[:, i], y_pred[:, i], digits=3, zero_division=0))
```

**4. Klassenspezifische Accuracy-Visualisierung:**

* Berechnung der Accuracy pro Likert-Stufe (1–5) für beide Modelle.
* Balkendiagramme für jede Kategorie: Vergleich Real vs. Synthetisch.

```python
# Visualisierung pro Kategorie
# axs[i].bar(...)
```

**5. Aggregierte Metriken:**

* MAE (`mean_absolute_error`) und Cohen's Kappa (`cohen_kappa_score` mit `weights='quadratic'`) für jede Kategorie.
* Ergebnis in einem `pandas.DataFrame` mit Spalten: `MAE_Real`, `MAE_Synthetisch`, `Kappa_Real`, `Kappa_Synthetisch`.
* Balkendiagramm zum Vergleich der MAE-Werte.

```python
import pandas as pd
# DataFrame erstellen und plotten
mae_kappa_df["MAE_Real","MAE_Synthetisch"].plot(kind="bar")
```

---

### Dimension 4: Semantische Erklärbarkeit (Explainability)

Datei: `src/explainability/dynamic_evaluator.py`

Diese Dimension kombiniert ein regelbasiertes Verfahren mit LLM-gestützter Validierung, um die semantische Konsistenz der synthetischen Daten im Bier-Sensorik-Kontext zu überprüfen.

**1. Regelbasierte Evaluierung**

* Laden von Plausibilitätsregeln aus `beer_rules.json` (manuell erstellte Expert\*innenregeln).
* Klasse: `PlausibilityRule` (Attribute: `rule_id`, `title`, `description`, `detection_logic`).
* Anwendung: `DataEvaluationAgent` prüft jede Zeile gegen alle Regeln und berechnet Punktabzug (100 ÷ AnzahlRegeln pro Verstoß).

**2. Dynamische Regelerzeugung**

* Klasse: `RuleDiscoveryAgent`

  * Analyse des Real-Datensatzes (`pandas.DataFrame`).
  * Extraktion von Plausibilitätsregeln durch System-Prompt an GPT-4o.
  * Ausgabe: JSON-Liste von `PlausibilityRule`-Objekten.

**3. LLM-basierte Validierung**

* Klasse: `DataEvaluationAgent`

  * Initialisierung mit Regelwerk (statisch oder dynamisch) und Modell-Parameter (`model='gpt-4o-mini'`).
  * Methode `evaluate_row()`: Sendet für jede Datenzeile das Regelwerk und die Werte an das LLM.
  * Rückgabe: JSON mit Feldern `quality_score` (0–100) und `violation_details`.

**4. Orchestrator**

* Klasse: `DynamicDataEvaluator`

  * Zusammenschluss von `RuleDiscoveryAgent` und `DataEvaluationAgent`.
  * Methode `run(input_csv, output_csv, mode)`:

    * `mode='static'`: Nutzt vorgegebene Regeln.
    * `mode='dynamic'`: Generiert Regeln zuerst automatisch.
  * Speichert das Ergebnis als CSV mit zusätzlichen Spalten: `quality_score`, `rule_violations`, `violation_details`, `quality_summary`.


## 5 Ausführung

* **Notebooks:** Alle Jupyter-Notebooks (`.ipynb`) im Repository lassen sich im aktivierten virtuellen Environment direkt mit **Jupyter Lab** oder **Jupyter Notebook** öffnen und ausführen. Alternativ können sie via CLI ausgeführt werden, z.B.:

  ```bash
  jupyter nbconvert --to notebook --execute notebooks/Usability_Test.ipynb
  ```

* **Plausibility (Dimension 4):** Die semantische Validierung muss über das Python-Skript `dynamic_evaluator.py` ausgeführt werden. Beispielaufruf:

  ```bash
  python src/plausibility/dynamic_evaluator.py data/real.csv \
    --evaluate-file data/synthetic.csv \
    --rules-file beer_rules.json \
    --output results/synthetic_evaluation.csv
  ```

* **API-Schlüssel:** Für den LLM-Zugriff verwendet `dynamic_evaluator.py` die OpenAI-API. Stellen Sie sicher, dass Ihr API-Token in der Umgebungsvariable `OPENAI_API_KEY` verfügbar ist:

  ```bash
  export OPENAI_API_KEY="<Ihr_OpenAI_API_Schlüssel>"
  ```

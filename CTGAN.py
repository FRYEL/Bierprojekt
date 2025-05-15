# State-of-the-Art CTGAN-Synthese für euren Datensatz

# 1. Bibliotheken importieren (stellt sicher, dass sdv installiert ist: pip install sdv)
import pandas as pd
from sdv.tabular import CTGAN

# 2. Datensatz einlesen
input_path = '/mnt/data/synthetic_dataset_schema.csv'
df = pd.read_csv(input_path)

# 3. Spalten definieren
#   - Ordinale Spalten (1–5-Skalen)
ordinal_columns = [
    'konsumhaeufigkeit',
    'situation_freunde_familie', 'situation_party', 'situation_zuhause_essen',
    'situation_zuhause_entspannung', 'situation_oeffentliche_veranstaltungen',
    'situation_gastro', 'situation_urlaub',
    'involvement_interesse', 'involvement_rolle', 'involvement_spass',
    'involvement_wichtigkeit', 'involvement_vorteilhaft', 'involvement_unverzichtbar',
    'gesundheit_wichtig', 'umweltfreundlich_wichtig', 'neue_produkte_probie',
    'bio_mehrpreis_bereit', 'tierfrei_wichtig', 'alkoholverzicht_wichtig',
    'einfluss_geschmack', 'einfluss_geruch', 'einfluss_preis', 'einfluss_markenbekanntheit',
    'einfluss_nachhaltigkeit', 'einfluss_herkunft', 'einfluss_verpackungsdesign',
    'einfluss_verpackungsform', 'einfluss_aktionsangebote', 'einfluss_alkoholgehalt',
    'einfluss_empfehlung', 'einfluss_verfuegbarkeit',
    'praef_sorte_alkoholfrei', 'praef_sorte_weizen', 'praef_sorte_koelsch',
    'praef_sorte_altbier', 'praef_sorte_pale_ale', 'praef_sorte_stout',
    'praef_sorte_maerzen_export', 'praef_sorte_radler', 'praef_sorte_pils',
    'praef_sorte_bock_schwarzrauchbier', 'praef_sorte_helles_lager', 'praef_sorte_craft_beer',
    'praef_alkoholfrei_0', 'praef_alkoholfrei_0_5', 'praef_bier_alkohol',
    'geschmack_rauchig', 'geschmack_hopfig', 'geschmack_suesslich_malzig',
    'geschmack_fruchtig', 'geschmack_kraeuter', 'geschmack_gewuerze',
    'geschmack_bitter', 'geschmack_saeuerlich', 'geschmack_zitrus'
]
#   - Kategorische / diskrete Spalten
categorical_columns = ['geschlecht', 'bundesland', 'beruf']

# 4. CTGAN-Modell initialisieren (Parameter anpassen nach Bedarf)
model = CTGAN(
    field_names=df.columns.tolist(),
    categorical_columns=categorical_columns,
    ordinal_columns=ordinal_columns,
    epochs=300,             # Anzahl der Trainings-Epochen
    batch_size=500,         # Batch-Größe
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    pac=10,                 # Pac für Minibatch-Discrimination
    verbose=True,
    random_state=42
)

# 5. Modell trainieren
model.fit(df)

# 6. Synthetische Daten generieren
num_samples = 1000
synthetic_data = model.sample(num_samples)

# 7. Ergebnisse anzeigen und speichern
import ace_tools as tools; tools.display_dataframe_to_user(name="Synth. Daten Vorschau", dataframe=synthetic_data.head())

output_path = '/mnt/data/synthetic_data_ctgan.csv'
synthetic_data.to_csv(output_path, index=False)

output_path

# CTGAN-Synthese für den vorbereiteten (one-hot-encodierten) Datensatz
import pandas as pd
from sdv.tabular import CTGAN

# 1. Datensatz laden
input_path = 'encoded_dataset_schema.csv'
df = pd.read_csv(input_path)

# 2. Ordinale Spalten (Likert, Präferenzen, Geschmäcker)
ordinal_columns = [
    'alter', 'konsumhaeufigkeit',
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

# 3. CTGAN initialisieren (ohne kategorische Variablen, da alles ge-one-hot-encoded ist)
model = CTGAN(
    field_names=df.columns.tolist(),
    ordinal_columns=ordinal_columns,
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    pac=10,
    verbose=True,
    random_state=42
)

# 4. Training
print("Model Fitting...")

model.fit(df)

# 5. Synthetische Daten erzeugen
num_samples = 1000
synthetic_data = model.sample(num_samples)

# 6. Ausgabe speichern
synthetic_data.to_csv("synthetic_data_ctgan.csv", index=False)
print("✅ 1000 synthetische Zeilen gespeichert als synthetic_data_ctgan.csv")

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# 1. Datensatz laden
input_path = '../../data/data_for_CTGAN/CTGAN_basedata.csv'
df = pd.read_csv(input_path)

# 2. Metadaten automatisch generieren
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# 3. Ordinale Spalten setzen
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

# Ordinale Spalten als numerisch kennzeichnen (ohne subtype oder computer_representation)
for col in ordinal_columns:
    metadata.update_column(column_name=col, sdtype='numerical')

# 4. Modell initialisieren
synthesizer = CTGANSynthesizer(metadata)

# 5. Training
print("🚀 Training CTGAN...")
synthesizer.fit(df)

# 6. data generieren
num_samples = 1000
synthetic_data = synthesizer.sample(num_samples)

# 7. Speichern
synthetic_data.to_csv("synthetic_data_ctgan.csv", index=False)
print("✅ 1000 synthetische Zeilen gespeichert als synthetic_data_ctgan.csv")
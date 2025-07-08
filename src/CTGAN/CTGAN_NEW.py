import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# 1. Datensatz laden
input_path = 'data/data_for_CTGAN/CTGAN_basedata.csv'
df = pd.read_csv(input_path)

# 2. Metadaten erzeugen und Spaltentypen verfeinern
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

# 2a) Ordinale Spalten (ohne 'alter' und 'konsumhaeufigkeit') als numerisch deklarieren
ordinal_columns = [
    'situation_freunde_familie','situation_party','situation_zuhause_essen',
    'situation_zuhause_entspannung','situation_oeffentliche_veranstaltungen',
    'situation_gastro','situation_urlaub',
    'involvement_interesse','involvement_rolle','involvement_spass',
    'involvement_wichtigkeit','involvement_vorteilhaft','involvement_unverzichtbar',
    'gesundheit_wichtig','umweltfreundlich_wichtig','neue_produkte_probie',
    'bio_mehrpreis_bereit','tierfrei_wichtig','alkoholverzicht_wichtig',
    'einfluss_geschmack','einfluss_geruch','einfluss_preis','einfluss_markenbekanntheit',
    'einfluss_nachhaltigkeit','einfluss_herkunft','einfluss_verpackungsdesign',
    'einfluss_verpackungsform','einfluss_aktionsangebote','einfluss_alkoholgehalt',
    'einfluss_empfehlung','einfluss_verfuegbarkeit',
    'praef_sorte_alkoholfrei','praef_sorte_weizen','praef_sorte_koelsch',
    'praef_sorte_altbier','praef_sorte_pale_ale','praef_sorte_stout',
    'praef_sorte_maerzen_export','praef_sorte_radler','praef_sorte_pils',
    'praef_sorte_bock_schwarzrauchbier','praef_sorte_helles_lager','praef_sorte_craft_beer',
    'praef_alkoholfrei_0','praef_alkoholfrei_0_5','praef_bier_alkohol',
    'geschmack_rauchig','geschmack_hopfig','geschmack_suesslich_malzig',
    'geschmack_fruchtig','geschmack_kraeuter','geschmack_gewuerze',
    'geschmack_bitter','geschmack_saeuerlich','geschmack_zitrus'
]
for col in ordinal_columns:
    metadata.update_column(column_name=col, sdtype='numerical')

# 2b) Kategoriale Spalten deklarieren (inkl. 'alter' und 'konsumhaeufigkeit')
categorical_columns = [
    'alter', 'konsumhaeufigkeit',
    'geschlecht', 'bundesland', 'beruf'
]
for col in categorical_columns:
    metadata.update_column(column_name=col, sdtype='categorical')

# 3. CTGAN mit Tunings initialisieren
synthesizer = CTGANSynthesizer(
    metadata,
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=2e-4,
    discriminator_lr=2e-4,
    generator_decay=1e-6,
    discriminator_decay=1e-6,
    pac=10,
    verbose=True
)

# 4. Training
print("🚀 Training CTGAN...")
synthesizer.fit(df)

# 5. Sampling
num_samples = 536
synthetic_data = synthesizer.sample(num_samples)

# 6. Speichern
output_path = "536_synthetic.csv"
synthetic_data.to_csv(output_path, index=False)
print(f"✅ {num_samples} synthetische Zeilen gespeichert als {output_path}")

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, regexp_replace, row_number
from pyspark.sql.window import Window
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Initialiser Spark
spark = SparkSession.builder.appName("Grilles tarifaires").getOrCreate()

# Charger les données depuis le fichier téléchargé avec encodage UTF-8
def load_data(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        temp_file.write(file.getbuffer())
        temp_path = temp_file.name
    df = spark.read.csv(temp_path, header=True, inferSchema=True, sep=';', encoding='UTF-8')
    # Filtrer automatiquement pour ne garder que les lignes où Photochromie et Teinte sont '0'
    df = df.filter((col('Photochromie') == 0) & (col('Teinte') == 0))
    return df

# Nettoyer les noms de colonnes
def clean_column_names(df):
    for col_name in df.columns:
        clean_name = col_name.replace(" ", "").replace(";", "_").strip()
        df = df.withColumnRenamed(col_name, clean_name)
    return df

# Fonction pour nettoyer les colonnes numériques et monétaires
def clean_numeric_and_monetary_columns(df):
    monetary_columns = [col_name for col_name in df.columns if col_name.startswith("Prix") or col_name.startswith("PV")]

    # Nettoyer les colonnes monétaires
    for column in monetary_columns:
        df = df.withColumn(column, regexp_replace(col(column), "[^0-9.,-]", ""))  # Garder chiffres, points, virgules, et signes négatifs
        df = df.withColumn(column, regexp_replace(col(column), ",", "."))  # Convertir les virgules en points
        df = df.withColumn(column, col(column).cast("float"))  # Convertir en float

    # Imputer les valeurs manquantes par la médiane pour les colonnes monétaires
    for column in monetary_columns:
        valid_values_count = df.filter(col(column).isNotNull()).count()
        if valid_values_count > 0:
            median_value = df.approxQuantile(column, [0.5], 0.25)
            if median_value:
                median_value = median_value[0]
                df = df.na.fill({column: median_value})

    return df

# Fonctions pour les tranches de prix
def Tranche_prix_unif(column):
    return when(col(column) <= 50, "0€-50€") \
            .when(col(column) <= 99, "51€-99€") \
            .when(col(column) <= 149, "100€-149€") \
            .otherwise(">150€")

def Tranche_prix_prog(column):
    return when((col(column) >= 45) & (col(column) <= 99), "45€-99€") \
            .when((col(column) >= 100) & (col(column) <= 149), "100€-149€") \
            .when((col(column) >= 150) & (col(column) <= 199), "150€-199€") \
            .when((col(column) >= 200) & (col(column) <= 249), "200€-249€") \
            .when((col(column) >= 250) & (col(column) <= 299), "250€-299€") \
            .when((col(column) >= 300) & (col(column) <= 349), "300€-349€") \
            .otherwise(">350€")

# Appliquer les tranches de prix en fonction des colonnes PV contenant "Precal"
def apply_price_tranches(df, geometrie_type):
    pv_precal_columns = [col_name for col_name in df.columns if col_name.startswith("PV") and "Precal" in col_name]

    for column in pv_precal_columns:
        if geometrie_type == "Unifocal":
            df = df.withColumn(f"Tranche_{column}", Tranche_prix_unif(column))
        elif geometrie_type == "Progressif":
            df = df.withColumn(f"Tranche_{column}", Tranche_prix_prog(column))
    return df

# Appliquer des filtres en fonction des sélections utilisateur
def apply_filters(df, marque_filter, geometrie_type, indices, mutuelle_filter, exclude_antireflets):
    # Filtrer par Geometrie
    if 'Geometrie' in df.columns:
        if geometrie_type != "Tous":
            df = df.filter(df['Geometrie'] == geometrie_type)
            # Ajouter les tranches de prix pour les colonnes PV contenant "Precal"
            df = apply_price_tranches(df, geometrie_type)

    # Filtrer par Marque
    if 'Marque' in df.columns and marque_filter:
        df = df.filter(df['Marque'].isin(marque_filter))

    # Filtrer par Indice
    if 'Indice' in df.columns:
        df = df.filter(df['Indice'].isin(indices))

    # Filtrer par la colonne de mutuelle sélectionnée
    if mutuelle_filter != "Tout":
        # Vérifier si la colonne de la mutuelle sélectionnée existe dans le DataFrame
        if mutuelle_filter in df.columns:
            df = df.filter(df[mutuelle_filter].isNotNull())
        else:
            st.warning(f"La colonne {mutuelle_filter} n'existe pas dans le fichier.")
    else:
        # Si l'utilisateur choisit "Tout", inclure toutes les mutuelles
        mutuelle_columns = [col_name for col_name in df.columns if col_name.startswith("PV_")]
        if mutuelle_columns:
            condition = None
            for col_name in mutuelle_columns:
                if condition is None:
                    condition = df[col_name].isNotNull()
                else:
                    condition |= df[col_name].isNotNull()
            df = df.filter(condition)

    # Exclure certains types d'antireflets
    if exclude_antireflets and 'Antireflets' in df.columns:
        df = df.filter(~(col("Antireflets").contains("Durci")) & ~(col("Antireflets").contains("Non")))

    return df


# Fonction pour visualiser un histogramme
def plot_histogram(df, column):
    pd_df = df.select(column).toPandas()
    plt.figure(figsize=(10, 6))
    sns.histplot(pd_df[column], bins=50, kde=True)
    plt.title(f'Distribution de {column}')
    plt.xlabel(column)
    plt.ylabel('Fréquence')
    st.pyplot(plt)

# Fonction pour visualiser la distribution des tranches de prix par catégorie
def plot_price_distribution_by_category(df):
    data_unif = df.toPandas()
    tranche_columns = [col for col in data_unif.columns if 'Tranche' in col]
    plt.figure(figsize=(15, 10))

    all_data = {}
    for column in tranche_columns:
        value_counts = data_unif[column].value_counts().sort_index()
        all_data[column] = value_counts

    plot_data = pd.DataFrame(all_data).fillna(0)
    plot_data = plot_data.apply(lambda x: x.sort_values(ascending=False))

    bar_width = 0.1
    positions = np.arange(len(plot_data))

    for i, column in enumerate(plot_data.columns):
        values = plot_data[column]
        plt.bar(positions + i * bar_width, values, width=bar_width, label=column)

    plt.title('Tranches de prix par catégorie pour les verres')
    plt.xlabel('Tranches de Prix')
    plt.ylabel('Nombre d\'Occurrences')
    plt.xticks(positions + bar_width * (len(plot_data.columns) - 1) / 2, plot_data.index, rotation=45)
    plt.legend(title='')
    plt.grid(True)
    st.pyplot(plt)

# Fonction pour visualiser les tranches de prix par catégorie et verrier
def plot_price_distribution_by_category_and_brand(df):
    filtered_pd = df.toPandas()
    tranche_columns = [col for col in filtered_pd.columns if 'Tranche' in col]

    all_data = {}
    for tranche_column in tranche_columns:
        tranche_data = filtered_pd.groupby([tranche_column, 'Marque']).size().unstack().fillna(0)
        all_data[tranche_column] = tranche_data

    fig, ax = plt.subplots(figsize=(15, 10))
    bar_width = 0.1
    positions = np.arange(len(all_data[tranche_columns[0]].index))

    for idx, (tranche_column, tranche_data) in enumerate(all_data.items()):
        bottom = np.zeros(len(positions))
        for verrier in tranche_data.columns:
            values = np.zeros(len(positions))
            for i, pos in enumerate(positions):
                if i < len(tranche_data):
                    values[i] = tranche_data.iloc[i][verrier]
            ax.bar(positions + idx * bar_width, values, width=bar_width, bottom=bottom, label=f'{verrier} ({tranche_column})' if idx == 0 else "")
            bottom += values

    ax.set_title('Distribution des Tranches de Prix par Catégorie et Verrier')
    ax.set_xlabel('Tranches de Prix')
    ax.set_ylabel('Nombre d\'Occurrences')
    ax.set_xticks(positions + bar_width * (len(tranche_columns) - 1) / 2)
    ax.set_xticklabels(all_data[tranche_columns[0]].index, rotation=45)
    ax.legend(title='Verriers et Catégories', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

# Interface Utilisateur
st.title('Facilitez votre analyse des grilles tarifaires')
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df = clean_column_names(df)  # Nettoyer les noms de colonnes
    df = clean_numeric_and_monetary_columns(df)  # Nettoyer les colonnes numériques et monétaires
    
    # Sélection des filtres
    marque_filter = st.multiselect("Filtrer les marques",  ("BBGR", "ESSILOR", "HOYA", "NIKON", "SHAMIR", "ZEISS", "SEIKO", "NOVACEL", "RODENSTOCK"), default=["ESSILOR", "HOYA"])
    geometrie_filter = st.selectbox("Filtrer par type de Geometrie", ("Tous", "Unifocal", "Progressif", "Bifocal", "Mi-Distance", "Trifocal"))
    indice_filter = st.multiselect("Choisissez les indices à inclure", options=["1500", "1600", "1670", "1740", "1590"], default=["1500"])
    
    # Liste des mutuelles 
    mutuelles_disponibles = [col for col in df.columns if col.startswith("PV_")]
    mutuelles_disponibles.insert(0, "Tout")  

    mutuelle_filter = st.selectbox("Choisissez une mutuelle", options=mutuelles_disponibles)
    
    exclude_antireflets = st.checkbox("Exclure 'Durci' et 'Non Traite' dans Antireflets")

    # Appliquer les filtres en fonction des choix
    final_df = apply_filters(df, marque_filter, geometrie_filter, indice_filter, mutuelle_filter, exclude_antireflets)

    
    # Afficher le résultat final et les analyses
    if final_df.count() > 0:
        st.write(final_df.toPandas())
        
        # Boutons pour afficher les graphiques
        if st.button("Distribution des Prix d'Achat"):
            plot_histogram(final_df, 'Prix_Achat')
        
        if st.button("Distribution des Tranches de Prix par Catégorie"):
            plot_price_distribution_by_category(final_df)
        
        if st.button("Distribution des Tranches de Prix par Catégorie et Verrier"):
            plot_price_distribution_by_category_and_brand(final_df)
    else:
        st.write("Aucune donnée ne correspond aux filtres sélectionnés.")

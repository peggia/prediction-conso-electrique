import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
# Charger les fichiers CSV
df_conso_all = pd.read_csv('df_conso_all.csv', encoding='utf-8')
df_hf_cvl_full = pd.read_csv('df_hf_cvl_full.csv', encoding='utf-8')
df = pd.read_csv('dfmlenedis.csv', encoding='utf-8')

# Ajouter les colonnes "Saison" et "Mois" pour le nommage
def nommer_saison(mois):
    if mois in [12, 1, 2]:
        return "Hiver"
    elif mois in [3, 4, 5]:
        return "Printemps"
    elif mois in [6, 7, 8]:
        return "Été"
    else:
        return "Automne"

df_conso_all['MOIS'] = pd.to_datetime(df_conso_all['DATE']).dt.month
df_conso_all['SAISON'] = df_conso_all['MOIS'].apply(nommer_saison)

# Configuration de la page Streamlit avec une disposition large et un titre personnalisé
st.set_page_config(page_title="Dashboard Énergétique", layout="wide", page_icon='logo PY²MN.png')

# Ajout de style CSS personnalisé pour correspondre au thème Enedis
st.markdown("""
    <style>
    /* Fond vert d'Enedis */
    .main {
        background-color: #88b949;
        padding: 20px;
    }
    h1 {
        color: #00509e;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
    }
    h2 {
        color: #00509e;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        font-size: 2rem;
    }
    /* Style du logo */
    .logo {
        position: absolute;
        top: 10px;
        left: 10px;
        width: 120px;
    }
    /* Boutons stylisés */
    .stButton button {
        background-color: #00509e;
        color: white;
        border-radius: 20px;
        padding: 12px 24px;
        font-size: 18px;
        border: 2px solid #88b949;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #88b949;
        color: #fff;
    }
    /* Champs de texte */
    .stTextInput input {
        border-radius: 10px;
        padding: 12px;
        border: 2px solid #00509e;
        font-size: 14px;
    }
    /* Conteneur des blocs */
    .block-container {
        padding: 30px;
        max-width: 1500px;
    }
    /* Style des sections */
    .section {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 50px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Insertion du logo
st.image('enedis.png', width=120)

# Titre principal de la page
st.markdown("<h1>Tableau de bord énergétique - Enedis</h1>", unsafe_allow_html=True)

# Barre latérale pour la navigation entre les sections
st.sidebar.title("Navigation")
section = st.sidebar.radio("Aller à :", 
                           ["Section 1 : Visualisation de la consommation", 
                            "Section 2 : Visualisation consommation et météo", 
                            "Section 3 : Prédiction basée sur données historiques"])

# Couleurs distinctes pour chaque région (adaptées aux daltoniens)
region_colors = {
    'Hauts-de-France': '#d73027',  # Red
    'Centre-Val de Loire': '#4575b4',  # Blue
    'Nouvelle-Aquitaine': '#fdae61',  # Orange
    'Île-de-France': '#fee090',  # Light yellow
    'Grand-Est': '#74add1',  # Light blue
    'Normandie': '#f46d43',  # Coral
    'Bretagne': '#a50026',  # Dark red
    'Auvergne-Rhône-Alpes': '#313695',  # Dark blue
    'Bourgogne-Franche-Comté': '#1a9850',  # Green
    'Pays de la Loire': '#66c2a5',  # Light Green
    'Provence-Alpes-Côte d\'Azur': '#3288bd',  # Medium blue
    'Occitanie': '#ffcc33',  # Light orange
    'Corse': '#f4a582'  # Peach
}

# ---------------------------------------------------------------------------
# Section 1 : Visualisation de la consommation d'énergie par région et périodes
# ---------------------------------------------------------------------------
if section == "Section 1 : Visualisation de la consommation":
    st.header("Section 1 : Visualisation de la consommation d'énergie par région et périodes")
    
    st.markdown("""
    ### Explication :
    Cette section vous permet de visualiser la consommation d'énergie par région, par mois et par saison, ainsi que d'autres analyses comme la corrélation avec les points de soutirage.
    """)

    # Visualisation 1 : Consommation par région
    fig1 = px.bar(df_conso_all, x='REGION', y='ENERGIE_SOUTIREE', color='REGION',
                  color_discrete_map=region_colors,
                  title="Consommation d'énergie par région")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre la répartition de la consommation d'énergie par région.")

    # # Visualisation 2 : Consommation par mois
    # fig2 = px.bar(df_conso_all, x='MOIS', y='ENERGIE_SOUTIREE', color='REGION',
    #               color_discrete_map=region_colors,
    #               title="Consommation par mois")
    # st.plotly_chart(fig2, use_container_width=True)
    # st.markdown("**Utilité :** Ce graphique illustre la consommation mensuelle d'énergie, permettant d'identifier les périodes de pic.")

    # Visualisation 3 : Consommation par saison (Box plot)
    fig3 = px.box(df_conso_all, x='SAISON', y='ENERGIE_SOUTIREE', color='REGION',
                  color_discrete_map=region_colors,
                  title="Consommation par saison")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre la variation de la consommation d'énergie selon les saisons.")

    # Visualisation 4 : Répartition de la consommation par région (pie chart)
    fig4 = px.pie(df_conso_all, names='REGION', values='ENERGIE_SOUTIREE',
                  color_discrete_map=region_colors,
                  title="Répartition de la consommation par région")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique en secteur montre la part de chaque région dans la consommation totale.")

    # Visualisation 5 : Nombre de points de soutirage par région
    fig5 = px.bar(df_conso_all, x='REGION', y='NB_POINTS_SOUTIRAGE', color='REGION',
                  color_discrete_map=region_colors,
                  title="Nombre de points de soutirage par région")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**Utilité :** Il montre le nombre de points de soutirage par région, essentiel pour comprendre l'infrastructure.")

    # Carte interactive de la consommation d'énergie par région
    st.markdown("### Carte interactive de la consommation par région")
    
    # Simuler les coordonnées géographiques des régions
    geo_data = {
        'Hauts-de-France': [50.6292, 3.0573],
        'Centre-Val de Loire': [47.7516, 1.6751],
        'Nouvelle-Aquitaine': [44.8378, -0.5792],
        'Île-de-France': [48.8566, 2.3522],
        'Grand-Est': [48.5734, 7.7521],
        'Normandie': [49.1829, -0.3707],
        'Bretagne': [48.1173, -1.6778],
        'Auvergne-Rhône-Alpes': [45.764, 4.8357],
        'Bourgogne-Franche-Comté': [47.2805, 5.9993],
        'Pays de la Loire': [47.2184, -1.5536],
        'Provence-Alpes-Côte d\'Azur': [43.9352, 6.0679],
        'Occitanie': [43.6045, 1.4442],
        'Corse': [42.0396, 9.0129]
    }
      # Ajouter les coordonnées au DataFrame
    df_conso_all['LAT'] = df_conso_all['REGION'].map(lambda x: geo_data[x][0] if x in geo_data else None)
    df_conso_all['LON'] = df_conso_all['REGION'].map(lambda x: geo_data[x][1] if x in geo_data else None)

    # Filtrer les lignes avec des valeurs manquantes dans les coordonnées
    df_conso_all = df_conso_all.dropna(subset=['LAT', 'LON'])

    # Créer la carte interactive
    fig_map = px.scatter_mapbox(df_conso_all, lat='LAT', lon='LON', size='ENERGIE_SOUTIREE',
                                color='REGION', color_discrete_map=region_colors,
                                hover_name='REGION', hover_data=['ENERGIE_SOUTIREE', 'NB_POINTS_SOUTIRAGE'],
                                title="Carte interactive de la consommation d'énergie par région", 
                                mapbox_style="open-street-map", zoom=5)
    fig_map.update_layout(mapbox_zoom=5, mapbox_center={"lat": 46.603354, "lon": 1.888334})  # Centrer sur la France
    st.plotly_chart(fig_map, use_container_width=True)

    # Visualisation 6 : Consommation par région
    df_conso_all['CONSO_PAR_REGION'] = df_conso_all['ENERGIE_SOUTIREE'] / df_conso_all['NB_POINTS_SOUTIRAGE']
    fig6 = px.bar(df_conso_all, x='REGION', y='CONSO_PAR_REGION', color='REGION',
                  color_discrete_sequence=px.colors.qualitative.Alphabet,
                  title="Consommation par rapport au nombre de points soutirage par région")
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique permet de comparer la consommation d'énergie par habitant selon les régions.")

    # Visualisation 7 : Histogramme de la consommation d'énergie
    # fig7 = px.histogram(df_conso_all, x='ENERGIE_SOUTIREE', nbins=50, color='REGION',
    #                     color_discrete_sequence=px.colors.qualitative.Pastel,
    #                     title="Histogramme des consommations d'énergie")
    # st.plotly_chart(fig7, use_container_width=True)
    # st.markdown("**Utilité :** Cet histogramme montre la distribution des consommations d'énergie.")

    # Visualisation 8 : Corrélation entre la consommation et le nombre de points de soutirage
    # fig8 = px.scatter(df_conso_all, x='NB_POINTS_SOUTIRAGE', y='ENERGIE_SOUTIREE', color='REGION',
    #                   color_discrete_sequence=px.colors.qualitative.Plotly,
    #                   title="Corrélation entre la consommation et les points de soutirage")
    # st.plotly_chart(fig8, use_container_width=True)
    # st.markdown("**Utilité :** Cette corrélation est utile pour comprendre la relation entre infrastructure et consommation.")

    fig21 = px.line(df_conso_all, x='DATE', y='ENERGIE_SOUTIREE', color='REGION',
                color_discrete_sequence=px.colors.qualitative.T10,
                title="Séries temporelles de la consommation d'énergie")
    st.plotly_chart(fig21, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre l'évolution de la consommation d'énergie dans le temps, permettant de visualiser les tendances et les pics de consommation.")
    
    fig23 = px.treemap(df_conso_all, path=['SAISON', 'REGION'], values='ENERGIE_SOUTIREE',
                   color='ENERGIE_SOUTIREE', hover_data=['REGION'],
                   title="Treemap de la consommation d'énergie par région et saison",
                   color_continuous_scale='Viridis')
    st.plotly_chart(fig23, use_container_width=True)
    st.markdown("**Utilité :** Le treemap montre la répartition de la consommation par région et par saison de manière hiérarchique.")
    
    # df_cascade = df_conso_all.groupby('MOIS')['ENERGIE_SOUTIREE'].sum().reset_index()
    # df_cascade['Variation'] = df_cascade['ENERGIE_SOUTIREE'].diff().fillna(df_cascade['ENERGIE_SOUTIREE'])
    # fig24 = px.bar(df_cascade, x='MOIS', y='Variation', color='Variation',
    #            color_continuous_scale='RdYlGn', title="Diagramme en cascade de la consommation d'énergie")
    # st.plotly_chart(fig24, use_container_width=True)
    # st.markdown("**Utilité :** Ce diagramme montre les changements mensuels dans la consommation d'énergie, permettant d'identifier les hausses et les baisses importantes.")

# ---------------------------------------------------------------------------
# Section 2 : Visualisation consommation et météo
# ---------------------------------------------------------------------------
elif section == "Section 2 : Visualisation consommation et météo":
    st.header("Section 2 : Visualisation consommation et météo")
    
    st.markdown("""
    ### Explication :
    Cette section vous permet d'analyser les relations entre la consommation d'énergie et les variables météorologiques telles que la température, les précipitations, et l'humidité.
    """)

    # Visualisation 1 : Température maximale vs consommation
    fig11 = px.scatter(df_hf_cvl_full, x='MAX_TEMPERATURE_C', y='ENERGIE_SOUTIREE', color='REGION',
                       color_discrete_map=region_colors, title="Température maximale vs consommation")
    st.plotly_chart(fig11, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre comment la température influence la consommation d'énergie dans chaque région.")

    # Visualisation 2 : Précipitations vs consommation
    fig12 = px.scatter(df_hf_cvl_full, x='PRECIP_TOTAL_DAY_MM', y='ENERGIE_SOUTIREE', color='REGION',
                       color_discrete_map=region_colors, title="Précipitations vs consommation")
    st.plotly_chart(fig12, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique illustre l'impact des précipitations sur la consommation d'énergie.")

    # Visualisation 3 : Humidité maximale vs consommation
    fig13 = px.scatter(df_hf_cvl_full, x='HUMIDITY_MAX_PERCENT', y='ENERGIE_SOUTIREE', color='REGION',
                       color_discrete_map=region_colors, title="Humidité maximale vs consommation")
    st.plotly_chart(fig13, use_container_width=True)
    st.markdown("**Utilité :** Il montre la corrélation entre le taux d'humidité et la consommation d'énergie dans chaque région.")

    # Visualisation 4 : Consommation pendant les fortes précipitations
    fig14 = px.bar(df_hf_cvl_full[df_hf_cvl_full['PRECIP_TOTAL_DAY_MM'] > 10], x='REGION', y='ENERGIE_SOUTIREE',
                   color='REGION', color_discrete_map=region_colors,
                   title="Consommation pendant les fortes précipitations")
    st.plotly_chart(fig14, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique compare la consommation d'énergie dans les jours de fortes pluies entre les régions.")

    # Visualisation 5 : Vitesse du vent vs consommation
    fig15 = px.scatter(df_hf_cvl_full, x='WINDSPEED_MAX_KMH', y='ENERGIE_SOUTIREE', color='REGION',
                       color_discrete_map=region_colors, title="Vitesse du vent vs consommation")
    st.plotly_chart(fig15, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre l'influence de la vitesse du vent sur la consommation d'énergie.")

    # Visualisation 6 : Couverture nuageuse vs consommation
    fig16 = px.scatter(df_hf_cvl_full, x='CLOUDCOVER_AVG_PERCENT', y='ENERGIE_SOUTIREE', color='REGION',
                       color_discrete_map=region_colors, title="Couverture nuageuse vs consommation")
    st.plotly_chart(fig16, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique analyse l'impact de la couverture nuageuse sur la consommation d'énergie.")

    # Visualisation 7 : Histogramme consommation et température
    # fig17 = px.histogram(df_hf_cvl_full, x='ENERGIE_SOUTIREE', nbins=50, color='MAX_TEMPERATURE_C',
    #                      color_discrete_map=region_colors, title="Histogramme consommation et température")
    # st.plotly_chart(fig17, use_container_width=True)
    # st.markdown("**Utilité :** Cet histogramme montre la répartition des consommations selon la température.")

    # Visualisation 8 : Histogramme consommation et précipitations
    # fig18 = px.histogram(df_hf_cvl_full, x='ENERGIE_SOUTIREE', nbins=50, color='PRECIP_TOTAL_DAY_MM',
    #                      color_discrete_map=region_colors, title="Histogramme consommation et précipitations")
    # st.plotly_chart(fig18, use_container_width=True)
    # st.markdown("**Utilité :** Cet histogramme illustre la distribution des consommations en fonction des précipitations.")

    # Visualisation 9 : Consommation par région pendant les vacances scolaires
    fig19 = px.bar(df_hf_cvl_full[df_hf_cvl_full['Vacances'] == 1], x='REGION', y='ENERGIE_SOUTIREE', color='REGION',
                   color_discrete_map=region_colors, title="Consommation pendant les vacances scolaires")
    st.plotly_chart(fig19, use_container_width=True)
    st.markdown("**Utilité :** Il permet d'analyser la consommation pendant les vacances scolaires dans les différentes régions.")

# Visualisation 12 : Heatmap de corrélation
    corr_matrix = df_hf_cvl_full[['ENERGIE_SOUTIREE', 'MAX_TEMPERATURE_C', 'PRECIP_TOTAL_DAY_MM', 'HUMIDITY_MAX_PERCENT']].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.markdown("**Utilité :** La heatmap montre les corrélations entre les variables climatiques et la consommation d'énergie, aidant à identifier des relations potentielles.")

# Visualisation 13 : Violin plot pour la distribution de la consommation
    fig22 = px.violin(df_hf_cvl_full, x='REGION', y='ENERGIE_SOUTIREE', color='REGION',
                  box=True, points="all", hover_data=df_hf_cvl_full.columns,
                  title="Distribution de la consommation d'énergie par région")
    st.plotly_chart(fig22, use_container_width=True)
    st.markdown("**Utilité :** Le violin plot permet de voir la distribution de la consommation dans chaque région ainsi que sa densité.")

# Visualisation 16 : Température vs consommation avec régression linéaire
    fig25 = px.scatter(df_hf_cvl_full, x='MAX_TEMPERATURE_C', y='ENERGIE_SOUTIREE', trendline='ols',
                   color='REGION', title="Température maximale vs consommation avec régression linéaire")
    st.plotly_chart(fig25, use_container_width=True)
    st.markdown("**Utilité :** La régression linéaire ajoute une ligne de tendance qui montre la relation entre la température et la consommation.")
# ---------------------------------------------------------------------------
# Section 3 : Prédiction basée sur les données historiques avec Random Forest
# ---------------------------------------------------------------------------
elif section == "Section 3 : Prédiction basée sur données historiques":
    st.header("Section 3 : Prédiction de la consommation énergétique avec Random Forest")

    # Explication pour l'utilisateur
    st.markdown("""
    ### Explication :
    Cette section vous permet de prédire la consommation énergétique à partir de données historiques (température, précipitations, etc.) en utilisant un modèle d'apprentissage automatique de type Random Forest.
    """)

    # Fonction pour entraîner le modèle
    def train_model(X, y):
        # Standardisation et imputation
        scaler_X = StandardScaler()
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler_X.fit_transform(X_imputed)

        # Séparation des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Entraînement du modèle
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        return model, scaler_X, X_train, X_test, y_train, y_test

    # Fonction pour faire des prédictions
    def make_prediction(model, scaler_X, input_data):
        input_data_scaled = scaler_X.transform(input_data)
        prediction = model.predict(input_data_scaled)
        return prediction

    # Ajouter une colonne binaire pour les précipitations
    df['Pluie'] = np.where(df['Avg_Précipitations_24h'] > 0, 1, 0)

    # Variables d'entrée et cible
    X = df[['NB_POINTS_SOUTIRAGE', 'Avg_Temperature', 'Pluie', 'month']]
    y = df['ENERGIE_SOUTIREE']

    # Entraînement du modèle
    model, scaler_X, X_train, X_test, y_train, y_test = train_model(X, y)

    # Interface utilisateur Streamlit
    st.title("Enedis: Prédiction de l'Énergie")

    # Sélection de la région
    regions = ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
               'Centre-Val de Loire', 'Grand-Est', 'Hauts-de-France', 'Normandie',
               'Nouvelle Aquitaine', 'Occitanie', 'Pays de la Loire',
               "Provence-Alpes-Côte d'Azur", 'Île-de-France']

    selected_region = st.selectbox('Sélectionnez une région', regions)

    # Sélection de l'année et du mois
    years = list(range(2024, 2026))  # Plage d'années disponibles
    selected_year = st.selectbox('Sélectionnez une année', years)
    months = list(range(1, 13))
    selected_month = st.selectbox('Sélectionnez un mois', months)

    # Entrées utilisateur pour la température et la longueur du jour
    feature_temperature = st.number_input('Entrez la température moyenne (°C)', value=0.0)
    feature_day_length = st.number_input("Entrez la longueur du jour (en heures)", value=12.0)

    # Détermination de la pluie sur la base de l'entrée utilisateur pour les précipitations
    feature_precipitations = st.number_input('Entrez les précipitations moyennes sur 24h (mm)', value=0.0)
    feature_pluie = 1 if feature_precipitations > 0 else 0  # 1 si pluie, sinon 0

    # Collecte des données d'entrée
    input_data = np.array([[1, feature_temperature, feature_pluie, selected_month]])  # 1 utilisé pour le nombre de points de soutirage par défaut

# Prédiction lorsqu'on clique sur le bouton
if st.button("Prédire"):
    prediction = make_prediction(model, scaler_X, input_data)
    
    # Affichage de la date future
    future_date = datetime(selected_year, selected_month, 1) 
    
    # Utilisation du formatage sans notation scientifique
    prediction_value = "{:,.2f}".format(prediction[0])
    
    st.write(f"La prédiction pour la région {selected_region} le {future_date.strftime('%d %B %Y')} est : {prediction_value} kWh")


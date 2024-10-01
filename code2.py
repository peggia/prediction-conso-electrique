import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime , timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from vacances_scolaires_france import SchoolHolidayDates

@st.cache_data
def get_df_from_csv(fn):
    return pd.read_csv(fn,encoding='utf-8')

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
    # Charger les fichiers CSV
    # même df utilisé pour les volets 1 et 3, chargé avec la fonciton qui le met en cache
    df = get_df_from_csv('dfmlenedis.csv')

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

    df['MOIS'] = pd.to_datetime(df['DATE']).dt.month
    df['SAISON'] = df['MOIS'].apply(nommer_saison)

    # Visualisation 1 : Consommation par région
    #tri par ordre décroissant
    df_conso_region= df.groupby(['REGION'])['ENERGIE_SOUTIREE'].sum().reset_index()
    df_sorted = df_conso_region.sort_values(by='ENERGIE_SOUTIREE', ascending=False)
    
    fig1 = px.bar(df_sorted, x='REGION', y='ENERGIE_SOUTIREE', color='REGION',
                  color_discrete_map=region_colors,
                  title="Consommation d'énergie par région")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre la répartition de la consommation d'énergie par région.")
    
    # Visualisation 2 : Nombre de points de soutirage par région
    #tri par ordre décroissant
    df_points_region= df.groupby(['REGION'])['NB_POINTS_SOUTIRAGE'].mean().reset_index()
    df_sorted_nb_points = df_points_region.sort_values(by='NB_POINTS_SOUTIRAGE', ascending=False)
    fig5 = px.bar(df_sorted_nb_points, x='REGION', y='NB_POINTS_SOUTIRAGE', color='REGION',
                  color_discrete_map=region_colors,
                  title="Nombre de points de soutirage par région")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**Utilité :** Il montre le nombre de points de soutirage par région, essentiel pour comprendre l'infrastructure et la taille de la région en nombre de foyers (abonnements actifs).")

     # Visualisation 3 : Consommation moyenne par région
    df['CONSO_MOYENNE'] = df['ENERGIE_SOUTIREE'] / df['NB_POINTS_SOUTIRAGE']
    df_conso_moyenne= df.groupby(['REGION'])['CONSO_MOYENNE'].mean().reset_index()
    #tri par ordre décroissant
    df_conso_moyenne_sorted = df_conso_moyenne.sort_values(by='CONSO_MOYENNE', ascending=False)
    fig6 = px.bar(df_conso_moyenne_sorted, x='REGION', y='CONSO_MOYENNE', color='REGION',
                  color_discrete_map=region_colors,
                  #color_discrete_sequence=px.colors.qualitative.Alphabet,
                  title="Consommation moyenne par rapport au nombre de points soutirage par région")
    # Réorganiser l'axe des x pour respecter l'ordre trié
    fig6.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique permet de comparer la consommation moyenne d'electricité par foyer dans chaque région, et revèle les régions les plus énergivores.")
    
  
    # # Visualisation 2 : Consommation par mois
    # fig2 = px.bar(df, x='MOIS', y='ENERGIE_SOUTIREE', color='REGION',
    #               color_discrete_map=region_colors,
    #               title="Consommation par mois")
    # st.plotly_chart(fig2, use_container_width=True)
    # st.markdown("**Utilité :** Ce graphique illustre la consommation mensuelle d'énergie, permettant d'identifier les périodes de pic.")

    # Visualisation 3 : Consommation par saison (Box plot)

    #garder uniquement les années differentes de 2024 car elle est incomplète et fausse les résultats
    df_conso_full_year = df[~(df['DATE'].str.contains('2024'))]
    fig3 = px.box(df_conso_full_year, x='SAISON', y='ENERGIE_SOUTIREE', color='REGION',
                  color_discrete_map=region_colors,
                  title="Consommation par saison")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre la variation de la consommation d'énergie selon les saisons.")

    # Visualisation 4 : Répartition de la consommation par région (pie chart)
    fig4 = px.pie(df, names='REGION', values='ENERGIE_SOUTIREE',
                  color_discrete_map=region_colors,
                  title="Répartition de la consommation par région")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique en secteur montre la part de chaque région dans la consommation totale.")

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
        'Bretagne': [48.1173, -1.6778],  # Ici la longitude est corrigée
        'Auvergne-Rhône-Alpes': [45.764, 4.8357],
        'Bourgogne-Franche-Comté': [47.2805, 5.9993],
        'Pays de la Loire': [47.2184, -1.5536],
        'Provence-Alpes-Côte d\'Azur': [43.9352, 6.0679],
        'Occitanie': [43.6045, 1.4442],
        'Corse': [42.0396, 9.0129]
    }

      # Ajouter les coordonnées au DataFrame
    df['LAT'] = df['REGION'].map(lambda x: geo_data[x][0] if x in geo_data else None)
    df['LON'] = df['REGION'].map(lambda x: geo_data[x][1] if x in geo_data else None)

    # Filtrer les lignes avec des valeurs manquantes dans les coordonnées
    df = df.dropna(subset=['LAT', 'LON'])

    # Créer la carte interactive
    fig_map = px.scatter_mapbox(df, lat='LAT', lon='LON', size='ENERGIE_SOUTIREE',
                                color='REGION', color_discrete_map=region_colors,
                                hover_name='REGION', hover_data=['ENERGIE_SOUTIREE', 'NB_POINTS_SOUTIRAGE'],
                                title="Carte interactive de la consommation d'énergie par région", 
                                mapbox_style="open-street-map", zoom=5)
    fig_map.update_layout(mapbox_zoom=5, mapbox_center={"lat": 46.603354, "lon": 1.888334})  # Centrer sur la France
    st.plotly_chart(fig_map, use_container_width=True)

   

    # Visualisation 7 : Histogramme de la consommation d'énergie
    # fig7 = px.histogram(df, x='ENERGIE_SOUTIREE', nbins=50, color='REGION',
    #                     color_discrete_sequence=px.colors.qualitative.Pastel,
    #                     title="Histogramme des consommations d'énergie")
    # st.plotly_chart(fig7, use_container_width=True)
    # st.markdown("**Utilité :** Cet histogramme montre la distribution des consommations d'énergie.")

    # Visualisation 8 : Corrélation entre la consommation et le nombre de points de soutirage
    # fig8 = px.scatter(df, x='NB_POINTS_SOUTIRAGE', y='ENERGIE_SOUTIREE', color='REGION',
    #                   color_discrete_sequence=px.colors.qualitative.Plotly,
    #                   title="Corrélation entre la consommation et les points de soutirage")
    # st.plotly_chart(fig8, use_container_width=True)
    # st.markdown("**Utilité :** Cette corrélation est utile pour comprendre la relation entre infrastructure et consommation.")

    fig21 = px.line(df, x='DATE', y='ENERGIE_SOUTIREE', color='REGION',
                #color_discrete_sequence=px.colors.qualitative.T10,
                color_discrete_map=region_colors,
                title="Séries temporelles de la consommation d'énergie")
    st.plotly_chart(fig21, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre l'évolution de la consommation d'énergie dans le temps, permettant de visualiser les tendances et les pics de consommation.")
    


    fig23 = px.treemap(df, path=['SAISON', 'REGION'], values='ENERGIE_SOUTIREE',
                   color='ENERGIE_SOUTIREE', hover_data=['ENERGIE_SOUTIREE'],
                   title="Treemap de la consommation d'énergie par région et saison",
                   color_continuous_scale='Viridis')
    st.plotly_chart(fig23, use_container_width=True)
    st.markdown("**Utilité :** Le treemap montre la répartition de la consommation par région et par saison de manière hiérarchique.")
    
   #Top 3 des régions qui consomment le plus d'énergie en moyenne par nombre de points de soutirage
    # Obtenir le top 3 des régions pour chaque saison
    df_conso_moyenne_saison= df.groupby(['REGION','SAISON'])['CONSO_MOYENNE'].mean().reset_index()
    df_top_3_regions = df_conso_moyenne_saison.groupby('SAISON')[['SAISON','CONSO_MOYENNE','REGION']].apply(lambda x: x.nlargest(5, 'CONSO_MOYENNE')).reset_index(drop=True)
    #ordonner par ordre décroissant
    df_conso_top_3_regions = df_top_3_regions.sort_values(by='CONSO_MOYENNE', ascending=False)

    fig24 = px.treemap(df_conso_top_3_regions, path=['SAISON', 'REGION'], values='CONSO_MOYENNE',
                   color='CONSO_MOYENNE', hover_data=['CONSO_MOYENNE'],
                   title="Treemap de la consommation moyenne par région et saison",
                   color_continuous_scale='Viridis')
    st.plotly_chart(fig24, use_container_width=True)
    st.markdown("**Utilité :** Le treemap montre la répartition de la consommation moyenne par région et par saison de manière hiérarchique.")
    
   
    # df_cascade = df.groupby('MOIS')['ENERGIE_SOUTIREE'].sum().reset_index()
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
    #(on n'utilise plus le suivant car contient seulement 2 régions)
    #df_hf_cvl_full = get_df_from_csv('df_hf_cvl_full.csv')
    # Charger les fichiers CSV
    # même df utilisé pour les volets 1 et 3, chargé avec la fonciton qui le met en cache
    df_all_regions = get_df_from_csv('dfmlenedis.csv')

    # Visualisation 1 : Température maximale vs consommation
    fig11 = px.scatter(df_all_regions, x='Avg_Temperature', y='ENERGIE_SOUTIREE', color='REGION',
                       color_discrete_map=region_colors, title="Consommation vs Température moyenne ")
    st.plotly_chart(fig11, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique montre comment la température influence la consommation d'énergie dans chaque région.")

    # Visualisation 2 : Précipitations vs consommation
    fig12 = px.scatter(df_all_regions, x='Avg_Précipitations_24h', y='ENERGIE_SOUTIREE', color='REGION',
                       color_discrete_map=region_colors, title="Consommation vs Précipitations")
    st.plotly_chart(fig12, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique illustre l'impact des précipitations sur la consommation d'énergie.")

    # # Visualisation 3 : Humidité maximale vs consommation
    # fig13 = px.scatter(df_all_regions, x='HUMIDITY_MAX_PERCENT', y='ENERGIE_SOUTIREE', color='REGION',
    #                    color_discrete_map=region_colors, title="Humidité maximale vs consommation")
    # st.plotly_chart(fig13, use_container_width=True)
    # st.markdown("**Utilité :** Il montre la corrélation entre le taux d'humidité et la consommation d'énergie dans chaque région.")

    # Visualisation 4 : Consommation pendant les fortes précipitations
    fig14 = px.bar(df_all_regions[df_all_regions['Avg_Précipitations_24h'] > 10], x='REGION', y='ENERGIE_SOUTIREE',
                   color='REGION', color_discrete_map=region_colors,
                   title="Consommation pendant les fortes précipitations")
    st.plotly_chart(fig14, use_container_width=True)
    st.markdown("**Utilité :** Ce graphique compare la consommation d'énergie dans les jours de fortes pluies entre les régions.")

    # # Visualisation 5 : Vitesse du vent vs consommation
    # fig15 = px.scatter(df_all_regions, x='WINDSPEED_MAX_KMH', y='ENERGIE_SOUTIREE', color='REGION',
    #                    color_discrete_map=region_colors, title="Vitesse du vent vs consommation")
    # st.plotly_chart(fig15, use_container_width=True)
    # st.markdown("**Utilité :** Ce graphique montre l'influence de la vitesse du vent sur la consommation d'énergie.")

    # # Visualisation 6 : Couverture nuageuse vs consommation
    # fig16 = px.scatter(df_all_regions, x='CLOUDCOVER_AVG_PERCENT', y='ENERGIE_SOUTIREE', color='REGION',
    #                    color_discrete_map=region_colors, title="Couverture nuageuse vs consommation")
    # st.plotly_chart(fig16, use_container_width=True)
    # st.markdown("**Utilité :** Ce graphique analyse l'impact de la couverture nuageuse sur la consommation d'énergie.")

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
    fig19 = px.bar(df_all_regions[df_all_regions['Vacances'] == 1], x='REGION', y='ENERGIE_SOUTIREE', color='REGION',
                   color_discrete_map=region_colors, title="Consommation pendant les vacances scolaires")
    st.plotly_chart(fig19, use_container_width=True)
    st.markdown("**Utilité :** Il permet d'analyser la consommation pendant les vacances scolaires dans les différentes régions.")

   # Visualisation 9.a : Consommation par région pendant les vacances scolaires
    # Créer une nouvelle colonne pour indiquer si c'est pendant ou en dehors des vacances
    df_all_regions['Vacances_Status'] = df_all_regions['Vacances'].map({1: 'En Vacances', 0: 'Hors Vacances'})

    # Regrouper par REGION et Vacances_Status et sommer la consommation
    df_comparaison = df_all_regions.groupby(['REGION', 'Vacances_Status'])['ENERGIE_SOUTIREE'].sum().reset_index()

    # Créer le graphique à barres
    fig20 = px.bar(df_comparaison, x='REGION', y='ENERGIE_SOUTIREE', color='Vacances_Status',
                    color_discrete_map={'En Vacances': '#88b949', 'Hors Vacances': 'orange'},
                    barmode='group',
                    title="Comparaison de la consommation d'énergie par région pendant et en dehors des vacances scolaires")

    # Afficher le graphique
    st.plotly_chart(fig20, use_container_width=True)
    st.markdown("**Utilité :** Il permet d'analyser la consommation d'énergie pendant et en dehors des vacances scolaires dans les différentes régions.")

# Visualisation 12 : Heatmap de corrélation
    corr_matrix = df_all_regions[['ENERGIE_SOUTIREE', 'Avg_Temperature', 'Avg_Précipitations_24h']].corr()
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.markdown("**Utilité :** La heatmap montre les corrélations entre les variables climatiques et la consommation d'énergie, aidant à identifier des relations potentielles.")

# Visualisation 13 : Violin plot pour la distribution de la consommation
   

    # Créer le violin plot
    fig22 = px.violin(df_all_regions, x='REGION', y='ENERGIE_SOUTIREE', color='REGION',
                    box=True, points="all", hover_data=df_all_regions.columns,
                    title="Distribution de la consommation d'énergie par région")

    # Afficher le graphique
    st.plotly_chart(fig22, use_container_width=True)
    st.markdown("**Utilité :** Le violin plot permet de voir la distribution de la consommation dans chaque région ainsi que sa densité.")

# Visualisation 16 : Température vs consommation avec régression linéaire
    fig25 = px.scatter(df_all_regions, x='Avg_Temperature', y='ENERGIE_SOUTIREE', trendline='ols',
                   color='REGION', title="Consommation vs Température moyenne avec régression linéaire")
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
    Vous pouvez également faire des prédictions pour des dates futures ou pour des conditions hypothétiques.
    """)

    # Importations nécessaires
    import datetime  # Importer le module datetime
    from datetime import timedelta

    # Charger le fichier CSV
    df = get_df_from_csv('dfmlenedis.csv')

    ####### Fonctions

    # Fonction pour déterminer si une date est pendant les vacances scolaires
    def vacances(date, region):
        # Définir les zones et les régions correspondantes
        zone_A = ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Nouvelle-Aquitaine']
        zone_B = ['Bretagne', 'Centre-Val de Loire', 'Grand-Est', 'Hauts-de-France',
                  'Normandie', 'Pays de la Loire', "Provence-Alpes-Côte d'Azur"]
        zone_C = ['Occitanie', 'Île-de-France']

        # Récupérer la zone de la région
        if region in zone_A:
            zone = 'A'
        elif region in zone_B:
            zone = 'B'
        elif region in zone_C:
            zone = 'C'
        else:
            raise ValueError(f"La région '{region}' n'appartient à aucune zone connue.")

        # Récupérer les vacances pour la zone et la date
        d = SchoolHolidayDates()
        is_vacances = d.is_holiday_for_zone(date.date(), zone)
        return int(is_vacances)

    # Fonction pour récupérer la température et les précipitations si la date est dans l'intervalle
    def get_weather_data(df, date, region):
        mask = (df['REGION'] == region) & (df['year'] == date.year) & (df['month'] == date.month) & (df['day'] == date.day)
        weather_data = df.loc[mask, ['Avg_Temperature', 'Avg_Précipitations_24h']].squeeze()
        return weather_data

    # Fonction de prédiction mise à jour avec remplissage automatique des données météo si la date est dans l'intervalle
    def predict(model, X_scaler, date, region, temperature=None, precipitation=None):
        if isinstance(date, datetime.date):
            date = datetime.datetime.combine(date, datetime.datetime.min.time())

        if (date < datetime.datetime(2022, 1, 1)) or (date > datetime.datetime(2024, 6, 30)):
            # Prédiction pour les dates hors de la plage définie
            Vacances = vacances(date, region)
            mask = (df['month'] == date.month) & (df['day'] == date.day)
            DayLength_hours = df.loc[mask, 'DayLength_hours'].median()
            mask_region = (df['REGION'] == region)
            nb_points_soutirage = df.loc[mask_region, 'NB_POINTS_SOUTIRAGE'].median()

            X_input = pd.DataFrame([[nb_points_soutirage, temperature, precipitation, DayLength_hours, Vacances, date.day, date.month]],
                                   columns=['NB_POINTS_SOUTIRAGE', 'Avg_Temperature', 'Avg_Précipitations_24h',
                                            'DayLength_hours', 'Vacances', 'day', 'month'])
        else:
            # Prédiction avec les données du CSV
            weather_data = get_weather_data(df, date, region)
            if weather_data.empty:
                st.error("Aucune donnée météo disponible pour la date et la région sélectionnées.")
                return None

            temperature = weather_data['Avg_Temperature']
            precipitation = weather_data['Avg_Précipitations_24h']

            Vacances = vacances(date, region)
            DayLength_hours = df.loc[(df['month'] == date.month) & (df['day'] == date.day), 'DayLength_hours'].median()
            nb_points_soutirage = df.loc[df['REGION'] == region, 'NB_POINTS_SOUTIRAGE'].median()

            X_input = pd.DataFrame([[nb_points_soutirage, temperature, precipitation, DayLength_hours, Vacances, date.day, date.month]],
                                   columns=['NB_POINTS_SOUTIRAGE', 'Avg_Temperature', 'Avg_Précipitations_24h',
                                            'DayLength_hours', 'Vacances', 'day', 'month'])

        X_scaled = X_scaler.transform(X_input)
        prediction = model.predict(X_scaled)

        return prediction

    # Ajouter une visualisation en courbe pour le mois entier
    def visualize_month(df, model, X_scaler, month, year, region):
        from calendar import monthrange
        days_in_month = monthrange(year, month)[1]
        dates = [datetime.datetime(year, month, day) for day in range(1, days_in_month + 1)]
        predictions = []
        for date in dates:
            prediction = predict(model, X_scaler, date, region)
            if prediction is not None:
                predictions.append(prediction[0])
            else:
                predictions.append(None)

        fig = px.line(x=dates, y=predictions, title=f"Prédiction de la consommation pour {month}/{year}", labels={'x': 'Date', 'y': 'Consommation (Wh)'})
        st.plotly_chart(fig, use_container_width=True)

    # # Ajouter une visualisation journalière par heure
    # def visualize_day_hours(df, model, X_scaler, date, region):
    #     hours = [f'{i}:00' for i in range(24)]
    #     predictions = []
    #     base_prediction = predict(model, X_scaler, date, region)
    #     if base_prediction is None:
    #         st.error("Impossible de générer la prédiction horaire.")
    #         return

    #     total_consumption = base_prediction[0]
    #     # Distribution simplifiée de la consommation sur 24 heures
    #     hourly_consumption = total_consumption / 24
    #     predictions = [hourly_consumption for _ in range(24)]

    #     fig = px.line(x=hours, y=predictions, title=f"Prédiction horaire pour le {date.strftime('%d/%m/%Y')}", labels={'x': 'Heure', 'y': 'Consommation (Wh)'})
    #     st.plotly_chart(fig, use_container_width=True)

    # Visualisation des prédictions pour l'année entière
    def visualize_year(df, model, X_scaler, year, region):
        months = range(1, 13)
        predictions = []
        for month in months:
            date = datetime.datetime(year, month, 15)
            prediction = predict(model, X_scaler, date, region)
            if prediction is not None:
                predictions.append(prediction[0])
            else:
                predictions.append(None)

        fig = px.bar(x=list(months), y=predictions, title=f"Prédiction de la consommation pour l'année {year}", labels={'x': 'Mois', 'y': 'Consommation (Wh)'})
        st.plotly_chart(fig, use_container_width=True)

    # Fonction pour entraîner le modèle Random Forest avec les données historiques
    def init_model(df):
        # Colonnes à utiliser pour les prédictions
        features = ['NB_POINTS_SOUTIRAGE', 'Avg_Temperature', 'Avg_Précipitations_24h',
                    'DayLength_hours', 'Vacances', 'day', 'month']
        X = df[features]
        y = df['ENERGIE_SOUTIREE']

        # Gérer les valeurs manquantes
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Normaliser les données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Diviser les données en train et test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Créer le modèle Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Évaluer le modèle
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        st.write(f"Score du modèle sur les données d'entraînement: {train_score:.2f}")
        st.write(f"Score du modèle sur les données de test: {test_score:.2f}")

        return model, scaler

    # Initialisation du modèle et du scaler
    model, scaler = init_model(df)

    # Interface utilisateur pour la sélection de la région, de la date et des entrées météorologiques
    regions = ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
               'Centre-Val de Loire', 'Grand-Est', 'Hauts-de-France', 'Normandie',
               'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
               "Provence-Alpes-Côte d'Azur", 'Île-de-France']
    selected_region = st.selectbox('Sélectionnez une région', regions)

    selected_date = st.date_input("Sélectionnez une date", min_value=datetime.date(2019, 1, 1), max_value=datetime.date(2039, 12, 31))

    # Initialisation des variables
    feature_temperature = None
    feature_precipitations = None

    # Vérifier si la date est dans la plage et remplir automatiquement les données météo
    if isinstance(selected_date, datetime.date):
        future_date = datetime.datetime.combine(selected_date, datetime.datetime.min.time())
    else:
        future_date = selected_date

    if datetime.datetime(2022, 1, 1) <= future_date <= datetime.datetime(2024, 6, 30):
        weather_data = get_weather_data(df, future_date, selected_region)
        if not weather_data.empty:
            feature_temperature = weather_data['Avg_Temperature']
            feature_precipitations = weather_data['Avg_Précipitations_24h']
            st.write(f"Température et précipitations automatiques pour cette date : {feature_temperature}°C, {feature_precipitations} mm")
        else:
            st.warning("Données météo indisponibles pour cette date. Veuillez entrer manuellement les valeurs.")
            feature_temperature = st.number_input('Température moyenne (°C)', value=10.0)
            feature_precipitations = st.number_input('Précipitations (mm)', value=0.0)
    else:
        st.info("Veuillez entrer les données météo pour la prédiction.")
        feature_temperature = st.number_input('Température moyenne (°C)', value=10.0)
        feature_precipitations = st.number_input('Précipitations (mm)', value=0.0)

    # Bouton de prédiction
    if st.button("Prédire"):
        prediction = predict(model, scaler, future_date, selected_region, feature_temperature, feature_precipitations)

        if prediction is not None:
            prediction_Mwh = prediction[0] / 1_000_000
            st.success(f"Prédiction de consommation : {int(prediction_Mwh)} MWh pour la date {future_date.strftime('%d/%m/%Y')}.")

            # Visualisation pour le mois
            visualize_month(df, model, scaler, future_date.month, future_date.year, selected_region)

            # Visualisation journalière
            #visualize_day_hours(df, model, scaler, future_date, selected_region)

            # Visualisation pour l'année
            visualize_year(df, model, scaler, future_date.year, selected_region)
        else:
            st.error("La prédiction n'a pas pu être effectuée.")

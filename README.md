# Contexte
La première version importée dans ce repository est le résultat du travail d'une équipe de 6 personnes, moi y compris, qui avons contribué aux différentes activités du projet, depuis la collecte, le traitement et l'analyse des données, à l'évaluation des modèles de Machine Learning,  l'élaboration de l'interface utilisateur et la mise au point de l'application ainsi que la présentation finale. J'ai essayé de mettre ma patte un peu partout tout au long du projet, histoire d'avoir ma propre vision globale.

Le projet a été effectué dans le cadre de notre préparation à la certification Data Analyst RNCP37429, et il a été présenté le 4/10/24.


# Objectifs

* Analyse des données et choix des principaux indicateurs de consommation éléctrique dans les différentes régions de France métropolitaine.
* Choix d'un modèle  de régression (Machine Learning) pour prédire la consommation en fonction de paramètres météorologiques et du calendrier 
* Interface Streamlit donnant accès à la prédiction de consommation et aux principaux des indicateurs de consommation électrique.

# Contenu du repository
* Fichier csv avec les données mergées et agregées (dfmlenedeis.csv)
* Fichier python avec le code de l'application streamlit (code2.py)
* sous-répertoire "Analyse" avec le notebook utilisé pendant l'analyse exploratoire des données
* sous-répertoire "ML" contenant le notebook utilisé pour l'evaluation des modèles, et les tests unitaires de prediction
  
# Sources des données
Données conso électrique:
https://data.enedis.fr/explore/dataset/consommation-agregee-demi-horaire-des-points-de-soutirage-inferieurs-a-36kva-par%40agenceore/

Données Météo: 
https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/table/

Calendrier: 
https://pypi.org/project/vacances-scolaires-france/

# Prédiction de consommation électrique
Modèle : Random Forest Regressor (Forêts Aléatoires)

## Contexte
La première version importée dans ce repository est le résultat du travail d'une équipe de 6 personnes, moi y compris, qui avons contribué aux différentes activités du projet, depuis la collecte, le traitement et l'analyse des données, à l'évaluation des modèles de Machine Learning,  l'élaboration de l'interface utilisateur et la mise au point de l'application ainsi que la présentation finale. J'ai essayé de mettre ma patte un peu partout tout au long du projet, histoire d'avoir ma propre vision globale.

Le projet initial a été effectué dans le cadre de notre préparation à la certification Data Analyst RNCP37429, et il a été présenté le 4/10/24.

Depuis, je fais des modifs à ma sauce 🥰

## Objectifs

* Analyse des données et choix des principaux indicateurs de consommation éléctrique dans les différentes régions de France métropolitaine.
* Choix d'un modèle  de régression (Machine Learning) pour prédire la consommation en fonction de paramètres météorologiques et du calendrier 
* Interface Streamlit donnant accès à la prédiction de consommation et aux principaux des indicateurs de consommation électrique.

## Contenu du repository
* Fichier csv avec les données mergées et agregées (dfmlenedeis.csv)
* Fichier python avec le code de l'application streamlit (code2.py)
* sous-répertoire "analyse" avec le notebook utilisé pendant l'analyse exploratoire des données
* sous-répertoire "machine-learning" contenant le notebook utilisé pour l'evaluation des modèles, et les tests unitaires de prediction
* sous-répertoire "ressources" contenant les ressources graphiques (icones) de l'application
## Scénario d'utilisation (Use Case)

![ENEDIS-UseCase](https://github.com/user-attachments/assets/05ac820e-9237-4ab2-8905-672b6545ec24)


## Application Streamlit
Voici le lien vers l'application: https://prediction-conso-electrique-enedis.streamlit.app/

## Analyse de la consommation en France
![image](https://github.com/user-attachments/assets/4b2d3f20-40db-4cdc-9e0a-f32b6541a93b)

## A propos du modèle Forêts Aléatoires (RandomForestRegressor)
> "Les forêts aléatoires consistent à entraîner de multiples arbres de décision en parallèle et à moyenner leurs prédictions. Contraindre la profondeur des arbres correspond à une régularisation qui compense le sur-apprentissage"

## Sources des données
* Données conso électrique:

https://data.enedis.fr/explore/dataset/consommation-agregee-demi-horaire-des-points-de-soutirage-inferieurs-a-36kva-par%40agenceore/

* Données Météo:
  
https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/table/

* Calendrier:
  
https://pypi.org/project/vacances-scolaires-france/

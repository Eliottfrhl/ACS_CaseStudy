# ACS_CaseStudy

## Description

Cette étude de cas porte sur la recherche du maximum d'un champ de potentiel inconnu par une flotte de robots mobiles. Chaque robot ne mesure que le potentiel local et la flotte doit :

- localiser le maximum global ;
- identifier des maxima locaux ;
- caractériser la forme du nuage de polluant.

Deux approches sont actuellement implémentées :

- `MED` : machine d'etats decentralisee, fichier [`src/control_algo_potential_ldc.py`](src/control_algo_potential_ldc.py)
- `BPS` : balayage partitionne supervise, fichier [`src/control_algo_potential.py`](src/control_algo_potential.py)

## Notice d'utilisation

### 1. Lancer une simulation

Configurer d'abord les parametres dans [`src/run_simulation.py`](src/run_simulation.py) :

- `CONTROL_ALGO_MODULE = 'control_algo_potential_ldc'` pour la `MED`
- `CONTROL_ALGO_MODULE = 'control_algo_potential'` pour la `BPS`
- `nbOfRobots = 3` ou `5`
- `DIFFICULTY = 1`, `2` ou `3`
- `RANDOM = False` pour un cas deterministe, `True` pour un cas aleatoire

Puis lancer :

```bash
python src/run_simulation.py
```

### 2. Lancer le benchmark de recherche du maximum

Ce benchmark evalue la Mission 1 avec des metriques spatiales (succes, distance au premier succes, distance totale, distance finale a la source, securite, partage des taches).

Exemples :

```bash
python src/run_max_search_benchmark.py --control-module control_algo_potential
python src/run_max_search_benchmark.py --control-module control_algo_potential_ldc
```

Les resultats sont exportes dans `outputs/max_search_benchmark/`.

### 3. Cas recommande pour la mise au point

- utiliser `RANDOM = False` pour reproduire un comportement ;
- commencer avec `N = 3` ;
- activer `SHOW_ANIMATION = True` et `SHOW_TRAJECTORY = True` dans [`src/run_simulation.py`](src/run_simulation.py).

## Metriques principales

- succes strict : un robot passe a moins de `1 m` du maximum global ;
- `d_min^source` : distance minimale entre la flotte et la source ;
- `D_hit` : distance parcourue jusqu'au premier succes ;
- `D_tot` : distance totale parcourue ;
- `d_min^robots` : distance minimale inter-robots ;
- coefficient de Gini : repartition de l'effort entre robots.

## Fichiers utiles

- [`src/run_simulation.py`](src/run_simulation.py) : simulation interactive
- [`src/run_max_search_benchmark.py`](src/run_max_search_benchmark.py) : benchmark Mission 1
- [`src/eval_metrics.py`](src/eval_metrics.py) : metriques de base
- [`src/eval_metrics_max_search.py`](src/eval_metrics_max_search.py) : metriques detaillees de recherche du maximum
- [`src/run_simulation_iso_benchmark.py`](src/run_simulation_iso_benchmark.py) : benchmark lie a la mission de contour

## Remarque

Pour la `MED`, le rassemblement final n'est declenche qu'apres la detection de trois maxima. Une distance finale elevee au maximum global ne signifie donc pas necessairement que la source principale n'a pas ete localisee.

# ACS_CaseStudy

## Description de l'étude de cas

Cette étude de cas porte sur la **recherche de maximum d'un champ de potentiel inconnu** par une flotte de robots mobiles. Les robots doivent collaborer pour localiser la position de maximum d'un champ scalaire dont ils ne peuvent mesurer que la valeur locale à leur position, puis évaluer la forme de la "tâche" de polluant.

## Métrique d'évaluation :
    - Trouver le maximum global (30%)
    - Rapidité (20%)
    - Trouver les maxima connexes (secondaires) (15%) -> 
    - Rassemblement autour du maximum global (10%)
    - Distance totale parcourue (10%)
    - Evitement des collisions (10%)
    - Partage du travail (5%)
    - Variance des résultats

## Comment lancer la simulation

```bash
python src/run_simulation.py
```

## Rôles des fichiers src

### Fichiers principaux

- **[run_simulation.py](src/run_simulation.py)** : Script principal de simulation restructuré et amélioré. Configure tous les paramètres de simulation (nombre de robots, durée, difficulté, mode d'initialisation), exécute la boucle de contrôle, calcule les métriques d'évaluation et génère les visualisations. Structure organisée en 5 sections :
  1. Paramètres de simulation (à configurer)
  2. Setup de la flotte et simulation
  3. Boucle de simulation
  4. Calcul des métriques d'évaluation
  5. Affichage des résultats

- **[control_algo_potential.py](src/control_algo_potential.py)** : **Fichier où implémenter votre algorithme de contrôle**. Contient la fonction `potential_seeking_ctrl(t, robotNo, robots_poses, difficulty, random)` qui calcule les commandes de vitesse `[vx, vy]` pour chaque robot. Signature mise à jour pour recevoir les paramètres de difficulté et aléatoire du champ de potentiel.

- **[eval_metrics.py](src/eval_metrics.py)** : Module d'évaluation automatique des performances de l'algorithme. Calcule deux métriques clés :
  - `relative_pot_found_error` : erreur relative entre le potentiel maximum trouvé et le maximum réel
  - `total_distance` : distance totale parcourue par l'ensemble de la flotte

### Bibliothèque (dossier lib/)

- **[robot.py](src/lib/robot.py)** : Classe `Robot` définissant la dynamique des robots (intégrateur simple 2D ou unicycle) et la classe `Fleet` pour gérer la flotte. Contient aussi la fonction `si_to_uni()` pour convertir les commandes cartésiennes en commandes unicycle.

- **[potential.py](src/lib/potential.py)** : Classe `Potential` générant le champ de potentiel à rechercher. Supporte 3 niveaux de difficulté et peut générer des champs aléatoires. Fournit la méthode `value()` pour mesurer le potentiel en un point. Domaine spatial : [-25, 25] m en x et y.

- **[potential_expe.py](src/lib/potential_expe.py)** : Variante expérimentale de la classe `Potential` avec un domaine spatial plus restreint ([-3, 3] m en x, [-5, 5] m en y) et des paramètres de gaussiennes différents (échelle réduite, poids modifiés). Utile pour des tests rapides ou des scénarios à plus petite échelle.

- **[simulation.py](src/lib/simulation.py)** : Classes `RobotSimulation` et `FleetSimulation` gérant l'intégration numérique des trajectoires et les visualisations (trajectoires 2D, évolution temporelle, champ de potentiel, animations). Mise à jour avec la fonction `generate_init_positions()` qui offre 3 modes d'initialisation des positions :
  - `'grid'` : grille régulière autour d'un centre avec espacement configurable
  - `'random'` : positions aléatoires uniformes
  - `'manual'` : positions explicites fournies par l'utilisateur

## Stratégies

Le fichier STRATEGIE.md a pour rôle de recueillir les différentes stratégies proposées par l'équipe pour cette étude de cas.

## Resources
- cenmpc.m : Code MPC présenté en cours d'Architecture, issu d'une université italienne à citer si utilisé.
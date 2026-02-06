# ACS_CaseStudy

## Description de l'étude de cas

Cette étude de cas porte sur la **recherche de maximum d'un champ de potentiel inconnu** par une flotte de robots mobiles. Les robots doivent collaborer pour localiser la position de maximum d'un champ scalaire dont ils ne peuvent mesurer que la valeur locale à leur position, puis évaluer la forme de la "tâche" de polluant.

## Rôles des fichiers src

### Fichiers principaux
- **[etude_de_cas.py](src/etude_de_cas.py)** : Script principal de simulation. Configure la flotte de robots (nombre, dynamique, positions initiales), initialise la simulation et exécute la boucle de contrôle. Gère l'affichage des mesures de potentiel et les visualisations.

- **[control_algo_potential.py](src/control_algo_potential.py)** : Fichier où implémenter l'algorithme de contrôle. Contient la fonction `potential_seeking_ctrl()` qui calcule les commandes de vitesse pour chaque robot en fonction des mesures de potentiel et des positions de la flotte. C'est le cœur de l'algorithme à développer.

### Bibliothèque (dossier lib/)
- **[robot.py](src/lib/robot.py)** : Classe `Robot` définissant la dynamique des robots (intégrateur simple 2D ou unicycle) et la classe `Fleet` pour gérer la flotte. Contient aussi la fonction `si_to_uni()` pour convertir les commandes cartésiennes en commandes unicycle.

- **[potential.py](src/lib/potential.py)** : Classe `Potential` générant le champ de potentiel à rechercher. Supporte 3 niveaux de difficulté et peut générer des champs aléatoires. Fournit la méthode `value()` pour mesurer le potentiel en un point.

- **[simulation.py](src/lib/simulation.py)** : Classes `RobotSimulation` et `FleetSimulation` gérant l'intégration numérique des trajectoires et les visualisations (trajectoires 2D, évolution temporelle, champ de potentiel).

## Resources
- cenmpc.m : Code MPC présenté en cours d'Architecture, issu d'une université italienne à citer si utilisé.
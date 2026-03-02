# Idée 1
- Mettre les robots en formation (laquelle ?) et évaluer des deltas de potentiel entre eux. Cela permet de réaliser une descente de gradient jusqu'à un minimum local. Ensuite s'éloigner (plus besoin de formation) pour obtenir le profil de la "tâche".

## Consignes
- Pas besoin du profil des tâches (en discuter dans rapport)
- Evaluer sur 5 robots

# Brainstorming
- Cartographier d'abord individuellement puis trouver minima
- Trouver un maxima avec formation, puis faire descendre un (ou plusieurs drones) vers une couche limite de la tâche. Ensuite virage (90°?) dans un sens (potentiellement 2 drones et chacun de son côté) et avec un Proportionnel, on rectifie le vecteur vitesse selon si le potentiel mesuré diminue ou augmente
- On peut faire 3 pour trouver les maxima et 2 pour cartographier les contours.
    - Les 2 pour cartographier font un trajet un peu random et ensuite quand ils entrent en contact avec un potentiel non nul, ils font une petite spirale pour trouver au moins 2 points de passage de potentiel = 0 à non nul pour calculer une tangente à la tâche, se déplacer un peu dans la tâche, suivre cette tangente et faire le tour avec le contrôleur proprtionnel
    - Les 3 autres se mettent en formation et font une "montée" de gradient vers un minimum local, puis s'écartent (spirale, ligne droite) pour voir s'il y a d'autres tâches (avec remontée de gradient ?)
        - attention, en s'écartant en ligne droite on peut louper d'autres tâches, et en spirale on peut penser que c'est une nouvelle tâche alors que c'est juste une forme non circulaire
- On lâche les robots sur le terrain, pour cartographier et obtenir des données en premier lieu et on fait ensuite des formations pour les maxima mais avec plus d'infos

    
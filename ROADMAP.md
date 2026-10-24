# Roadmap de réalisation — NeuroVision AI

Cette roadmap est pensée pour une réalisation autonome, avec des livrables vérifiables à chaque étape. Les durées sont indicatives : ajuste-les selon l’état actuel du projet.

## Vue d’ensemble

| Phase                      | Objectif                                             | Livrable                                  | Critère de validation                                          |
| -------------------------- | ---------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| 1. Cadrage                 | Définir le problème, les classes et les limites      | Description du projet + dataset référencé | Classes, provenance et limites documentées                     |
| 2. Exploration des données | Vérifier les fichiers et la distribution des classes | Notebook EDA                              | Extensions, images illisibles, dimensions et équilibre étudiés |
| 3. Prétraitement           | Préparer les images sans fuite de données            | Pipeline documenté                        | Taille, canaux, normalisation et labels cohérents              |
| 4. Modèle de base          | Construire et entraîner un CNN                       | Notebook/script d’entraînement            | Entraînement reproductible et historique conservé              |
| 5. Amélioration            | Examiner le surapprentissage et ajuster le modèle    | Expériences comparées                     | Décisions expliquées à partir de résultats mesurés             |
| 6. Évaluation              | Évaluer une seule fois le modèle final sur le test   | `metrics.json` + matrice de confusion     | Métriques calculées sur les labels et prédictions alignés      |
| 7. Sauvegarde              | Sauvegarder le modèle sélectionné                    | `models/best_model.keras`                 | Le modèle sauvegardé peut être rechargé                        |
| 8. Interface               | Permettre l’analyse d’une image dans Streamlit       | `app.py`                                  | Chargement, prétraitement et affichage fonctionnent            |
| 9. Documentation           | Expliquer installation, usage et limites             | README + guide                            | Une autre personne peut reproduire la démonstration            |
| 10. Livraison              | Vérifier dépôt Git et tâches Jira                    | Commit final + backlog actualisé          | Pas de secrets/données brutes ajoutés par erreur               |

## Détail des phases

### Phase 1 — Cadrage et dataset

- Décrire l’objectif de classification et les quatre classes.
- Noter la source du dataset, sa licence et les conditions d’utilisation.
- Vérifier que les dossiers correspondent bien aux classes attendues.
- Définir le périmètre : prototype éducatif, pas de diagnostic médical.

**Documentation à écrire :** pourquoi le problème est une classification multi-classes, ce que le CNN apprend, et les limites des données.

### Phase 2 — EDA et contrôle qualité

- Parcourir récursivement les dossiers.
- Vérifier les extensions prises en charge et les fichiers illisibles.
- Compter les images par classe.
- Afficher un échantillon d’images avec leur label.
- Rechercher les dimensions inhabituelles et les doublons potentiels.

**Documentation à écrire :** distribution des classes, problèmes observés et décisions prises.

### Phase 3 — Prétraitement et séparation

- Lire correctement les images et gérer les canaux couleur.
- Redimensionner selon la taille d’entrée du modèle (par exemple 224 × 224 si c’est celle utilisée à l’entraînement).
- Normaliser de la même façon pendant l’entraînement et l’inférence.
- Séparer les données en train/validation/test, idéalement de façon stratifiée si les données le permettent.
- Éviter les fuites de données et le mélange du jeu de test.

**Documentation à écrire :** pourquoi chaque transformation est appliquée et comment les labels sont encodés.

### Phase 4 — CNN et entraînement

- Expliquer les couches convolutionnelles, ReLU, pooling et couches de classification.
- Vérifier la fonction de perte et la forme de sortie selon l’encodage des labels.
- Conserver l’objet retourné par `model.fit()` dans `history`.
- Suivre `accuracy`, `val_accuracy`, `loss` et `val_loss`.
- Sauvegarder le meilleur modèle selon un critère de validation choisi.

**Documentation à écrire :** architecture, paramètres, nombre d’époques, batch size, optimiseur et choix de sauvegarde.

### Phase 5 — Analyse et amélioration

- Comparer les courbes d’entraînement et de validation.
- Rechercher un éventuel surapprentissage ou sous-apprentissage.
- Si nécessaire, tester une amélioration à la fois (augmentation de données, dropout, early stopping, etc.).
- Noter les expériences et ne pas choisir un modèle en regardant uniquement l’accuracy d’entraînement.

### Phase 6 — Évaluation finale

- Prédire les images du jeu de test.
- S’assurer que `y_true` et `y_pred` correspondent aux mêmes images dans le même ordre.
- Calculer accuracy, precision, recall et F1-score.
- Générer la matrice de confusion et le rapport par classe.
- Exporter les métriques et les historiques disponibles vers `metrics.json`.
- Expliquer les erreurs fréquentes sans prétendre que les résultats démontrent une aptitude clinique.

### Phase 7 — Application Streamlit

- Charger le modèle une seule fois avec un mécanisme de cache.
- Valider le fichier envoyé et appliquer le prétraitement exact de l’entraînement.
- Afficher la classe prédite et les scores, en expliquant qu’ils ne constituent pas un diagnostic.
- Afficher les métriques et la matrice de confusion à partir de `metrics.json`.
- Afficher les courbes si l’historique est disponible.
- Tester une image valide, un fichier non image et un modèle manquant.

### Phase 8 — Livraison et Git

- Vérifier README, guide, dépendances et structure réelle du dépôt.
- Ajouter `.gitignore` pour environnements, caches, secrets et datasets non destinés au dépôt.
- Vérifier que le modèle peut être distribué selon les contraintes de taille et de licence.
- Tester les instructions depuis un environnement propre si possible.
- Mettre à jour les tickets Jira et créer un commit descriptif.

## Définition de « terminé »

Une tâche est terminée lorsque le code fonctionne, que le résultat a été vérifié, que les décisions importantes sont expliquées et que la documentation associée a été mise à jour. Ne marque pas une étape comme terminée sur la seule base d’un code écrit mais non exécuté.

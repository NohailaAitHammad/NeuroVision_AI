# NeuroVision AI — Classification de tumeurs cérébrales à partir d’IRM

## 1. Présentation

**NeuroVision AI** est un projet pédagogique de Deep Learning qui vise à classifier des images d’IRM cérébrales en quatre catégories :

- **Glioma**
- **Méningioma**
- **Non Tumor**
- **Pituitary**

Le projet utilise un réseau de neurones convolutif (**CNN**) pour apprendre des caractéristiques visuelles à partir des images. Une interface **Streamlit** permet à l’utilisateur de charger une image et de visualiser la classe prédite ainsi que les scores du modèle.

> **Avertissement médical :** ce projet est un prototype éducatif. Il ne constitue pas un dispositif médical et ne doit jamais être utilisé pour établir, confirmer ou exclure un diagnostic. Toute interprétation clinique relève de professionnels de santé qualifiés.

## 2. Objectifs pédagogiques

- Comprendre le parcours d’un projet de classification d’images.
- Explorer et vérifier un dataset organisé par classes.
- Prétraiter les images de manière cohérente entre entraînement et prédiction.
- Entraîner et évaluer un CNN avec TensorFlow/Keras.
- Analyser les métriques, la matrice de confusion et les courbes d’apprentissage.
- Sauvegarder le meilleur modèle et exposer une démonstration avec Streamlit.
- Documenter le projet et versionner les changements avec Git.

## 3. Technologies

- Python
- TensorFlow / Keras
- OpenCV et/ou Pillow pour le traitement des images
- NumPy et Pandas
- scikit-learn pour les métriques d’évaluation
- Matplotlib pour les visualisations éventuelles
- Streamlit pour l’interface web
- Git et GitHub pour le versionnement

Les versions réellement utilisées doivent être précisées dans `requirements.txt` ou `pyproject.toml`.

## 4. Classes et ordre des labels

L’ordre des classes doit être **identique** à celui utilisé lors de l’entraînement du modèle. Dans la configuration actuelle du projet, l’ordre attendu est :

```python
CLASS_NAMES = [
    "Glioma",
    "Méningioma",
    "Non Tumor",
    "Pituitary",
]
```

Vérifier cet ordre dans le code qui construit les labels du dataset. Ne pas le modifier uniquement pour l’affichage : une différence d’ordre peut associer une prédiction à la mauvaise classe.

## 5. Structure de fichiers proposée

Adapte cette arborescence à tes fichiers réels. Elle est une proposition de rangement, pas une affirmation sur le contenu actuel du dépôt.

```text
NeuroVision_AI/
├── app.py                     # Interface Streamlit
├── README.md                  # Présentation et instructions
├── ROADMAP.md                 # Plan de réalisation
├── metrics.json               # Métriques calculées après évaluation
├── models/
│   └── best_model.keras       # Modèle entraîné sauvegardé
├── notebooks/
│   └── training.ipynb        # Exploration, entraînement et évaluation
└──
```

**Données :** ne pas versionner automatiquement l’intégralité du dataset ni des fichiers volumineux. Vérifier les droits d’utilisation du dataset et utiliser `.gitignore` pour exclure les données brutes, environnements virtuels, caches et secrets.

## 6. Installation

### Prérequis

- Python compatible avec la version de TensorFlow choisie.
- Git.
- Un environnement virtuel recommandé.
- Le fichier de modèle entraîné `models/best_model.keras` pour lancer la prédiction dans l’interface.

### Créer un environnement virtuel

Windows (PowerShell) :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS / WSL :

```bash
git clone https://github.com/NohailaAitHammad/NeuroVision_AI.git
python -m venv .venv
source .venv/bin/activate
```

### Installer les dépendances

```bash
python -m uv run
pip install -r requirements.txt
```

Le fichier `requirements.txt` doit refléter les bibliothèques réellement utilisées. Si TensorFlow ne s’installe pas dans l’environnement choisi, vérifier la compatibilité entre la version de Python, TensorFlow et le système d’exploitation.

## 7. Exécution

Depuis le dossier racine du projet :

```bash
streamlit run app.py
```

Si l’application indique que le modèle est introuvable, vérifier que le fichier `models/best_model.keras` existe et que le chemin défini dans `app.py` correspond à son emplacement.

Pour entraîner ou réévaluer le modèle, ouvrir le notebook d’entraînement ou lancer le script prévu par le projet. Les commandes exactes doivent être adaptées aux noms réels des notebooks/scripts et aux chemins du dataset.

## 8. Évaluation du modèle

L’évaluation doit être réalisée sur un jeu de test séparé, jamais utilisé pour ajuster les poids du modèle. Le fichier `metrics.json` peut contenir :

- `accuracy`
- `precision`, `recall` et `f1_score` (préciser la moyenne utilisée, par exemple `weighted`)
- `class_names`
- `confusion_matrix`
- `classification_report`
- `learning_curves` avec les historiques `accuracy`, `val_accuracy`, `loss` et `val_loss`, si l’objet `History` a été conservé pendant `model.fit()`.

Les métriques doivent être calculées à partir des prédictions réelles du modèle. Les exemples de scores dans la documentation ou l’interface ne doivent pas être présentés comme des résultats obtenus.

## 9. Reproductibilité et bonnes pratiques

1. Garder la même taille d’image et la même normalisation entre entraînement et prédiction.
2. Vérifier l’ordre des classes et la correspondance label ↔ nom.
3. Évaluer le modèle sur des données de test indépendantes.
4. Sauvegarder le modèle sélectionné et les métriques correspondantes.
5. Enregistrer les versions de dépendances et les étapes de prétraitement.
6. Éviter de publier des données sensibles ou des données sans autorisation.
7. Documenter les limites, les erreurs de classification et les biais possibles du dataset.

## 10. Limites

Les performances dépendent de la qualité, de la représentativité et de l’étiquetage du dataset. De bons scores sur un jeu de test ne garantissent pas une performance équivalente sur des données cliniques externes. Le projet n’est pas validé pour un usage médical.

## 11. État du projet

À compléter au fur et à mesure :

- Vérification du dataset et des labels
- Prétraitement documenté
- Entraînement reproductible
- Meilleur modèle sauvegardé
- Évaluation sur le jeu de test
- `metrics.json` généré depuis les résultats réels
- Interface Streamlit testée
- Documentation et instructions vérifiées

# 🧠 Atelier TensorFlow – Prédiction de la consommation énergétique

## 📌 Contexte

Une entreprise dispose de données provenant de plusieurs bâtiments équipés de capteurs : température moyenne, humidité, nombre d'occupants, heure, jour de la semaine, surface du bâtiment et consommation énergétique. L'objectif de cet atelier est de construire, avec **TensorFlow/Keras**, un réseau de neurones capable de **prédire la consommation énergétique** d'un bâtiment à partir de la température, de l'humidité et du nombre d'occupants.

Atelier réalisé dans le cadre du cours **Python pour ML et IA 2** – P1 IA, Orange Digital Center (ODC) .

## 🎯 Objectifs

- Générer un dataset synthétique réaliste
- Construire un réseau de neurones séquentiel avec Keras
- Compiler, entraîner et évaluer un modèle de régression
- Visualiser et interpréter les courbes d'apprentissage
- Sauvegarder, recharger et réutiliser le modèle pour l'inférence

## 🗂️ Structure du projet

```
atelier_tensorflow_iot/
├── notebooks/
│   └── atelier_tensorflow_iot.ipynb
└── models/
    └── modele_consommation.keras
```

## 🧩 Contenu de l'atelier

| Partie | Sujet |
|---|---|
| 0 | Mise en place de l'environnement (structure, import TensorFlow/Matplotlib/NumPy) |
| 1 | Génération du dataset (1000 observations simulées) |
| 2 | Découpage Train/Test (80/20, reproductible) |
| 3 | Création du modèle (réseau séquentiel Dense 16→8→1) |
| 4 | Compilation (optimiseur, fonction de perte, métrique MAE) |
| 5 | Entraînement (50 époques, batch de 16, 20% de validation) |
| 6 | Évaluation sur les données de test |
| 7 | Prédictions et comparaison avec les valeurs réelles |
| 8 | Sauvegarde du modèle (`.keras`) |
| 9 | Chargement et réutilisation du modèle |
| 10 | Fonction d'inférence `predire_consommation()` |
| 11 | Bonus – fonctionnalité additionnelle pertinente |

## 🧮 Génération des données (Partie 1)

| Variable | Distribution |
|---|---|
| `temperature` | Loi normale, moyenne 25 °C, écart-type 4 °C |
| `humidite` | Loi uniforme entre 30 % et 80 % |
| `occupants` | Entiers uniformes entre 1 et 49 |

**Formule de la consommation (cible) :**

```
consommation = 50                          # base (0°C, 0% humidité, pièce vide)
             + 5   × temperature
             + 1.5 × humidite
             + 4   × occupants
             + bruit(moyenne=0, écart-type=10)   # variation aléatoire réaliste
```

- `X` : matrice de caractéristiques (1000 × 3), format `float32`
- `y` : cible (consommation), format `float32`

## 🏗️ Architecture du réseau de neurones

| Couche | Neurones | Activation |
|---|---|---|
| Couche 1 (entrée) | 16 | ReLU |
| Couche 2 (cachée) | 8 | ReLU |
| Couche 3 (sortie) | 1 | linéaire (régression) |

## ⚙️ Compilation et entraînement

- **Optimiseur** : ajustement itératif des poids (ex. Adam)
- **Fonction de perte** : mesure de l'erreur de prédiction, adaptée à la régression (ex. MSE)
- **Métrique suivie** : erreur absolue moyenne (**MAE**)
- **Entraînement** : 50 époques, batch size = 16, 20 % des données d'entraînement réservées à la validation
- L'historique (`history`) permet de tracer l'évolution de la perte (train/validation) et de diagnostiquer un éventuel sur/sous-apprentissage

## 📊 Évaluation et prédiction

- Évaluation finale sur l'ensemble de test (perte + MAE)
- Comparaison graphique prédictions vs valeurs réelles : un modèle parfait alignerait tous les points sur la diagonale (y_pred = y_true)

## ▶️ Utilisation

```bash
pip install tensorflow matplotlib numpy jupyter
jupyter notebook notebooks/atelier_tensorflow_iot.ipynb
```

## 💾 Sauvegarde et inférence

```python
model.save("models/modele_consommation.keras")

from tensorflow import keras
modele_charge = keras.models.load_model("models/modele_consommation.keras")

def predire_consommation(temperature, humidite, occupants):
    import numpy as np
    entree = np.array([[temperature, humidite, occupants]], dtype="float32")
    return modele_charge.predict(entree)[0][0]
```

## 📦 Livrable attendu

Dossier `atelier_tensorflow_iot/` complet, poussé sur un dépôt public GitHub avec commits explicites au fur et à mesure.

## 👤 Auteure

**Rokhaya Coumba Diouf** –  parcours IA (P1 IA) Orange Digital Center (ODC)

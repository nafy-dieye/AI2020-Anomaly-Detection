# Détecteur d'anomalies — Django

Application Django exposant le modèle auto-encodeur (AI4I 2020, détection
d'anomalies non supervisée) via une interface web et une API JSON.

## Fonctionnalités

- **Page « Analyser »** (`/`) :
  - Seuil de détection actuel affiché en permanence
  - **Boutons de simulation rapide** : « Simuler un cas normal » / « Simuler une panne » pré-remplissent le formulaire avec des valeurs plausibles, sans saisie manuelle
  - Verdict immédiat avec **jauge visuelle** (position de l'erreur de reconstruction par rapport au seuil)
  - Bouton **🚩 Signaler** pour marquer un verdict jugé incorrect par un opérateur (faux positif/négatif)
- **Dashboard** (`/dashboard/`) : statistiques (lectures, taux d'anomalies, anomalies sur 24h, lectures signalées), journal des pannes horodaté (avec signalement possible directement depuis le tableau), graphique de l'erreur de reconstruction dans le temps, et **bouton de réinitialisation** (supprime tout l'historique, avec confirmation).
- **Simulation aléatoire** : les boutons « Simuler un cas normal / une panne » génèrent de nouvelles valeurs à chaque clic (pas toujours les mêmes), dans des plages réalistes — 3 profils de panne possibles tirés au hasard (usure/surcontrainte, dissipation thermique, puissance anormale).
- **API JSON** (`/api/predict/`) : mêmes prédictions, pour intégration externe.

Chaque prédiction (web ou API) est enregistrée en base (`SensorReading`) et alimente automatiquement le dashboard.

## Contenu

```
config/                  # Configuration du projet Django
detector/
  ml_model.py             # Chargement du modèle + feature engineering + inférence
  forms.py                 # Formulaire de saisie des lectures capteurs
  views.py                 # Vue web (index) + vue API (PredictAPIView)
  urls.py
  templates/detector/index.html
  ml_artifacts/            # autoencoder.keras, scaler.pkl, threshold.pkl, feature_names.pkl
requirements.txt
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Puis ouvrir http://127.0.0.1:8000/

## Utilisation

### Interface web

Formulaire avec les 5 mesures brutes (température air, température process,
vitesse de rotation, couple, usure outil). Les 5 features dérivées
(temperature_difference, mechanical_power, tool_stress, temperature_ratio,
torque_speed_ratio) sont calculées automatiquement côté serveur, dans le même
ordre que celui utilisé à l'entraînement (`feature_names.pkl`).

### API JSON

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 108
      }'
```

Réponse :

```json
{
  "reconstruction_error": 0.1523,
  "threshold": 0.9256,
  "threshold_percentile": 97,
  "is_anomaly": false,
  "severity_ratio": 0.16,
  "engineered_features": { "...": "..." }
}
```

## Points d'attention avant mise en production

- **`ALLOWED_HOSTS`** est actuellement à `["*"]` pour faciliter les tests —
  à restreindre au(x) nom(s) de domaine réel(s).
- **`DEBUG = True`** par défaut dans `settings.py` (généré par Django) — à
  passer à `False` en production, avec une vraie `SECRET_KEY` en variable
  d'environnement plutôt qu'en dur dans le fichier.
- Le modèle Keras est chargé **une seule fois** au premier appel (singleton
  thread-safe dans `AnomalyModel.get_instance()`), pas à chaque requête —
  important pour la latence en production. Si vous déployez avec plusieurs
  workers (gunicorn, uwsgi), chaque worker charge sa propre copie du modèle
  en mémoire (~quelques dizaines de Mo, sans souci pour un modèle aussi
  petit).
- Le endpoint `/api/predict/` est actuellement en `csrf_exempt` car pensé
  comme une API sans session — si vous l'appelez depuis un navigateur avec
  authentification par session plutôt qu'un client externe, envisagez une
  authentification par token (Django REST Framework + TokenAuthentication,
  par exemple) plutôt que de désactiver CSRF.
- Le seuil et le scaler ont été entraînés uniquement sur des données
  normales : toute dérive du process industriel (nouvelle machine,
  recalibration des capteurs) peut nécessiter un ré-entraînement.

## Comparaison Auto-encodeur / ANN

La page « Analyser » propose maintenant un switch entre :
- **Auto-encodeur** : erreur de reconstruction, seuil P97.
- **ANN** : classifieur supervisé, score sigmoïde et seuil 0,5.

Pour comparer les modèles, gardez exactement les mêmes valeurs capteurs,
analysez avec le premier modèle, puis basculez le switch sur le second et
cliquez à nouveau sur « Analyser ».

L'ANN utilise aussi la variable `Type` du dataset AI4I 2020.
L'encodage est H=0, L=1, M=2. Le choix du Type apparaît uniquement
quand l'ANN est sélectionné.

Les trois artefacts ANN sont inclus dans `detector/ml_artifacts/` :
`ann_model.keras`, `scaler_ann.pkl`, `ann_threshold.pkl`.


## ⚠️ Audit de cohérence ANN

Le notebook `PROJET_DEEP_LEARNING_audite_corrige_FINAL_COHERENT(2).ipynb`
contient une cellule où `X` retire `Type` avant l'entraînement de l'ANN.
Cependant, les artefacts ANN fournis avec ce projet (`ann_model.keras` et
`scaler_ann.pkl`) attendent **11 variables**, dont `Type` en première position.

Cette version Django est donc volontairement alignée sur **les artefacts
effectivement fournis**, afin que l'application fonctionne sans inventer un
modèle différent. Le sélecteur `H/L/M` reste nécessaire avec ces artefacts.

Pour être strictement identique au notebook (10 variables sans `Type`), il faut
réexécuter l'entraînement ANN du notebook après avoir vérifié la définition de
`X`, puis remplacer `ann_model.keras` et `scaler_ann.pkl` par les nouveaux
artefacts. Il ne faut pas simplement supprimer `Type` dans Django : le modèle
actuel refuserait alors l'entrée car il possède 11 entrées.

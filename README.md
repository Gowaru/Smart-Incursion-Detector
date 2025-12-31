# 🎯 Système de Détection et Tracking d'Objets en Temps Réel

Système de surveillance vidéo intelligent utilisant YOLOv8 pour détecter et suivre des objets (personnes, véhicules) en temps réel, avec alertes d'intrusion dans une zone définie.

## ✨ Fonctionnalités

- 🎥 **Multi-sources** : Support webcam et fichiers vidéo
- 🤖 **Détection YOLOv8** : Détection précise des personnes, voitures et motos
- 🔍 **Tracking multi-objets** : Suivi avec IDs persistants (ByteTrack)
- 🚨 **Système d'alertes** : Détection d'intrusion dans une zone configurable
- 📊 **Visualisation temps réel** : Bounding boxes, statistiques, FPS
- ⚙️ **Configuration centralisée** : Paramètres facilement modifiables

## 📁 Architecture du Projet

```
Computer Vision/
├── config/
│   └── config.py              # Configuration centralisée
├── utils/
│   ├── __init__.py
│   ├── video_source.py        # Gestion des sources vidéo
│   └── visualization.py       # Fonctions d'affichage
├── models/
│   ├── __init__.py
│   ├── alert_zone.py          # Gestion de la zone d'alerte
│   └── object_tracker.py      # Système de tracking principal
├── main.py                    # Point d'entrée
├── requirements.txt           # Dépendances
└── README.md                  # Documentation
```

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- Webcam (pour le mode temps réel) ou fichiers vidéo
- Windows, Linux ou macOS

### Installation des dépendances

```bash
# Cloner ou naviguer vers le répertoire du projet
cd "Computer Vision"

# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Sur Windows (CMD):
.\venv\Scripts\activate.bat
# Sur Linux/Mac:
source venv/bin/activate

# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances
pip install -r requirements.txt
```

Les dépendances installées :
- `opencv-python` : Traitement vidéo et affichage
- `ultralytics` : YOLOv8 avec tracking intégré
- `torch` : Backend PyTorch pour YOLO
- `numpy` : Calculs numériques

Au premier lancement, le modèle YOLOv8n (~6 MB) sera téléchargé automatiquement.

## 💻 Utilisation

### Lancement du système

```bash
python main.py
```

### Menu interactif

Au lancement, un menu vous permet de choisir la source vidéo :

```
  [1] Webcam (temps réel)
  [2] Fichier vidéo
  [0] Quitter
```

**Option 1 - Webcam** : Détection en temps réel depuis votre webcam

**Option 2 - Fichier vidéo** : Analyse d'un fichier vidéo local
- Formats supportés : MP4, AVI, MOV, MKV, FLV, WMV, M4V
- Entrez le chemin complet du fichier

### Contrôles pendant l'exécution

| Touche | Action |
|--------|--------|
| `Q` ou `ESC` | Quitter le programme |
| `R` | Réinitialiser les alertes |

### Exemple de session

```bash
$ python main.py

  [1] Webcam (temps réel)
  [2] Fichier vidéo
  [0] Quitter

Votre choix (0-2) : 1

🎥 Initialisation de la webcam...
✅ Webcam initialisée avec succès
   Résolution: 1280x720
   FPS: 30.0

🤖 Chargement du modèle YOLO: yolov8n.pt...
✅ Modèle chargé avec succès

🚀 SYSTÈME DÉMARRÉ
📹 Traitement en cours...
```

## ⚙️ Configuration

Tous les paramètres sont centralisés dans `config/config.py` :

### Modèle YOLO

```python
MODEL_NAME = "yolov8n.pt"       # n=nano, s=small, m=medium, l=large, x=extra
CONFIDENCE_THRESHOLD = 0.5       # Seuil de confiance (0.0 à 1.0)
```

### Classes cibles

```python
TARGET_CLASSES = ["person", "car", "motorbike"]
```

Pour ajouter d'autres classes (voir [classes COCO](https://docs.ultralytics.com/datasets/detect/coco/)) :

```python
TARGET_CLASSES = ["person", "car", "motorbike", "bicycle", "bus", "truck"]
CLASS_IDS = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
```

### Zone d'alerte

Coordonnées relatives (0.0 à 1.0 = pourcentage de l'écran) :

```python
ALERT_ZONE = (0.2, 0.2, 0.8, 0.8)  # (x1, y1, x2, y2)
# Zone centrale : de 20% à 80% de l'écran
```

Exemples de zones :
- Zone gauche : `(0.0, 0.0, 0.3, 1.0)`
- Zone droite : `(0.7, 0.0, 1.0, 1.0)`
- Zone centrale petite : `(0.3, 0.3, 0.7, 0.7)`
- Zone basse : `(0.0, 0.6, 1.0, 1.0)`

### Affichage

```python
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FRAME_RESIZE = True              # Redimensionner pour optimiser performances
```

### Couleurs

```python
COLORS = {
    "person": (0, 255, 0),      # Vert (BGR)
    "car": (255, 0, 0),         # Bleu
    "motorbike": (0, 165, 255)  # Orange
}
```

## 🎬 Fonctionnement

### Pipeline de traitement

1. **Capture** : Lecture de la frame depuis webcam ou vidéo
2. **Détection** : YOLOv8 détecte les objets avec scores de confiance
3. **Filtrage** : Seules les classes cibles sont conservées
4. **Tracking** : Attribution d'IDs uniques et persistants (ByteTrack)
5. **Vérification** : Détection d'intrusion dans la zone d'alerte
6. **Alerte** : Notification visuelle et console si intrusion
7. **Affichage** : Rendu avec bounding boxes, IDs, statistiques

### Système d'alertes

Quand un objet entre dans la zone d'alerte :

**Visuel** :
- Zone devient rouge
- Message "ALERTE INTRUSION!" affiché en haut
- Bounding box de l'objet en surbrillance

**Console** :
```
======================================================================
🚨 ALERTE INTRUSION!
======================================================================
Timestamp    : 2025-12-30 14:30:15
Objet ID     : 42
Classe       : person
Position     : x=450, y=320
Total alertes: 3
======================================================================
```

**Unicité** : Chaque objet ne déclenche qu'une seule alerte (évite les répétitions)

**Réinitialisation** : Appuyez sur `R` pour permettre de nouvelles alertes

## 📊 Performances

### FPS attendus

| Modèle | Webcam 720p | Webcam 1080p | Fichier vidéo |
|--------|-------------|--------------|---------------|
| yolov8n | 25-30 FPS | 15-20 FPS | 30-40 FPS |
| yolov8s | 20-25 FPS | 12-15 FPS | 25-30 FPS |
| yolov8m | 10-15 FPS | 8-12 FPS | 15-20 FPS |

*Sur CPU i7, 16GB RAM. GPU accélère significativement.*

### Optimisation

Pour améliorer les performances :

1. **Utiliser un modèle plus léger** : `yolov8n.pt` (déjà par défaut)
2. **Réduire la résolution** : Diminuer `DISPLAY_WIDTH` et `DISPLAY_HEIGHT`
3. **Augmenter le seuil** : `CONFIDENCE_THRESHOLD = 0.6` (moins de détections)
4. **Utiliser GPU** : `DEVICE = 0` dans config (si CUDA disponible)

## 🐛 Dépannage

### Webcam non détectée

```
❌ Erreur: Impossible d'ouvrir la webcam.
```

**Solutions** :
- Vérifier que la webcam est branchée
- Fermer les autres applications utilisant la webcam
- Essayer un autre index : Modifier `VideoSourceManager._init_webcam()` avec `cv2.VideoCapture(1)` ou `2`

### Erreur de modèle YOLO

```
❌ Erreur: Model 'yolov8n.pt' not found
```

**Solutions** :
- Connexion Internet requise pour le premier téléchargement
- Le modèle se télécharge automatiquement (~6 MB)
- Vérifier l'espace disque disponible

### Performance faible

**Solutions** :
- Réduire la résolution dans `config.py`
- Utiliser `yolov8n.pt` (modèle nano, le plus rapide)
- Activer GPU si disponible : `DEVICE = 0`
- Fermer les autres applications gourmandes

### Pas de détection

**Solutions** :
- Diminuer `CONFIDENCE_THRESHOLD` (ex: 0.3)
- Vérifier l'éclairage de la scène
- S'assurer que les objets sont dans les classes cibles
- Augmenter la taille des objets dans la frame

## 📝 Notes techniques

### Tracking

Le système utilise **ByteTrack**, un algorithme de tracking robuste intégré à Ultralytics :
- Associe les détections entre frames
- Maintient les IDs même avec occlusions temporaires
- Gère l'entrée/sortie d'objets dans le champ

### Classes COCO

YOLOv8 est pré-entraîné sur COCO dataset (80 classes). IDs des classes principales :

- 0: person
- 1: bicycle  
- 2: car
- 3: motorbike
- 5: bus
- 7: truck

[Liste complète des classes COCO](https://docs.ultralytics.com/datasets/detect/coco/)

## 🔮 Extensions possibles

- ✅ Zones multiples avec alertes différenciées
- ✅ Enregistrement vidéo lors d'alertes
- ✅ Notifications (email, SMS) en cas d'intrusion
- ✅ Base de données pour historique des événements
- ✅ Interface web pour configuration à distance
- ✅ Comptage d'objets (entrées/sorties)

## 📄 Licence

Projet éducatif - Utilisation libre

## 🙏 Remerciements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

**Auteur** : Système de détection et tracking d'objets  
**Version** : 1.0.0  
**Date** : 2025-12-30

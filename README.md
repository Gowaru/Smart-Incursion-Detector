# 🎯 Smart Incursion Detector (YOLO11 + Supervision)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Model-YOLO11-red.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Library-Supervision-green.svg)](https://github.com/roboflow/supervision)

Système intelligent de vidéosurveillance et de détection d'intrusions optimisé pour la détection d'objets complexes (sacs, bagages, personnes) à longue portée.

---

## ✨ Fonctionnalités Clés
- **Détection Multi-Classes** : Optimisé pour `person`, `car`, `motorbike`, `backpack`, `handbag`, `suitcase`.
- **Analyse Longue Portée** : Traitement en haute résolution (HD) pour identifier les objets lointains.
- **Rendu Fluide (Decoupled Rendering)** : Maintien de 30 FPS pour l'affichage tout en effectuant l'analyse IA en arrière-plan.
- **Tracking Robuste** : Utilisation de **ByteTrack** et filtres de Kalman pour la persistance des objets.
- **Zones d'Alerte Dynamiques** : Déclenchement d'alertes visuelles et logs lors de l'entrée dans une zone protégée.
- **Visualisation de Données** : Heatmaps d'activité, affichage des trajectoires et graphiques de performance FPS.

---

## 🛠️ Stack Technique
- **IA** : YOLO11 (Ultralytics) - Modèles Nano (`yolo11n.pt`).
- **Vision Library** : Supervision (Roboflow) pour l'annotation et le traitement des détections.
- **Tracking** : ByteTrack (Yaml config).
- **Core** : OpenCV, PyTorch, NumPy.

---

## 🚧 Défis Techniques & Solutions

### 1. Optimisation CPU (Le Triangle Impossible)
**Défi** : Obtenir de la haute résolution (720p) pour voir loin, tout en restant fluide (30 FPS) sur un processeur sans accélération GPU stable.
**Solution** : Mise en place du **Decoupled Rendering**. L'affichage vidéo tourne à plein régime, tandis que l'IA analyse une image sur 12 (`FRAME_SKIP = 12`).

### 2. Détection des Sacs (Objets Superposés)
**Défi** : L'IA confond souvent les sacs avec la personne qui les porte ou les ignore à cause du chevauchement (NMS).
**Solution** : Abaissement du seuil d'exclusion **IOU à 0.3** et mise en place de **Seuils Adaptatifs par Classe** (Bags @ 0.25 vs People @ 0.50).

### 3. Filtrage des Faux Positifs (Pieds vs Valises)
**Défi** : À longue distance, la forme des chaussures peut être interprétée comme une petite valise.
**Solution** : Calibration fine des seuils de confiance : `Suitcase` relevé à **0.45** pour exiger une certitude quasi-totale du modèle.

---

## 🚀 Guide de Démarrage

### Installation
1. Clonez le dépôt.
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

### Lancement
```bash
python main.py
```

### Contrôles In-App
| Touche | Action |
| :--- | :--- |
| `H` | Afficher / Masquer la **Heatmap** d'activité |
| `Q` ou `Esc` | Quitter proprement le système |

---

## ⚙️ Configuration (`config/config.py`)
Le fichier de configuration centralise tous les paramètres critiques :
- `MODEL_NAME` : Choix du modèle (n, s, m, l, x).
- `PROCESSING_WIDTH` : Détermine la portée de détection (1280 recommandée pour la distance).
- `ADAPTIVE_CONFIDENCE` : Réglage fin de la sensibilité pour chaque type d'objet.

---

## 🙏 Remerciements
- **Ultralytics** : Pour leur travail remarquable sur YOLO11.
- **Roboflow** : Pour la bibliothèque Supervision qui facilite l'analyse visuelle.
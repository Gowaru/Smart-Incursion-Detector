# 🎯 Smart Incursion Detector (YOLO11 + Supervision)

Système avancé de détection d'intrusions optimisé pour la détection d'objets difficiles (sacs, valises) sur du matériel grand public.

---

## �️ Technologies Utilisées
Le projet repose sur un écosystème de pointe en Computer Vision :
- **IA Core** : [YOLO11 par Ultralytics](https://github.com/ultralytics/ultralytics) (Modèles Nano/Small)
- **Visualisation & Analyse** : [Supervision par Roboflow](https://github.com/roboflow/supervision)
- **Tracking Logic** : **ByteTrack** (pour la persistance des IDs d'objets)
- **Traitement d'Image** : **OpenCV** (gestion des flux vidéo et interface fenêtrée)
- **Backend Numérique** : **PyTorch** & **NumPy**
- **Optimisation** : Algorithme de filtrage de Kalman et Multi-threading.

---

## 🚀 Utilisation

### Lancement
Exécutez la commande suivante dans votre terminal :
```bash
python main.py
```

### Interface Interactive
Au démarrage, un menu CLI vous permet de choisir :
1.  **Webcam** : Flux en temps réel.
2.  **Fichier Vidéo** : Chemin vers un fichier local (.mp4, .avi, etc.).

### Raccourcis Clavier (Pendant l'exécution)
| Touche | Action |
| :--- | :--- |
| `H` | Activer/Désactiver la **Heatmap** d'activité |
| `Q` | Quitter proprement le système |
| `Esc` | Quitter l'affichage vidéo |

---

## 🚧 Défis et Difficultés rencontrés

### 1. Le "Triangle Impossible" (CPU-only)
Sur CPU, nous avons dû équilibrer trois facteurs contradictoires :
- **Haute Résolution** (720p) vs **Précision** (YOLO11s) vs **Fluidité** (30 FPS).
- **Solution** : Utilisation du **Decoupled Rendering** (Affichage 30 FPS, IA traitée 1 image sur 8).

### 2. Détection des Sacs et Mobilité
Les sacs à dos et sacs à main sont difficiles car souvent collés à une personne.
- **Solution** : Abaissement agressif du seuil **IOU (0.3)** et mise en place de **seuils de confiance adaptatifs** (très sensibles pour les sacs à 0.15).

### 3. Instabilité de la Carte Graphique (Quadro T1000)
- **Problème** : Des erreurs de types de données (Half vs Float) ont forcé le retour au CPU.
- **Leçon** : Importance de la compatibilité exacte entre PyTorch-CUDA et les drivers NVIDIA.

---

## ⚙️ Configuration Recommandée (`config/config.py`)
- **Modèle** : `yolo11n.pt`
- **Résolution IA** : `1024x576`
- **Classes cibles** : `person`, `car`, `motorbike`, `backpack`, `handbag`, `suitcase`.

---

## 🙏 Remerciements
- Un grand merci à **Ultralytics** pour leur modèle YOLO11 exceptionnel et leur écosystème open-source qui rend ces technologies accessibles.
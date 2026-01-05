"""
Configuration centralisée pour le système de détection et tracking d'objets.
Tous les paramètres du système sont définis ici pour faciliter la maintenance.
"""

# ============================================================================
# MODÈLE YOLO
# ============================================================================

# Nom du modèle YOLOv8 à utiliser (n=nano, s=small, m=medium, l=large, x=extra-large)
# yolov8n.pt est le plus rapide, idéal pour le temps réel
MODEL_NAME = "yolo11n.pt"

# Seuil de confiance minimal pour la détection (0.0 à 1.0)
# Plus élevé = moins de faux positifs, mais peut manquer des objets
CONFIDENCE_THRESHOLD = 0.15  # Seuil de base très bas pour laisser passer les sacs vers le filtrage adaptatif

# Seuil IOU (Intersection Over Union) pour le NMS (Non-Maximum Suppression)
IOU_THRESHOLD = 0.3   # Plus bas pour mieux détecter les objets chevauchants (ex: sac sur le dos)

# ============================================================================
# CLASSES CIBLES
# ============================================================================

# Classes d'objets à détecter (noms COCO)
TARGET_CLASSES = ["person", "car", "motorbike", "backpack", "handbag", "suitcase"]

# Mapping des IDs COCO vers les noms de classes
# COCO dataset: 0=person, 2=car, 3=motorbike, etc.
CLASS_IDS = {
    0: "person",
    2: "car",
    3: "motorbike",
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}

# Mapping inverse pour faciliter la recherche
CLASS_NAMES_TO_IDS = {v: k for k, v in CLASS_IDS.items()}

# ============================================================================
# ZONE D'ALERTE
# ============================================================================

# Coordonnées de la zone d'alerte (x1, y1, x2, y2) en pourcentage (0.0 à 1.0)
# (0.2, 0.2, 0.8, 0.8) = zone centrale, de 20% à 80% de l'écran
ALERT_ZONE = (0.2, 0.2, 0.8, 0.8)

# ============================================================================
# AFFICHAGE
# ============================================================================

# Résolution d'affichage
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Activer le redimensionnement des frames pour améliorer les performances
FRAME_RESIZE = True

# Épaisseur des lignes pour les bounding boxes et la zone d'alerte
BOX_THICKNESS = 2
ZONE_THICKNESS = 3

# Taille de la police pour les labels
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Transparence de l'overlay de la zone d'alerte (0.0 = transparent, 1.0 = opaque)
ZONE_ALPHA = 0.3

# ============================================================================
# COULEURS (format BGR pour OpenCV)
# ============================================================================

# Couleurs par classe d'objet
COLORS = {
    "person": (0, 255, 0),      # Vert
    "car": (255, 0, 0),         # Bleu
    "motorbike": (0, 165, 255), # Orange
}

# Couleur d'alerte (quand un objet entre dans la zone)
ALERT_COLOR = (0, 0, 255)  # Rouge

# Couleur de la zone d'alerte (mode normal)
ZONE_COLOR = (255, 255, 0)  # Cyan

# Couleur du texte
TEXT_COLOR = (255, 255, 255)  # Blanc
TEXT_BG_COLOR = (0, 0, 0)     # Noir

# ============================================================================
# TRACKING
# ============================================================================

# Type de tracker à utiliser avec YOLO
# Options: 'botsort', 'bytetrack' 
# Type de tracker à utiliser avec YOLO
# Options: 'botsort', 'bytetrack' 
import os
TRACKER_TYPE = os.path.abspath("bytetrack.yaml")

# Persistance des tracks (nombre de frames avant de perdre un ID)
TRACK_BUFFER = 30

# ============================================================================
# PERFORMANCE
# ============================================================================

# Activer le mode verbose pour YOLO (affiche plus d'informations)
VERBOSE = False

# Device à utiliser: 'cpu', 'cuda', '0', '1', etc.
# None = détection automatique
DEVICE = 'cpu'

# FPS cible pour l'affichage (None = pas de limitation)
TARGET_FPS = None

# ============================================================================
# OPTIMISATIONS DE PERFORMANCE
# ============================================================================

# Activer le threading pour la capture vidéo
ENABLE_THREADING = True

# Taille du buffer de frames (pour threading)
FRAME_BUFFER_SIZE = 2

# Skip frames pour améliorer le FPS (traiter 1 frame sur N)
# 1 = traiter toutes les frames, 2 = une sur deux, etc.
FRAME_SKIP = 8  # Plus espacé pour garantir la fluidité CPU (IA ~4 fois/sec)

# Utiliser un modèle demi-précision (FP16) si GPU disponible
USE_HALF_PRECISION = False

# Résolution de traitement (différente de l'affichage)
# Plus petit = plus rapide, mais moins précis
PROCESSING_WIDTH = 1024  # Compromis idéal HD/Vitesse sur CPU
PROCESSING_HEIGHT = 576

# ============================================================================
# OPTIMISATIONS DE PRÉCISION
# ============================================================================

# Activer le filtrage Kalman pour smooth tracking
ENABLE_KALMAN_FILTER = True

# Seuils de confiance adaptatifs par classe
# Seuils de confiance adaptatifs par classe
ADAPTIVE_CONFIDENCE = {
    "person": 0.50,      # Augmenté pour éviter les faux positifs
    "car": 0.60,         # Augmenté
    "motorbike": 0.55,   # Augmenté
    "backpack": 0.15,    # Sensibilité maximale pour les sacs
    "handbag": 0.15,     # Sensibilité maximale
    "suitcase": 0.15     # Sensibilité maximale
}

# Nombre minimum de frames consécutives pour valider une détection
MIN_DETECTION_FRAMES = 2

# Distance maximale pour associer des détections (pixels)
MAX_ASSOCIATION_DISTANCE = 100

# ============================================================================
# OPTIMISATIONS DE VISUALISATION
# ============================================================================

# Activer l'affichage des trajectoires
SHOW_TRAJECTORIES = True

# Longueur maximale des trajectoires (nombre de points)
TRAJECTORY_LENGTH = 30

# Épaisseur de la ligne de trajectoire
TRAJECTORY_THICKNESS = 2

# Activer la heatmap des zones d'activité
SHOW_HEATMAP = False

# Transparence de la heatmap (0.0 à 1.0)
HEATMAP_ALPHA = 0.3

# Activer le graphique de performance FPS
SHOW_FPS_GRAPH = True

# Longueur de l'historique FPS pour le graphe
FPS_HISTORY_LENGTH = 100

# Afficher les informations de debug
SHOW_DEBUG_INFO = False

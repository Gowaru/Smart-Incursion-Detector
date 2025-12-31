"""
Fonctions de visualisation pour le système de tracking.
Gère l'affichage des bounding boxes, zones d'alerte, statistiques, etc.
"""

import cv2
import numpy as np
from typing import Tuple

# Import de la configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    class_name: str,
    color: Tuple[int, int, int],
    confidence: float = None
) -> None:
    """
    Dessine une bounding box sur la frame avec l'ID de tracking et la classe.
    
    Args:
        frame: Frame sur laquelle dessiner
        bbox: Coordonnées (x1, y1, x2, y2) de la bounding box
        track_id: ID unique de l'objet tracké
        class_name: Nom de la classe de l'objet
        color: Couleur BGR de la box
        confidence: Score de confiance (optionnel)
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Dessiner le rectangle principal avec coins arrondis (simulés)
    thickness = config.BOX_THICKNESS
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Ajouter des coins décoratifs
    corner_length = 15
    corner_thickness = thickness + 1
    # Coin haut gauche
    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
    # Coin haut droit
    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
    # Coin bas gauche
    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
    # Coin bas droit
    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
    
    # Préparer le label
    if confidence is not None:
        label = f"ID:{track_id} {class_name.upper()} {confidence:.0%}"
    else:
        label = f"ID:{track_id} {class_name.upper()}"
    
    # Calculer la taille du texte
    font_scale = 0.55
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        font_thickness
    )
    
    # Fond avec dégradé (simulé avec rectangle semi-transparent)
    padding = 6
    bg_top = y1 - text_height - baseline - padding * 2
    bg_bottom = y1
    
    # Dessiner le fond du label avec bordure
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1, bg_top),
        (x1 + text_width + padding * 2, bg_bottom),
        color,
        -1
    )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Bordure du label
    cv2.rectangle(
        frame,
        (x1, bg_top),
        (x1 + text_width + padding * 2, bg_bottom),
        color,
        1
    )
    
    # Dessiner le texte
    cv2.putText(
        frame,
        label,
        (x1 + padding, y1 - baseline - padding),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )


def draw_alert_zone(
    frame: np.ndarray,
    zone_coords: Tuple[int, int, int, int],
    is_alert_active: bool = False
) -> None:
    """
    Dessine la zone d'alerte sur la frame avec un overlay semi-transparent.
    
    Args:
        frame: Frame sur laquelle dessiner
        zone_coords: Coordonnées (x1, y1, x2, y2) de la zone
        is_alert_active: True si une alerte est en cours (change la couleur)
    """
    x1, y1, x2, y2 = map(int, zone_coords)
    
    # Choisir la couleur selon l'état d'alerte
    color = config.ALERT_COLOR if is_alert_active else config.ZONE_COLOR
    
    # Créer un overlay semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Appliquer l'overlay avec transparence
    cv2.addWeighted(overlay, config.ZONE_ALPHA, frame, 1 - config.ZONE_ALPHA, 0, frame)
    
    # Dessiner le contour de la zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.ZONE_THICKNESS)
    
    # Ajouter le label "ZONE D'ALERTE"
    label = "ZONE D'ALERTE" if not is_alert_active else "ALERTE ACTIVE!"
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        config.FONT_THICKNESS
    )
    
    # Position du label au centre haut de la zone
    label_x = x1 + (x2 - x1 - text_width) // 2
    label_y = y1 + text_height + 10
    
    # Fond du label
    cv2.rectangle(
        frame,
        (label_x - 5, label_y - text_height - 5),
        (label_x + text_width + 5, label_y + 5),
        config.TEXT_BG_COLOR,
        -1
    )
    
    # Texte du label
    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        color,
        config.FONT_THICKNESS,
        cv2.LINE_AA
    )


def draw_alert_message(
    frame: np.ndarray,
    message: str,
    position: str = 'top'
) -> None:
    """
    Affiche un message d'alerte proéminent sur la frame.
    
    Args:
        frame: Frame sur laquelle dessiner
        message: Message à afficher
        position: Position du message ('top', 'center', 'bottom')
    """
    height, width = frame.shape[:2]
    
    # Calculer la taille du texte avec une police plus grande
    font_scale = 1.5
    thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(
        message,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )
    
    # Déterminer la position
    if position == 'top':
        x = (width - text_width) // 2
        y = text_height + 30
    elif position == 'center':
        x = (width - text_width) // 2
        y = (height + text_height) // 2
    else:  # bottom
        x = (width - text_width) // 2
        y = height - 30
    
    # Dessiner un fond avec bordure
    padding = 15
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + padding),
        config.ALERT_COLOR,
        -1
    )
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + padding),
        config.TEXT_COLOR,
        3
    )
    
    # Dessiner le texte
    cv2.putText(
        frame,
        message,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        config.TEXT_COLOR,
        thickness,
        cv2.LINE_AA
    )


def draw_stats(
    frame: np.ndarray,
    fps: float,
    num_objects: int,
    source_type: str = "Unknown",
    alert_count: int = 0
) -> None:
    """
    Affiche un panneau de statistiques moderne sur la frame.
    
    Args:
        frame: Frame sur laquelle dessiner
        fps: FPS actuel du système
        num_objects: Nombre d'objets actuellement trackés
        source_type: Type de source vidéo ('webcam' ou 'file')
        alert_count: Nombre total d'alertes déclenchées
    """
    height, width = frame.shape[:2]
    
    # Dimensions du panneau
    panel_width = 280
    panel_height = 160
    panel_x = 15
    panel_y = 15
    
    # Créer un fond semi-transparent avec dégradé
    overlay = frame.copy()
    
    # Fond principal avec bordure
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (30, 30, 30),
        -1
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Bordure avec accent coloré
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (100, 200, 255),
        2
    )
    
    # Barre supérieure (header)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + 35),
        (100, 200, 255),
        -1
    )
    
    # Titre du panneau
    cv2.putText(
        frame,
        "SURVEILLANCE SYSTEM",
        (panel_x + 15, panel_y + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    
    # Statistiques
    stats = [
        ("FPS", f"{fps:.1f}", (0, 255, 0) if fps > 15 else (0, 165, 255)),
        ("Objects", str(num_objects), (255, 200, 0)),
        ("Alerts", str(alert_count), (0, 0, 255) if alert_count > 0 else (200, 200, 200)),
        ("Source", source_type.upper(), (200, 200, 200))
    ]
    
    start_y = panel_y + 55
    line_spacing = 26
    
    for i, (label, value, color) in enumerate(stats):
        y_pos = start_y + (i * line_spacing)
        
        # Label
        cv2.putText(
            frame,
            f"{label}:",
            (panel_x + 20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )
        
        # Valeur (en couleur)
        cv2.putText(
            frame,
            value,
            (panel_x + 140, y_pos),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )


def draw_controls_info(frame: np.ndarray) -> None:
    """
    Affiche les informations sur les contrôles clavier en bas de la frame.
    
    Args:
        frame: Frame sur laquelle dessiner
    """
    height, width = frame.shape[:2]
    
    # Créer un panneau de contrôles moderne
    panel_height = 45
    panel_y = height - panel_height
    
    # Fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, panel_y),
        (width, height),
        (20, 20, 20),
        -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Ligne supérieure décorative
    cv2.line(frame, (0, panel_y), (width, panel_y), (100, 200, 255), 2)
    
    # Contrôles
    controls = [
        ("Q/ESC", "Quitter"),
        ("R", "Reset Alertes")
    ]
    
    total_width = sum([100 for _ in controls])
    start_x = (width - total_width) // 2
    
    for i, (key, action) in enumerate(controls):
        x = start_x + (i * 180)
        y = panel_y + 28
        
        # Touche
        cv2.putText(
            frame,
            key,
            (x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (100, 200, 255),
            2,
            cv2.LINE_AA
        )
        
        # Action
        cv2.putText(
            frame,
            action,
            (x + 70, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )


def draw_alert_history(
    frame: np.ndarray,
    recent_alerts: list,
    max_display: int = 5
) -> None:
    """
    Affiche l'historique récent des alertes dans un panneau latéral.
    
    Args:
        frame: Frame sur laquelle dessiner
        recent_alerts: Liste des alertes récentes (dicts avec timestamp, id, classe)
        max_display: Nombre maximum d'alertes à afficher
    """
    if not recent_alerts:
        return
    
    height, width = frame.shape[:2]
    
    # Dimensions du panneau
    panel_width = 250
    panel_height = min(200, 50 + len(recent_alerts[:max_display]) * 35)
    panel_x = width - panel_width - 15
    panel_y = 15
    
    # Fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (30, 30, 30),
        -1
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Bordure
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (0, 0, 255),
        2
    )
    
    # Header
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + 35),
        (0, 0, 255),
        -1
    )
    
    cv2.putText(
        frame,
        "ALERT HISTORY",
        (panel_x + 15, panel_y + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    # Alertes récentes
    for i, alert in enumerate(recent_alerts[:max_display]):
        y = panel_y + 55 + (i * 35)
        
        # Temps écoulé
        time_str = alert.get('time_ago', '---')
        
        # Icône d'alerte
        cv2.circle(frame, (panel_x + 20, y - 5), 5, (0, 0, 255), -1)
        
        # Texte
        text = f"{time_str} - ID{alert['track_id']} ({alert['class_name']})"
        cv2.putText(
            frame,
            text,
            (panel_x + 35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )


# ============================================================================
# VISUALISATIONS AVANCÉES
# ============================================================================

def draw_trajectory(frame: np.ndarray, trajectory: list, color: Tuple[int, int, int], thickness: int = 2) -> None:
    """Dessine la trajectoire d'un objet avec gradient d'opacité."""
    if len(trajectory) < 2:
        return
    
    for i in range(1, len(trajectory)):
        alpha = i / len(trajectory)
        current_thickness = max(1, int(thickness * alpha))
        current_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
        cv2.line(frame, trajectory[i-1], trajectory[i], current_color, current_thickness, cv2.LINE_AA)
    
    if trajectory:
        cv2.circle(frame, trajectory[-1], 4, color, -1)
        cv2.circle(frame, trajectory[-1], 5, (255, 255, 255), 1)


def draw_all_trajectories(frame: np.ndarray, trajectories: dict, colors: dict, class_map: dict, thickness: int = 2) -> None:
    """Dessine toutes les trajectoires sur la frame."""
    for track_id, trajectory in trajectories.items():
        class_name = class_map.get(track_id, "unknown")
        color = colors.get(class_name, (255, 255, 255))
        draw_trajectory(frame, trajectory, color, thickness)


class HeatmapGenerator:
    """Génère une heatmap des zones d'activité."""
    
    def __init__(self, width: int, height: int, decay_factor: float = 0.95, colormap: int = cv2.COLORMAP_JET):
        self.width = width
        self.height = height
        self.decay_factor = decay_factor
        self.colormap = colormap
        self.heatmap = np.zeros((height, width), dtype=np.float32)
    
    def update(self, positions: list, intensity: float = 10.0):
        """Met à jour la heatmap avec de nouvelles positions."""
        self.heatmap *= self.decay_factor
        for x, y in positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                radius = 30
                y1, y2 = max(0, y - radius), min(self.height, y + radius)
                x1, x2 = max(0, x - radius), min(self.width, x + radius)
                yy, xx = np.ogrid[y1:y2, x1:x2]
                gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * (radius/3)**2))
                self.heatmap[y1:y2, x1:x2] += gaussian * intensity
    
    def get_overlay(self, alpha: float = 0.5) -> np.ndarray:
        """Génère l'overlay de heatmap colorée."""
        normalized = np.clip(self.heatmap, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, self.colormap)
    
    def reset(self):
        """Réinitialise la heatmap."""
        self.heatmap.fill(0)


class FPSGraph:
    """Affiche un graphique du FPS en temps réel."""
    
    def __init__(self, width: int = 200, height: int = 60, max_history: int = 100):
        self.width, self.height, self.max_history = width, height, max_history
        self.fps_history = []
    
    def update(self, fps: float):
        """Met à jour l'historique FPS."""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history:
            self.fps_history.pop(0)
    
    def draw(self, frame: np.ndarray, position: Tuple[int, int], target_fps: float = 30.0) -> None:
        """Dessine le graphique sur la frame."""
        if not self.fps_history:
            return
        x, y = position
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + self.width, y + self.height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x + self.width, y + self.height), (100, 200, 255), 1)
        cv2.putText(frame, "FPS", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        max_fps = max(max(self.fps_history), target_fps) * 1.1
        target_y = int(y + self.height - (target_fps / max_fps) * (self.height - 20))
        cv2.line(frame, (x, target_y), (x + self.width, target_y), (0, 255, 0), 1, cv2.LINE_AA)
        
        points = [(x + int((i / len(self.fps_history)) * self.width), int(y + self.height - (fps_val / max_fps) * (self.height - 20))) for i, fps_val in enumerate(self.fps_history)]
        
        if len(points) > 1:
            for i in range(1, len(points)):
                fps_val = self.fps_history[i]
                color = (0, 255, 0) if fps_val >= target_fps else (0, 200, 255) if fps_val >= target_fps * 0.7 else (0, 0, 255)
                cv2.line(frame, points[i-1], points[i], color, 2, cv2.LINE_AA)
        
        if self.fps_history:
            cv2.putText(frame, f"{self.fps_history[-1]:.1f}", (x + self.width - 35, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

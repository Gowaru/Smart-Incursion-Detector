"""
Module utilitaires pour le système de détection et tracking d'objets.
Contient les fonctions de gestion vidéo et de visualisation.
"""

from .video_source import VideoSourceManager
from .visualization import (
    draw_bounding_box,
    draw_alert_zone,
    draw_alert_message,
    draw_stats,
    draw_controls_info,
    draw_alert_history
)

__all__ = [
    'VideoSourceManager',
    'draw_bounding_box',
    'draw_alert_zone',
    'draw_alert_message',
    'draw_stats',
    'draw_controls_info',
    'draw_alert_history'
]

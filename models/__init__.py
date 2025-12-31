"""
Module de classes métier pour le système de détection et tracking d'objets.
Contient les classes AlertZone et ObjectTracker.
"""

from .alert_zone import AlertZone
from .object_tracker import ObjectTracker

__all__ = [
    'AlertZone',
    'ObjectTracker'
]

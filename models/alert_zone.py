"""
Classe de gestion de la zone d'alerte.
Détecte les intrusions d'objets dans une zone définie et gère les alertes.
"""

from typing import Tuple, Set
from datetime import datetime


class AlertZone:
    """
    Gère une zone d'alerte rectangulaire pour détecter les intrusions d'objets.
    
    La zone peut être définie en coordonnées relatives (0.0 à 1.0) qui seront
    converties en coordonnées absolues selon la taille de la frame.
    """
    
    def __init__(
        self,
        zone_coords: Tuple[float, float, float, float],
        frame_width: int,
        frame_height: int
    ):
        """
        Initialise la zone d'alerte.
        
        Args:
            zone_coords: Coordonnées de la zone (x1, y1, x2, y2)
                        Peut être en relatif (0.0-1.0) ou absolu
            frame_width: Largeur de la frame en pixels
            frame_height: Hauteur de la frame en pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Convertir les coordonnées en absolues si elles sont relatives
        x1, y1, x2, y2 = zone_coords
        
        # Si toutes les coordonnées sont entre 0 et 1, c'est relatif
        if all(0 <= coord <= 1 for coord in [x1, y1, x2, y2]):
            self.x1 = int(x1 * frame_width)
            self.y1 = int(y1 * frame_height)
            self.x2 = int(x2 * frame_width)
            self.y2 = int(y2 * frame_height)
        else:
            # Sinon, c'est déjà en absolu
            self.x1 = int(x1)
            self.y1 = int(y1)
            self.x2 = int(x2)
            self.y2 = int(y2)
        
        # Ensemble des IDs d'objets ayant déjà déclenché une alerte
        # Pour éviter les alertes répétées pour le même objet
        self.alerted_objects: Set[int] = set()
        
        # État d'alerte actif
        self.is_alert_active = False
        
        # Historique des alertes
        self.alert_history = []
    
    def is_object_in_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Vérifie si une bounding box intersecte avec la zone d'alerte.
        
        Args:
            bbox: Coordonnées de la bounding box (x1, y1, x2, y2)
        
        Returns:
            bool: True si l'objet est dans la zone
        """
        obj_x1, obj_y1, obj_x2, obj_y2 = bbox
        
        # Vérifier l'intersection rectangulaire
        # Deux rectangles s'intersectent s'ils ne sont PAS complètement séparés
        horizontal_overlap = not (obj_x2 < self.x1 or obj_x1 > self.x2)
        vertical_overlap = not (obj_y2 < self.y1 or obj_y1 > self.y2)
        
        return horizontal_overlap and vertical_overlap
    
    def check_and_alert(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        class_name: str
    ) -> bool:
        """
        Vérifie si un objet entre dans la zone et déclenche une alerte si nécessaire.
        
        Args:
            track_id: ID unique de l'objet tracké
            bbox: Coordonnées de la bounding box
            class_name: Classe de l'objet
        
        Returns:
            bool: True si une nouvelle alerte a été déclenchée
        """
        # Vérifier si l'objet est dans la zone
        if not self.is_object_in_zone(bbox):
            return False
        
        # Si cet objet a déjà déclenché une alerte, ne pas répéter
        if track_id in self.alerted_objects:
            return False
        
        # Nouvelle intrusion détectée !
        self.alerted_objects.add(track_id)
        self.is_alert_active = True
        
        # Enregistrer l'alerte
        alert_info = {
            'timestamp': datetime.now(),
            'track_id': track_id,
            'class_name': class_name,
            'bbox': bbox
        }
        self.alert_history.append(alert_info)
        
        # Afficher dans la console
        timestamp_str = alert_info['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*70}")
        print(f"🚨 ALERTE INTRUSION!")
        print(f"{'='*70}")
        print(f"Timestamp    : {timestamp_str}")
        print(f"Objet ID     : {track_id}")
        print(f"Classe       : {class_name}")
        print(f"Position     : x={bbox[0]}, y={bbox[1]}")
        print(f"Total alertes: {len(self.alert_history)}")
        print(f"{'='*70}\n")
        
        return True
    
    def reset_alerts(self) -> None:
        """
        Réinitialise l'état des alertes.
        Permet aux objets de déclencher à nouveau des alertes.
        """
        self.alerted_objects.clear()
        self.is_alert_active = False
        print("\n✅ Alertes réinitialisées. Les objets peuvent à nouveau déclencher des alertes.\n")
    
    def get_zone_coords(self) -> Tuple[int, int, int, int]:
        """
        Retourne les coordonnées absolues de la zone d'alerte.
        
        Returns:
            Tuple[int, int, int, int]: Coordonnées (x1, y1, x2, y2)
        """
        return (self.x1, self.y1, self.x2, self.y2)
    
    def get_alert_count(self) -> int:
        """
        Retourne le nombre total d'alertes déclenchées.
        
        Returns:
            int: Nombre d'alertes
        """
        return len(self.alert_history)
    
    def get_alert_history(self) -> list:
        """
        Retourne l'historique complet des alertes.
        
        Returns:
            list: Liste des alertes avec leurs informations
        """
        return self.alert_history.copy()
    
    def update_zone_size(self, frame_width: int, frame_height: int) -> None:
        """
        Met à jour la taille de la zone si la résolution de la frame change.
        
        Args:
            frame_width: Nouvelle largeur de la frame
            frame_height: Nouvelle hauteur de la frame
        """
        # Calculer les ratios relatifs actuels
        rel_x1 = self.x1 / self.frame_width if self.frame_width > 0 else 0
        rel_y1 = self.y1 / self.frame_height if self.frame_height > 0 else 0
        rel_x2 = self.x2 / self.frame_width if self.frame_width > 0 else 1
        rel_y2 = self.y2 / self.frame_height if self.frame_height > 0 else 1
        
        # Mettre à jour avec les nouvelles dimensions
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.x1 = int(rel_x1 * frame_width)
        self.y1 = int(rel_y1 * frame_height)
        self.x2 = int(rel_x2 * frame_width)
        self.y2 = int(rel_y2 * frame_height)

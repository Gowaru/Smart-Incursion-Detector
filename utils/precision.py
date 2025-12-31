"""
Utilitaires pour améliorer la précision du tracking.
Filtrage Kalman, validation temporelle, et smoothing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


class KalmanFilter:
    """
    Filtre de Kalman pour smooth le tracking des bounding boxes.
    Prédit et corrige la position des objets pour un tracking plus fluide.
    """
    
    def __init__(self):
        """Initialise le filtre de Kalman."""
        # État: [x, y, width, height, vx, vy, vw, vh]
        # x, y = centre de la box
        # width, height = dimensions
        # vx, vy = vitesse
        # vw, vh = vitesse de changement de taille
        self.state = np.zeros((8, 1))
        
        # Matrice de transition (modèle de mouvement constant)
        self.F = np.eye(8)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh
        
        # Matrice d'observation (on observe x, y, w, h)
        self.H = np.eye(4, 8)
        
        # Covariance de l'état
        self.P = np.eye(8) * 1000
        
        # Bruit de processus
        self.Q = np.eye(8)
        self.Q[0:4, 0:4] *= 0.01  # Position
        self.Q[4:8, 4:8] *= 0.1   # Vitesse
        
        # Bruit de mesure
        self.R = np.eye(4) * 10
        
        self.initialized = False
    
    def predict(self) -> np.ndarray:
        """
        Prédit l'état suivant.
        
        Returns:
            État prédit [x, y, w, h]
        """
        # Prédiction de l'état
        self.state = self.F @ self.state
        # Prédiction de la covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[0:4].flatten()
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Met à jour l'état avec une nouvelle mesure.
        
        Args:
            measurement: Mesure [x, y, w, h]
        
        Returns:
            État corrigé [x, y, w, h]
        """
        measurement = np.array(measurement).reshape(4, 1)
        
        if not self.initialized:
            # Première mesure: initialiser l'état
            self.state[0:4] = measurement
            self.initialized = True
            return measurement.flatten()
        
        # Innovation (différence mesure - prédiction)
        y = measurement - self.H @ self.state
        
        # Covariance de l'innovation
        S = self.H @ self.P @ self.H.T + self.R
        
        # Gain de Kalman
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Mise à jour de l'état
        self.state = self.state + K @ y
        
        # Mise à jour de la covariance
        self.P = (np.eye(8) - K @ self.H) @ self.P
        
        return self.state[0:4].flatten()
    
    def bbox_to_measurement(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Convertit une bounding box (x1, y1, x2, y2) en mesure (cx, cy, w, h).
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Mesure [cx, cy, w, h]
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h])
    
    def measurement_to_bbox(self, measurement: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Convertit une mesure (cx, cy, w, h) en bounding box (x1, y1, x2, y2).
        
        Args:
            measurement: Mesure [cx, cy, w, h]
        
        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        cx, cy, w, h = measurement
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)


class TrackingEnhancer:
    """
    Améliore le tracking avec filtrage Kalman et validation temporelle.
    """
    
    def __init__(self, min_consecutive_frames: int = 2):
        """
        Initialise l'améliorateur de tracking.
        
        Args:
            min_consecutive_frames: Nombre min de frames pour valider une détection
        """
        self.filters: Dict[int, KalmanFilter] = {}
        self.min_frames = min_consecutive_frames
        self.detection_counts: Dict[int, int] = {}
        self.last_seen: Dict[int,float] = {}
        self.validated_tracks: set = set()
    
    def update_track(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        confidence: float
    ) -> Tuple[Tuple[int, int, int, int], bool]:
        """
        Met à jour un track avec filtrage Kalman.
        
        Args:
            track_id: ID du track
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Confiance de la détection
        
        Returns:
            Tuple (bbox_lissée, est_validé)
        """
        current_time = time.time()
        
        # Créer le filtre si nouveau track
        if track_id not in self.filters:
            self.filters[track_id] = KalmanFilter()
            self.detection_counts[track_id] = 0
        
        # Convertir bbox en mesure
        kf = self.filters[track_id]
        measurement = kf.bbox_to_measurement(bbox)
        
        # Prédire puis mettre à jour
        kf.predict()
        smoothed = kf.update(measurement)
        smoothed_bbox = kf.measurement_to_bbox(smoothed)
        
        # Validation temporelle
        self.detection_counts[track_id] += 1
        self.last_seen[track_id] = current_time
        
        is_validated = self.detection_counts[track_id] >= self.min_frames
        if is_validated:
            self.validated_tracks.add(track_id)
        
        return smoothed_bbox, is_validated
    
    def predict_missing(self, track_id: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Prédit la position d'un track manquant.
        
        Args:
            track_id: ID du track
        
        Returns:
            Bounding box prédite ou None
        """
        if track_id not in self.filters:
            return None
        
        kf = self.filters[track_id]
        predicted = kf.predict()
        return kf.measurement_to_bbox(predicted)
    
    def cleanup_old_tracks(self, max_age: float = 2.0):
        """
        Nettoie les tracks trop anciens.
        
        Args:
            max_age: Âge maximum en secondes
        """
        current_time = time.time()
        to_remove = []
        
        for track_id, last_time in self.last_seen.items():
            if current_time - last_time > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            self.filters.pop(track_id, None)
            self.detection_counts.pop(track_id, None)
            self.last_seen.pop(track_id, None)
            self.validated_tracks.discard(track_id)
    
    def is_validated(self, track_id: int) -> bool:
        """Vérifie si un track est validé."""
        return track_id in self.validated_tracks


class TrajectoryTracker:
    """
    Garde en mémoire les trajectoires des objets trackés.
    """
    
    def __init__(self, max_length: int = 30):
        """
        Initialise le tracker de trajectoires.
        
        Args:
            max_length: Longueur maximale des trajectoires
        """
        self.max_length = max_length
        self.trajectories: Dict[int, deque] = {}
    
    def update(self, track_id: int, bbox: Tuple[int, int, int, int]):
        """
        Met à jour la trajectoire d'un objet.
        
        Args:
            track_id: ID du track
            bbox: Bounding box (x1, y1, x2, y2)
        """
        # Calculer le centre de la box
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Créer la trajectoire si nouvelle
        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=self.max_length)
        
        # Ajouter le point
        self.trajectories[track_id].append(center)
    
    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """
        Récupère la trajectoire d'un objet.
        
        Args:
            track_id: ID du track
        
        Returns:
            Liste de points (x, y)
        """
        if track_id not in self.trajectories:
            return []
        return list(self.trajectories[track_id])
    
    def cleanup(self, active_tracks: set):
        """
        Nettoie les trajectoires des tracks inactifs.
        
        Args:
            active_tracks: Set des IDs actifs
        """
        to_remove = [tid for tid in self.trajectories if tid not in active_tracks]
        for tid in to_remove:
            del self.trajectories[tid]
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """Retourne toutes les trajectoires."""
        return {tid: list(traj) for tid, traj in self.trajectories.items()}

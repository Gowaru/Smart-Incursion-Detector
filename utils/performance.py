"""
Utilitaires pour l'optimisation des performances.
Threading, caching, et optimisations diverses.
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple


class ThreadedVideoCapture:
    """
    Wrapper pour cv2.VideoCapture avec threading pour améliorer les performances.
    Lit les frames dans un thread séparé pour éviter les blocages.
    """
    
    def __init__(self, source, buffer_size: int = 2):
        """
        Initialise la capture vidéo threadée.
        
        Args:
            source: Source vidéo (index webcam ou chemin fichier)
            buffer_size: Taille du buffer de frames
        """
        self.cap = source if isinstance(source, cv2.VideoCapture) else cv2.VideoCapture(source)
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        
        # Statistiques
        self.frames_read = 0
        self.frames_dropped = 0
        
        # Démarrer le thread de lecture
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        """Thread de lecture des frames."""
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                
                if not ret:
                    self.stopped = True
                    break
                
                try:
                    # Ajouter la frame au buffer
                    self.frame_queue.put((ret, frame), block=False)
                    self.frames_read += 1
                except queue.Full:
                    # Buffer plein, drop la frame
                    self.frames_dropped += 1
            else:
                # Buffer plein, attendre un peu
                time.sleep(0.001)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lit la prochaine frame du buffer.
        
        Returns:
            Tuple (success, frame)
        """
        if self.stopped and self.frame_queue.empty():
            return False, None
        
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, None
    
    def get(self, prop_id: int):
        """Wrapper pour get() de VideoCapture."""
        return self.cap.get(prop_id)
    
    def set(self, prop_id: int, value):
        """Wrapper pour set() de VideoCapture."""
        return self.cap.set(prop_id, value)
    
    def isOpened(self) -> bool:
        """Vérifie si la capture est ouverte."""
        return self.cap.isOpened() and not self.stopped
    
    def release(self):
        """Arrête le thread et libère les ressources."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.cap.release()
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de performance."""
        return {
            'frames_read': self.frames_read,
            'frames_dropped': self.frames_dropped,
            'drop_rate': self.frames_dropped / max(self.frames_read, 1)
        }


class FrameProcessor:
    """
    Optimise le traitement des frames avec diverses techniques.
    """
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialise le processeur de frames.
        
        Args:
            target_size: Taille cible (width, height) ou None pour garder originale
        """
        self.target_size = target_size
        self.last_frame = None
        self.frame_count = 0
    
    def process(self, frame: np.ndarray, skip_factor: int = 1) -> Optional[np.ndarray]:
        """
        Traite une frame avec optimisations.
        
        Args:
            frame: Frame à traiter
            skip_factor: Facteur de skip (1 = tout traiter, 2 = une sur deux, etc.)
        
        Returns:
            Frame traitée ou None si skippée
        """
        self.frame_count += 1
        
        # Skip frames si nécessaire
        if skip_factor > 1 and self.frame_count % skip_factor != 0:
            return None
        
        # Redimensionner si nécesssaire
        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        self.last_frame = frame
        return frame
    
    def get_last_frame(self) -> Optional[np.ndarray]:
        """Retourne la dernière frame traitée."""
        return self.last_frame


class FPSCounter:
    """
    Calcule le FPS avec lissage exponentiel pour affichage stable.
    """
    
    def __init__(self, smoothing: float = 0.9):
        """
        Initialise le compteur FPS.
        
        Args:
            smoothing: Facteur de lissage (0-1, plus haut = plus lisse)
        """
        self.smoothing = smoothing
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.fps_history = []
    
    def update(self) -> float:
        """
        Met à jour et retourne le FPS actuel.
        
        Returns:
            FPS lissé
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            
            # Lissage exponentiel
            if self.fps == 0:
                self.fps = instant_fps
            else:
                self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
            
            # Historique
            self.fps_history.append(self.fps)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
        
        self.last_time = current_time
        self.frame_count += 1
        
        return self.fps
    
    def get_average_fps(self) -> float:
        """Retourne le FPS moyen."""
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_history(self) -> list:
        """Retourne l'historique des FPS."""
        return self.fps_history.copy()


class PerformanceMonitor:
    """
    Moniteur de performance pour suivre les métriques du système.
    """
    
    def __init__(self):
        """Initialise le moniteur."""
        self.metrics = {
            'fps': [],
            'processing_time': [],
            'detection_time': [],
            'visualization_time': []
        }
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Démarre un timer."""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Arrête un timer et enregistre la durée.
        
        Returns:
            Durée écoulée en secondes
        """
        if name not in self.start_times:
            return 0
        
        elapsed = time.time() - self.start_times[name]
        
        if name in self.metrics:
            self.metrics[name].append(elapsed)
            # Garder seulement les 100 dernières mesures
            if len(self.metrics[name]) > 100:
                self.metrics[name].pop(0)
        
        return elapsed
    
    def get_average(self, metric: str) -> float:
        """Retourne la moyenne d'une métrique."""
        if metric not in self.metrics or not self.metrics[metric]:
            return 0
        return sum(self.metrics[metric]) / len(self.metrics[metric])
    
    def get_stats(self) -> dict:
        """Retourne toutes les statistiques."""
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'current': values[-1] if values else 0
                }
        return stats

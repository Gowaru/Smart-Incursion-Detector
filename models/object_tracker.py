"""
Classe principale du système de tracking d'objets.
Coordonne la détection YOLO, le tracking multi-objets et les alertes.
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from typing import Optional

# Import des autres modules du projet
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.alert_zone import AlertZone
from utils.visualization import (
    draw_bounding_box,
    draw_alert_zone,
    draw_alert_message,
    draw_stats,
    draw_controls_info,
    draw_alert_history,
    draw_all_trajectories,
    HeatmapGenerator,
    FPSGraph
)
from utils.performance import (
    ThreadedVideoCapture,
    FrameProcessor,
    FPSCounter,
    PerformanceMonitor
)
from utils.precision import (
    TrackingEnhancer,
    TrajectoryTracker
)


class ObjectTracker:
    """
    Système principal de détection et tracking d'objets en temps réel OPTIMISÉ.
    
    Pipeline avec optimisations:
    1. Capture de frames (threadée si activé)
    2. Détection d'objets avec YOLOv8
    3. Tracking multi-objets avec IDs persistants + Kalman filter
    4. Vérification des intrusions dans la zone d'alerte
    5. Affichage avec visualisation avancée (trajectoires, heatmap, graphs)
    
    Optimisations:
    - Threading pour capture vidéo
    - Filtrage Kalman pour smooth tracking
    - Trajectoires avec gradients
    - Heatmap d'activité
    - Graphique FPS temps réel
    - Monitoring de performance
    """
    
    def __init__(
        self,
        video_source: cv2.VideoCapture,
        source_type: str = "unknown",
        model_name: str = None
    ):
        """
        Initialise le système de tracking.
        
        Args:
            video_source: Source vidéo (VideoCapture déjà initialisé)
            source_type: Type de source ('webcam' ou 'file')
            model_name: Nom du modèle YOLO à utiliser (ou None pour config par défaut)
        """
        self.cap = video_source
        self.source_type = source_type
        
        # OPTIMISATION: Capture vidéo threadée si activé
        if config.ENABLE_THREADING and isinstance(video_source, cv2.VideoCapture):
            print("🚀 Threading activé")
            self.cap = ThreadedVideoCapture(video_source, config.FRAME_BUFFER_SIZE)
        else:
            self.cap = video_source
        
        # Initialiser le modèle YOLO
        model_path = model_name or config.MODEL_NAME
        print(f"\n🤖 Chargement du modèle YOLO: {model_path}...")
        self.model = YOLO(model_path)
        
        # OPTIMISATION: Half precision si disponible
        if config.USE_HALF_PRECISION:
            try:
                self.model.to('cuda').half()
                print("✅ Half-precision (FP16) activé")
            except:
                pass
        print(f"✅ Modèle chargé avec succès\n")
        
        # Obtenir les dimensions de la vidéo
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialiser la zone d'alerte
        self.alert_zone = AlertZone(
            config.ALERT_ZONE,
            self.frame_width,
            self.frame_height
        )
        
        # OPTIMISATION: Compteur FPS amélioré
        self.fps_counter = FPSCounter(smoothing=0.9)
        self.fps = 0
        
        # OPTIMISATION: Frame processor
        processing_size = None
        if config.PROCESSING_WIDTH and config.PROCESSING_HEIGHT:
            processing_size = (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
        self.frame_processor = FrameProcessor(processing_size)
        
        # OPTIMISATION: Monitoring de performance
        self.perf_monitor = PerformanceMonitor()
        
        # OPTIMISATION PRÉCISION: Kalman filter
        if config.ENABLE_KALMAN_FILTER:
            print("🎯 Filtrage Kalman activé")
            self.tracking_enhancer = TrackingEnhancer(config.MIN_DETECTION_FRAMES)
        else:
            self.tracking_enhancer = None
        
        # OPTIMISATION VIZ: Trajectoires
        if config.SHOW_TRAJECTORIES:
            print("📍 Trajectoires activées")
            self.trajectory_tracker = TrajectoryTracker(config.TRAJECTORY_LENGTH)
        else:
            self.trajectory_tracker = None
        
        # OPTIMISATION VIZ: Heatmap
        if config.SHOW_HEATMAP:
            print("🔥 Heatmap activée")
            self.heatmap = HeatmapGenerator(self.frame_width, self.frame_height)
        else:
            self.heatmap = None
        
        # OPTIMISATION VIZ: Graphique FPS
        if config.SHOW_FPS_GRAPH:
            print("📊 Graphique FPS activé")
            self.fps_graph = FPSGraph(max_history=config.FPS_HISTORY_LENGTH)
        else:
            self.fps_graph = None
        
        # Mapping des IDs de classes COCO vers les noms
        self.class_names = config.CLASS_IDS
        
        # IDs des classes cibles pour le filtrage
        self.target_class_ids = [
            config.CLASS_NAMES_TO_IDS.get(name, -1)
            for name in config.TARGET_CLASSES
        ]
        
        print(f"📊 Configuration:")
        print(f"   Résolution: {self.frame_width}x{self.frame_height}")
        print(f"   FPS source: {self.original_fps:.2f}")
        print(f"   Classes cibles: {', '.join(config.TARGET_CLASSES)}")
        print(f"   Zone d'alerte: {self.alert_zone.get_zone_coords()}")
        print()
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Redimensionne la frame si nécessaire selon la configuration.
        
        Args:
            frame: Frame originale
        
        Returns:
            np.ndarray: Frame redimensionnée
        """
        if not config.FRAME_RESIZE:
            return frame
        
        if (self.frame_width != config.DISPLAY_WIDTH or 
            self.frame_height != config.DISPLAY_HEIGHT):
            # Mettre à jour les dimensions
            self.frame_width = config.DISPLAY_WIDTH
            self.frame_height = config.DISPLAY_HEIGHT
            
            # Mettre à jour la zone d'alerte
            self.alert_zone.update_zone_size(self.frame_width, self.frame_height)
            
            return cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        
        return frame
    
    def _process_detections(self, results, scale_x: float = 1.0, scale_y: float = 1.0) -> tuple:
        """Traite les résultats de détection avec optimisations et mise à l'échelle."""
        tracked_objects = []
        is_alert_active = False
        active_tracks = set()
        positions = []  # Pour heatmap
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return tracked_objects, is_alert_active
        
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            
            # Récupérer et mettre à l'échelle la bbox
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            bbox = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
            
            if cls_id not in self.target_class_ids:
                continue
            
            # Seuil adaptatif
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            threshold = config.ADAPTIVE_CONFIDENCE.get(class_name, config.CONFIDENCE_THRESHOLD)
            
            if conf < threshold:
                continue
            
            track_id = int(boxes.id[i]) if hasattr(boxes, 'id') and boxes.id is not None else i
            active_tracks.add(track_id)
            
            # PRÉCISION: Filtrage Kalman
            if self.tracking_enhancer:
                smoothed_bbox, is_validated = self.tracking_enhancer.update_track(track_id, bbox, conf)
                if not is_validated:
                    continue
                bbox = smoothed_bbox
            
            # VISUALISATION: Trajectoires
            if self.trajectory_tracker:
                self.trajectory_tracker.update(track_id, bbox)
            
            # VISUALISATION: Heatmap
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            positions.append(center)
            
            obj = {
                'track_id': track_id,
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': conf,
                'bbox': bbox
            }
            tracked_objects.append(obj)
            
            # Alertes
            if self.alert_zone.check_and_alert(track_id, bbox, class_name):
                is_alert_active = True
        
        # Mettre à jour heatmap
        if self.heatmap and positions:
            self.heatmap.update(positions)
        
        # Cleanup
        if self.tracking_enhancer:
            self.tracking_enhancer.cleanup_old_tracks()
        if self.trajectory_tracker:
            self.trajectory_tracker.cleanup(active_tracks)
        
        return tracked_objects, is_alert_active

    def _draw_frame(self, frame: np.ndarray, tracked_objects: list, is_alert_active: bool) -> None:
        """Dessine tous les éléments visuels sur la frame."""
        # Heatmap
        if self.heatmap:
            heatmap_overlay = self.heatmap.get_overlay(config.HEATMAP_ALPHA)
            if heatmap_overlay.shape[:2] != frame.shape[:2]:
                heatmap_overlay = cv2.resize(heatmap_overlay, (frame.shape[1], frame.shape[0]))
            cv2.addWeighted(frame, 0.7, heatmap_overlay, 0.3, 0, frame)
        
        # Zone d'alerte
        draw_alert_zone(frame, self.alert_zone.get_zone_coords(), is_alert_active or self.alert_zone.is_alert_active)
        
        # Trajectoires
        if self.trajectory_tracker:
            trajectories = self.trajectory_tracker.get_all_trajectories()
            # On recrée un map temporaire pour les couleurs
            track_classes = {obj['track_id']: obj['class_name'] for obj in tracked_objects}
            draw_all_trajectories(frame, trajectories, config.COLORS, track_classes, config.TRAJECTORY_THICKNESS)
        
        # Bounding boxes
        for obj in tracked_objects:
            color = config.COLORS.get(obj['class_name'], (255, 255, 255))
            draw_bounding_box(frame, obj['bbox'], obj['track_id'], obj['class_name'], color, obj['confidence'])
        
        # Alerte message
        if is_alert_active or self.alert_zone.is_alert_active:
            draw_alert_message(frame, "ALERTE INTRUSION!", position='top')
        
        # Historique alertes
        alert_history = self.alert_zone.get_alert_history()
        if alert_history:
            recent_alerts = []
            for alert in alert_history[-5:]:
                elapsed = time.time() - alert['timestamp'].timestamp()
                time_ago = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed//60)}m"
                recent_alerts.append({'time_ago': time_ago, 'track_id': alert['track_id'], 'class_name': alert['class_name']})
            if recent_alerts:
                draw_alert_history(frame, recent_alerts)
        
        # Statistiques
        draw_stats(frame, self.fps, len(tracked_objects), self.source_type, self.alert_zone.get_alert_count())
        
        # Graphique FPS
        if self.fps_graph:
            self.fps_graph.update(self.fps)
            self.fps_graph.draw(frame, (15, frame.shape[0] - 80), target_fps=self.original_fps)
        
        # Contrôles
        draw_controls_info(frame)

    def run(self) -> None:
        """Lance la boucle principale."""
        print("="*70)
        print("🚀 SYSTÈME OPTIMISÉ DÉMARRÉ")
        print("="*70)
        print("\n📹 Traitement en cours...")
        print("   'Q' / 'ESC' : Quitter")
        print("   'R' : Réinitialiser alertes")
        print("   'H' : Toggle heatmap")
        print("   'T' : Toggle trajectoires\n")
        
        # Variables pour persister l'état entre les frames skippées
        last_tracked_objects = []
        last_is_alert_active = False
        
        try:
            while True:
                self.perf_monitor.start_timer('total')
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.source_type == "file":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Frame processing (skip, resize)
                processed_frame = self.frame_processor.process(frame, config.FRAME_SKIP)
                
                # Si on doit traiter cette frame (pas skippée)
                if processed_frame is not None:
                    # Calcul des échelles
                    ph, pw = processed_frame.shape[:2]
                    h, w = frame.shape[:2]
                    scale_x = w / pw
                    scale_y = h / ph
                    
                    # Détection
                    self.perf_monitor.start_timer('detection')
                    results = self.model.track(
                        processed_frame,
                        conf=config.CONFIDENCE_THRESHOLD,
                        iou=config.IOU_THRESHOLD,
                        persist=True,
                        tracker='bytetrack.yaml',
                        verbose=False,
                        device=config.DEVICE
                    )
                    self.perf_monitor.stop_timer('detection')
                    
                    # Traitement
                    self.perf_monitor.start_timer('tracking')
                    last_tracked_objects, last_is_alert_active = self._process_detections(results, scale_x, scale_y)
                    self.perf_monitor.stop_timer('tracking')
                
                # Visualization (TOUJOURS dessiner, même si detection skippée)
                display_frame = frame.copy()
                self.perf_monitor.start_timer('visualization')
                self._draw_frame(display_frame, last_tracked_objects, last_is_alert_active)
                self.perf_monitor.stop_timer('visualization')
                
                # FPS update
                self.fps = self.fps_counter.update()
                
                cv2.imshow('Système de Détection et Tracking - OPTIMISÉ', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    break
                elif key in [ord('r'), ord('R')]:
                    self.alert_zone.reset_alerts()
                elif key in [ord('h'), ord('H')]:
                    config.SHOW_HEATMAP = not config.SHOW_HEATMAP
                    print(f"🔥 Heatmap: {'ON' if config.SHOW_HEATMAP else 'OFF'}")
                elif key in [ord('t'), ord('T')]:
                    config.SHOW_TRAJECTORIES = not config.SHOW_TRAJECTORIES
                    print(f"📍 Trajectoires: {'ON' if config.SHOW_TRAJECTORIES else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Interruption clavier détectée...")
        finally:
            self.release()

    def release(self) -> None:
        """Libère les ressources."""
        print("\n🧹 Nettoyage des ressources...")
        avg_fps = self.fps_counter.get_average_fps()
        perf_stats = self.perf_monitor.get_stats()
        
        print(f"\n📊 STATISTIQUES FINALES:")
        print(f"   FPS moyen: {avg_fps:.2f}")
        print(f"   Alertes déclenchées: {self.alert_zone.get_alert_count()}")
        
        if perf_stats:
            print(f"\n⚡ PERFORMANCE:")
            for metric, stats in perf_stats.items():
                print(f"   {metric}: {stats['avg']*1000:.1f}ms")
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Système arrêté proprement.\n")

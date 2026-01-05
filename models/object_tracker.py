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
    draw_alert_zone,
    draw_alert_message,
    draw_stats,
    draw_controls_info,
    draw_alert_history,
    FPSGraph
)
from utils.performance import (
    ThreadedVideoCapture,
    FrameProcessor,
    FPSCounter,
    PerformanceMonitor
)

import supervision as sv

class ObjectTracker:
    """
    Système principal de détection et tracking d'objets en temps réel OPTIMISÉ avec SUPERVISION.
    """
    
    def __init__(
        self,
        video_source: cv2.VideoCapture,
        source_type: str = "unknown",
        model_name: str = None
    ):
        """Initialise le système de tracking."""
        self.cap = video_source
        self.source_type = source_type
        
        # Capture vidéo threadée
        if config.ENABLE_THREADING and isinstance(video_source, cv2.VideoCapture):
            print("🚀 Threading activé")
            self.cap = ThreadedVideoCapture(video_source, config.FRAME_BUFFER_SIZE)
        else:
            self.cap = video_source
        
        # Modèle YOLO
        model_path = model_name or config.MODEL_NAME
        print(f"\n🤖 Chargement du modèle YOLO: {model_path}...")
        self.model = YOLO(model_path)
        
        if config.USE_HALF_PRECISION and config.DEVICE != 'cpu':
            try:
                self.model.to(config.DEVICE).half()
                print(f"✅ Half-precision (FP16) activé sur {config.DEVICE}")
            except Exception as e:
                print(f"⚠️  Impossible d'activer FP16: {e}")
        print(f"✅ Modèle chargé avec succès\n")
        
        # Dimensions vidéo
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Zone d'alerte
        self.alert_zone = AlertZone(
            config.ALERT_ZONE,
            self.frame_width,
            self.frame_height
        )
        
        # Utils Perf
        self.fps_counter = FPSCounter(smoothing=0.9)
        self.fps = 0
        self.frame_processor = FrameProcessor(
            (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT) 
            if config.PROCESSING_WIDTH else None
        )
        self.perf_monitor = PerformanceMonitor()
        
        # --- SUPERVISION ANNOTATORS ---
        print("🎨 Initialisation des annotateurs Supervision...")
        
        # Annotateur de boîtes (arrondies et colorées)
        self.box_annotator = sv.BoxAnnotator(
            thickness=config.BOX_THICKNESS,
            color_lookup=sv.ColorLookup.CLASS
        )
        
        # Annotateur de labels (Texte avec fond)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=config.FONT_SCALE,
            text_thickness=config.FONT_THICKNESS,
            text_padding=5,
            text_position=sv.Position.TOP_CENTER,
            color_lookup=sv.ColorLookup.CLASS
        )
        
        # Annotateur de trace (Trajectoires)
        self.trace_annotator = sv.TraceAnnotator(
            trace_length=config.TRAJECTORY_LENGTH if config.SHOW_TRAJECTORIES else 0,
            thickness=config.TRAJECTORY_THICKNESS,
            color_lookup=sv.ColorLookup.CLASS
        )
        
        # Annotateur de Heatmap
        self.heatmap_annotator = sv.HeatMapAnnotator(
            position=sv.Position.BOTTOM_CENTER,
            opacity=config.HEATMAP_ALPHA,
            kernel_size=25,
            step=10,
        ) if config.SHOW_HEATMAP else None
        
        if config.SHOW_FPS_GRAPH:
            print("📊 Graphique FPS activé")
            self.fps_graph = FPSGraph(max_history=config.FPS_HISTORY_LENGTH)
        else:
            self.fps_graph = None

        # Configuration classes
        self.class_names = config.CLASS_IDS
        self.target_class_ids = [
            config.CLASS_NAMES_TO_IDS.get(name, -1)
            for name in config.TARGET_CLASSES
        ]
        
        print(f"📊 Configuration:")
        print(f"   Résolution: {self.frame_width}x{self.frame_height}")
        print(f"   Classes cibles: {', '.join(config.TARGET_CLASSES)}")
        print()

    def _process_detections(self, results, scale_x: float = 1.0, scale_y: float = 1.0):
        """Convertit résultats YOLO en Detections Supervision et gère les alertes."""
        is_alert_active = False
        
        # Conversion directe Ultralytics -> Supervision
        detections = sv.Detections.from_ultralytics(results[0])
        
        # 1. Filtrage par classes cibles
        detections = detections[np.isin(detections.class_id, self.target_class_ids)]
        
        # 2. Filtrage adaptatif par classe (beaucoup plus précis)
        mask = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            class_name = self.class_names.get(class_id, "unknown")
            # Seuil spécifique ou seuil global par défaut
            threshold = config.ADAPTIVE_CONFIDENCE.get(class_name, config.CONFIDENCE_THRESHOLD)
            mask.append(confidence >= threshold)
        
        detections = detections[np.array(mask, dtype=bool)]
        
        # Mise à l'échelle des coordonnées (xyxy)
        if scale_x != 1.0 or scale_y != 1.0:
            detections.xyxy[:, 0] *= scale_x
            detections.xyxy[:, 1] *= scale_y
            detections.xyxy[:, 2] *= scale_x
            detections.xyxy[:, 3] *= scale_y
        
        # Gestion des IDs de tracking (si absents, on assigne -1)
        if detections.tracker_id is None:
             # Fallback si tracking échoue, mais avec model.track ça devrait aller
             detections.tracker_id = np.array([-1] * len(detections), dtype=int)
        
        # Vérification des alertes
        current_active_tracks = []
        
        for xyxy, tracker_id, class_id in zip(detections.xyxy, detections.tracker_id, detections.class_id):
            bbox = tuple(map(int, xyxy))
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # Vérifier l'alerte
            if self.alert_zone.check_and_alert(tracker_id, bbox, class_name):
                is_alert_active = True
                
        return detections, is_alert_active

    def _draw_frame(self, frame: np.ndarray, detections: sv.Detections, is_alert_active: bool) -> None:
        """Dessine les annotations sur la frame."""
        
        # 1. Heatmap (en arrière plan)
        if self.heatmap_annotator and config.SHOW_HEATMAP:
            frame = self.heatmap_annotator.annotate(scene=frame, detections=detections)
            
        # 2. Zone d'alerte
        draw_alert_zone(frame, self.alert_zone.get_zone_coords(), is_alert_active or self.alert_zone.is_alert_active)
        
        # 3. Trajectoires
        if config.SHOW_TRAJECTORIES:
            frame = self.trace_annotator.annotate(scene=frame, detections=detections)
            
        # 4. Boîtes
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        
        # 5. Labels
        labels = [
            f"#{tracker_id} {self.class_names.get(class_id, 'unk')} {confidence:.2f}"
            for tracker_id, class_id, confidence
            in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        # 6. UI Overlay (Alertes, Stats) - Fonctions existantes
        if is_alert_active or self.alert_zone.is_alert_active:
            draw_alert_message(frame, "ALERTE INTRUSION!", position='top')
            
        # Alert History
        alert_history = self.alert_zone.get_alert_history()
        if alert_history:
            recent_alerts = []
            for alert in alert_history[-5:]:
                elapsed = time.time() - alert['timestamp'].timestamp()
                time_ago = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed//60)}m"
                recent_alerts.append({'time_ago': time_ago, 'track_id': alert['track_id'], 'class_name': alert['class_name']})
            if recent_alerts:
                draw_alert_history(frame, recent_alerts)

        # Stats Globales
        draw_stats(frame, self.fps, len(detections), self.source_type, self.alert_zone.get_alert_count())
        
        # Graphique FPS
        if self.fps_graph:
            self.fps_graph.update(self.fps)
            self.fps_graph.draw(frame, (15, frame.shape[0] - 80), target_fps=self.original_fps)
        
        # Controls
        draw_controls_info(frame)

    def run(self) -> None:
        """Boucle principale."""
        print("="*70)
        print("🚀 SYSTÈME SUPERVISION + YOLO11 DÉMARRÉ")
        print("="*70)
        
        # État persistant pour frame skipping
        last_detections = sv.Detections.empty()
        last_detections.tracker_id = np.array([], dtype=int)
        last_is_alert_active = False

        # Initialisation de la fenêtre redimensionnable
        cv2.namedWindow('Supervision Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Supervision Tracking', config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
        
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
                
                # Processing Frame (Low Res)
                processed_frame = self.frame_processor.process(frame, config.FRAME_SKIP)
                
                if processed_frame is not None:
                    # Scaling factors
                    ph, pw = processed_frame.shape[:2]
                    h, w = frame.shape[:2]
                    scale_x = w / pw
                    scale_y = h / ph
                    
                    # Détection + Tracking (Ultralytics)
                    results = self.model.track(
                        processed_frame,
                        conf=config.CONFIDENCE_THRESHOLD,
                        iou=config.IOU_THRESHOLD,
                        persist=True,
                        tracker=config.TRACKER_TYPE,
                        verbose=False,
                        device=config.DEVICE,
                        imgsz=config.PROCESSING_WIDTH
                    )
                    
                    # Conversion vers Supervision
                    last_detections, last_is_alert_active = self._process_detections(results, scale_x, scale_y)
                
                # Visualisation (Toujours sur frame originale)
                # Note: sv.annotate modifie l'image in-place ou retourne une copie
                # Supervision annotators return the annotated frame usually.
                self._draw_frame(frame, last_detections, last_is_alert_active)
                
                # FPS
                self.fps = self.fps_counter.update()
                
                cv2.imshow('Supervision Tracking', frame)
                
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
            pass
        finally:
            self.release()

    def release(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Système arrêté.")

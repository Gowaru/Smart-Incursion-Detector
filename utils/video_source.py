"""
Gestion des sources vidéo pour le système de tracking.
Fournit une interface CLI interactive pour sélectionner la source vidéo au lancement.
"""

import cv2
import os
from typing import Tuple, Optional


class VideoSourceManager:
    """
    Gestionnaire de sources vidéo avec interface CLI interactive.
    
    Permet à l'utilisateur de choisir entre:
    - Webcam (temps réel)
    - Fichier vidéo préenregistré
    """
    
    def __init__(self):
        """Initialise le gestionnaire de sources vidéo."""
        self.source_type = None
        self.source_path = None
    
    def select_source(self) -> Tuple[Optional[cv2.VideoCapture], str]:
        """
        Affiche un menu interactif pour sélectionner la source vidéo.
        
        Returns:
            Tuple[Optional[cv2.VideoCapture], str]: 
                - VideoCapture initialisé (ou None si erreur/annulation)
                - Type de source ('webcam', 'file', ou 'quit')
        """
        print("\n" + "="*60)
        print("  SYSTÈME DE DÉTECTION ET TRACKING D'OBJETS")
        print("="*60)
        print("\nSélectionnez une source vidéo :\n")
        print("  [1] Webcam (temps réel)")
        print("  [2] Fichier vidéo")
        print("  [0] Quitter\n")
        print("-"*60)
        
        while True:
            try:
                choice = input("\nVotre choix (0-2) : ").strip()
                
                if choice == "0":
                    print("\n❌ Programme annulé par l'utilisateur.")
                    return None, "quit"
                
                elif choice == "1":
                    return self._init_webcam()
                
                elif choice == "2":
                    return self._init_video_file()
                
                else:
                    print("⚠️  Choix invalide. Veuillez entrer 0, 1 ou 2.")
            
            except KeyboardInterrupt:
                print("\n\n❌ Programme interrompu par l'utilisateur.")
                return None, "quit"
    
    def _init_webcam(self) -> Tuple[Optional[cv2.VideoCapture], str]:
        """
        Initialise la capture depuis la webcam.
        
        Returns:
            Tuple[Optional[cv2.VideoCapture], str]: VideoCapture et type 'webcam'
        """
        print("\n🎥 Initialisation de la webcam...")
        
        # Essayer d'ouvrir la webcam par défaut (index 0)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Erreur: Impossible d'ouvrir la webcam.")
            print("   Vérifiez que:")
            print("   - La webcam est connectée")
            print("   - Aucune autre application n'utilise la webcam")
            print("   - Vous avez les permissions nécessaires")
            return None, "webcam"
        
        # Vérifier qu'on peut lire une frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur: Impossible de lire depuis la webcam.")
            cap.release()
            return None, "webcam"
        
        # Obtenir les informations de la webcam
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Webcam initialisée avec succès")
        print(f"   Résolution: {width}x{height}")
        print(f"   FPS: {fps if fps > 0 else 'Non disponible'}")
        
        self.source_type = "webcam"
        return cap, "webcam"
    
    def _init_video_file(self) -> Tuple[Optional[cv2.VideoCapture], str]:
        """
        Initialise la capture depuis un fichier vidéo.
        Demande le chemin du fichier à l'utilisateur.
        
        Returns:
            Tuple[Optional[cv2.VideoCapture], str]: VideoCapture et type 'file'
        """
        print("\n📁 Sélection d'un fichier vidéo")
        print("-"*60)
        
        while True:
            file_path = input("\nChemin du fichier vidéo (ou 'q' pour retour): ").strip()
            
            if file_path.lower() == 'q':
                return self.select_source()  # Retour au menu principal
            
            # Supprimer les guillemets si présents
            file_path = file_path.strip('"\'')
            
            # Vérifier l'existence du fichier
            if not os.path.exists(file_path):
                print(f"❌ Erreur: Fichier introuvable: {file_path}")
                print("   Vérifiez le chemin et réessayez.")
                continue
            
            # Vérifier l'extension
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext not in valid_extensions:
                print(f"⚠️  Attention: Extension '{file_ext}' non standard.")
                print(f"   Extensions recommandées: {', '.join(valid_extensions)}")
                confirm = input("   Continuer quand même? (o/N): ").lower()
                if confirm not in ['o', 'oui', 'y', 'yes']:
                    continue
            
            # Essayer d'ouvrir le fichier
            print(f"\n🎬 Ouverture de: {os.path.basename(file_path)}...")
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                print("❌ Erreur: Impossible d'ouvrir le fichier vidéo.")
                print("   Le fichier peut être corrompu ou dans un format non supporté.")
                continue
            
            # Vérifier qu'on peut lire une frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur: Impossible de lire le fichier vidéo.")
                cap.release()
                continue
            
            # Remettre au début
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Obtenir les informations de la vidéo
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"✅ Vidéo chargée avec succès")
            print(f"   Résolution: {width}x{height}")
            print(f"   FPS: {fps:.2f}")
            print(f"   Frames: {frame_count}")
            print(f"   Durée: {duration:.2f}s ({int(duration//60)}m {int(duration%60)}s)")
            
            self.source_type = "file"
            self.source_path = file_path
            return cap, "file"
    
    def get_source_info(self) -> dict:
        """
        Retourne les informations sur la source vidéo actuelle.
        
        Returns:
            dict: Informations sur la source (type, chemin, etc.)
        """
        return {
            'type': self.source_type,
            'path': self.source_path
        }

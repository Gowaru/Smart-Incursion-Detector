"""
Point d'entrée principal du système de détection et tracking d'objets.
Interface CLI interactive pour sélectionner la source vidéo et lancer le système.
"""

import sys
import os

# Ajouter le répertoire racine au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.video_source import VideoSourceManager
from models.object_tracker import ObjectTracker
from config import config


def display_welcome() -> None:
    """
    Affiche le message de bienvenue et les informations du système.
    """
    print("\n" + "="*70)
    print("  🎯 SYSTÈME DE DÉTECTION ET TRACKING D'OBJETS")
    print("="*70)
    print("\n🚀 FONCTIONNALITÉS ACTIVÉES:")
    print("   ✅ Threading pour capture vidéo")
    print("   ✅ Filtrage Kalman pour tracking précis")
    print("   ✅ Trajectoires des objets")
    print("   ✅ Graphique FPS en temps réel")
    print("   ✅ Heatmap d'activité (toggle H)")
    print("\n📋 Informations du système:")
    print(f"   Modèle YOLO      : {config.MODEL_NAME}")
    print(f"   Classes cibles   : {', '.join(config.TARGET_CLASSES)}")
    print(f"   Seuil confiance  : {config.CONFIDENCE_THRESHOLD}")
    print()


def main():
    """
    Fonction principale du programme.
    
    Workflow:
    1. Afficher le message de bienvenue
    2. Sélectionner la source vidéo (webcam ou fichier)
    3. Initialiser le système de tracking
    4. Lancer la boucle de traitement
    5. Nettoyer les ressources
    """
    try:
        # Afficher le message de bienvenue
        display_welcome()
        
        # Initialiser le gestionnaire de sources vidéo
        video_manager = VideoSourceManager()
        
        # Sélectionner la source vidéo (interface interactive)
        video_source, source_type = video_manager.select_source()
        
        # Vérifier si l'utilisateur a quitté
        if video_source is None or source_type == "quit":
            print("\n👋 Au revoir!\n")
            return 0
        
        # Initialiser le système de tracking
        print("\n" + "="*70)
        print("  🔧 INITIALISATION DU SYSTÈME")
        print("="*70)
        
        tracker = ObjectTracker(
            video_source=video_source,
            source_type=source_type,
            model_name=config.MODEL_NAME
        )
        
        # Lancer le système
        tracker.run()
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Programme interrompu par l'utilisateur (Ctrl+C)")
        return 1
    
    except Exception as e:
        print(f"\n❌ ERREUR FATALE: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        print("\n💡 Conseils de dépannage:")
        print("   - Vérifiez que toutes les dépendances sont installées")
        print("   - Assurez-vous que le modèle YOLO est téléchargé")
        print("   - Vérifiez les permissions d'accès à la webcam/fichier")
        print("\n   Pour plus d'aide, consultez le README.md\n")
        
        # Afficher le traceback complet en mode debug
        import traceback
        print("\n📝 Traceback complet:")
        print("-"*70)
        traceback.print_exc()
        print("-"*70 + "\n")
        
        return 1


if __name__ == "__main__":
    """
    Point d'entrée du programme.
    Lance la fonction main() et gère le code de sortie.
    """
    exit_code = main()
    sys.exit(exit_code)

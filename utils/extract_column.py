import cv2
import numpy as np
import os
from doctr.models import ocr_predictor
from typing import Tuple, List
import matplotlib.pyplot as plt
model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31', pretrained=True)
class TableColumnExtractor:
    def __init__(self, debug_mode: bool = False):
        """
        Extracteur de colonnes de tableau avec détection automatique.
        
        Args:
            debug_mode: Active l'affichage des étapes de traitement
        """
        self.debug_mode = debug_mode
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraite l'image pour améliorer la détection des lignes.
        """
        # Conversion en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Débruitage
        gray = cv2.medianBlur(gray, 3)
        
        return gray
    
    def detect_table_lines(self, image: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Détecte les lignes horizontales et verticales du tableau.
        
        Returns:
            Tuple contenant (lignes_horizontales, lignes_verticales)
        """
        gray = self.preprocess_image(image)
        height, width = gray.shape
        
        # Détection des contours
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Morphologie pour connecter les lignes brisées
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Détection des lignes avec HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=width//4, maxLineGap=10)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculer l'angle de la ligne
                angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                
                # Classifier les lignes
                if angle < 30:  # Ligne horizontale
                    y_avg = (y1 + y2) // 2
                    if 10 < y_avg < height - 10:  # Éviter les bords
                        horizontal_lines.append(y_avg)
                elif angle > 60:  # Ligne verticale
                    x_avg = (x1 + x2) // 2
                    if 10 < x_avg < width - 10:  # Éviter les bords
                        vertical_lines.append(x_avg)
        
        # Méthode alternative : projection de pixels
        if len(vertical_lines) < 2:
            vertical_lines = self.detect_columns_by_projection(gray)
        
        # Nettoyer et trier les lignes
        horizontal_lines = sorted(list(set(horizontal_lines)))
        vertical_lines = sorted(list(set(vertical_lines)))
        
        # Fusionner les lignes proches
        horizontal_lines = self.merge_close_lines(horizontal_lines, threshold=20)
        vertical_lines = self.merge_close_lines(vertical_lines, threshold=15)
        
        if self.debug_mode:
            self.visualize_detected_lines(image, horizontal_lines, vertical_lines)
            
        return horizontal_lines, vertical_lines
    
    def detect_columns_by_projection(self, gray: np.ndarray) -> List[int]:
        """
        Détecte les colonnes par projection verticale des pixels blancs/noirs.
        """
        height, width = gray.shape
        
        # Binarisation
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Projection verticale (compter les pixels noirs par colonne)
        vertical_projection = np.sum(binary == 0, axis=0)
        
        # Lisser la projection
        try:
            from scipy import ndimage
            vertical_projection = ndimage.gaussian_filter1d(vertical_projection.astype(float), sigma=2)
        except ImportError:
            # Si scipy n'est pas disponible, utiliser un lissage simple
            kernel = np.ones(5) / 5
            vertical_projection = np.convolve(vertical_projection, kernel, mode='same')
        
        # Trouver les pics et vallées
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(vertical_projection, height=height*0.1, distance=width//10)
            valleys, _ = find_peaks(-vertical_projection, height=-height*0.05, distance=width//15)
        except ImportError:
            # Méthode alternative sans scipy
            valleys = []
            threshold = np.mean(vertical_projection) * 0.5
            for i in range(1, len(vertical_projection)-1):
                if (vertical_projection[i] < threshold and 
                    vertical_projection[i] < vertical_projection[i-1] and 
                    vertical_projection[i] < vertical_projection[i+1]):
                    valleys.append(i)
        
        if self.debug_mode:
            plt.figure(figsize=(12, 4))
            plt.plot(vertical_projection)
            if 'peaks' in locals():
                plt.scatter(peaks, vertical_projection[peaks], color='red', label='Pics (colonnes)')
            plt.scatter(valleys, vertical_projection[valleys], color='blue', label='Vallées (séparateurs)')
            plt.title('Projection verticale pour détection des colonnes')
            plt.legend()
            plt.show()
        
        return list(valleys)
    
    def merge_close_lines(self, lines: List[int], threshold: int) -> List[int]:
        """
        Fusionne les lignes qui sont proches les unes des autres.
        """
        if not lines:
            return lines
            
        merged = [lines[0]]
        for line in lines[1:]:
            if line - merged[-1] > threshold:
                merged.append(line)
            else:
                # Prendre la moyenne des lignes proches
                merged[-1] = (merged[-1] + line) // 2
        
        return merged
    
    def extract_column_boundaries(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Détermine les limites de chaque colonne.
        
        Returns:
            Liste de tuples (x_debut, x_fin) pour chaque colonne
        """
        horizontal_lines, vertical_lines = self.detect_table_lines(image)
        height, width = image.shape[:2]
        
        # Si aucune ligne verticale n'est détectée, essayer une méthode adaptative
        if not vertical_lines:
            print("Aucune ligne verticale détectée, utilisation de la méthode adaptative...")
            return self.adaptive_column_detection(image)
        
        # Ajouter les bords si nécessaire
        if not vertical_lines or vertical_lines[0] > 20:
            vertical_lines.insert(0, 0)
        if not vertical_lines or vertical_lines[-1] < width - 20:
            vertical_lines.append(width)
        
        # Créer les limites des colonnes
        column_boundaries = []
        for i in range(len(vertical_lines) - 1):
            x_start = vertical_lines[i]
            x_end = vertical_lines[i + 1]
            
            # S'assurer que la colonne a une largeur minimale
            if x_end - x_start > 30:
                column_boundaries.append((x_start, x_end))
        
        return column_boundaries
    
    def adaptive_column_detection(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Méthode adaptative pour détecter les colonnes sans lignes visibles.
        """
        gray = self.preprocess_image(image)
        height, width = gray.shape
        
        # Analyser le contenu par segments
        num_segments = max(6, width // 100)  # Au moins 6 segments
        segment_width = width // num_segments
        
        content_density = []
        for i in range(num_segments):
            x_start = i * segment_width
            x_end = min((i + 1) * segment_width, width)
            
            segment = gray[:, x_start:x_end]
            
            # Calculer la densité de contenu (pixels sombres)
            _, binary = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            density = np.sum(binary == 0) / (segment.shape[0] * segment.shape[1])
            content_density.append(density)
        
        # Trouver les transitions significatives dans la densité
        density_diff = np.diff(content_density)
        threshold = np.std(density_diff) * 0.5
        
        column_separators = [0]  # Commencer par le bord gauche
        
        for i, diff in enumerate(density_diff):
            if abs(diff) > threshold:
                separator_x = (i + 1) * segment_width
                if separator_x - column_separators[-1] > width // 10:  # Largeur minimale
                    column_separators.append(separator_x)
        
        column_separators.append(width)  # Ajouter le bord droit
        
        # Créer les limites des colonnes
        column_boundaries = []
        for i in range(len(column_separators) - 1):
            x_start = column_separators[i]
            x_end = column_separators[i + 1]
            column_boundaries.append((x_start, x_end))
        
        return column_boundaries
    
    def visualize_detected_lines(self, image: np.ndarray, horizontal_lines: List[int], 
                                vertical_lines: List[int]):
        """
        Visualise les lignes détectées sur l'image.
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        height, width = image.shape[:2]
        
        # Dessiner les lignes horizontales en rouge
        for y in horizontal_lines:
            cv2.line(vis_image, (0, y), (width, y), (0, 0, 255), 2)
        
        # Dessiner les lignes verticales en bleu
        for x in vertical_lines:
            cv2.line(vis_image, (x, 0), (x, height), (255, 0, 0), 2)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title('Lignes détectées (Rouge: horizontales, Bleu: verticales)')
        plt.axis('off')
        plt.show()
    
    def extract_columns(self, image_path: str, output_dir: str = None) -> List[str]:
        """
        Extrait toutes les colonnes du tableau et les sauvegarde.
        
        Args:
            image_path: Chemin vers l'image contenant le tableau
            output_dir: Répertoire de sortie (par défaut: même dossier que l'image)
            
        Returns:
            Liste des chemins des images de colonnes extraites
        """
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        print(f"Image chargée: {image.shape}")
        
        # Définir le répertoire de sortie
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
            if not output_dir:
                output_dir = "."
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraire les limites des colonnes
        column_boundaries = self.extract_column_boundaries(image)
        
        if not column_boundaries:
            print("Aucune colonne détectée!")
            return []
        
        print(f"Nombre de colonnes détectées: {len(column_boundaries)}")
        
        # Extraire et sauvegarder chaque colonne
        extracted_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for i, (x_start, x_end) in enumerate(column_boundaries):
            # Extraire la colonne avec une petite marge
            margin = 5
            x_start = max(0, x_start - margin)
            x_end = min(image.shape[1], x_end + margin)
            
            column_image = image[:, x_start:x_end]
            
            # Nom du fichier de sortie
            output_path = os.path.join(output_dir, f"{base_name}_colonne_{i+1}.png")
            
            # Sauvegarder la colonne
            success = cv2.imwrite(output_path, column_image)
            
            if success:
                extracted_paths.append(output_path)
                print(f"✅ Colonne {i+1} sauvegardée: {x_start}-{x_end} → {output_path}")
                print(f"   Taille: {column_image.shape[1]}x{column_image.shape[0]} pixels")
            else:
                print(f"❌ Erreur lors de la sauvegarde de la colonne {i+1}: {output_path}")
        
        # Vérifier que des colonnes ont été extraites
        if extracted_paths:
            print(f"\n🎉 Extraction terminée avec succès!")
            print(f"📁 {len(extracted_paths)} colonnes sauvegardées dans: {output_dir}")
            for path in extracted_paths:
                file_size = os.path.getsize(path) / 1024  # Taille en KB
                print(f"   • {os.path.basename(path)} ({file_size:.1f} KB)")
        else:
            print(f"\n⚠️ Aucune colonne n'a pu être extraite et sauvegardée!")
        
        return extracted_paths


def get_user_input():
    """Demande à l'utilisateur de saisir les paramètres."""
    print("=" * 60)
    print("🔧 EXTRACTEUR DE COLONNES DE TABLEAU")
    print("=" * 60)
    
    # Demander le chemin de l'image
    while True:
        image_path = input("\n📁 Entrez le chemin de votre image de tableau: ").strip().strip('"\'')
        
        if not image_path:
            print("❌ Veuillez entrer un chemin valide.")
            continue
            
        if not os.path.exists(image_path):
            print(f"❌ Fichier non trouvé: {image_path}")
            print("💡 Astuce: Vérifiez le chemin ou glissez-déposez le fichier dans le terminal")
            continue
            
        # Vérifier si c'est une image
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            print(f"❌ Format non supporté. Extensions valides: {', '.join(valid_extensions)}")
            continue
            
        break
    
    # Demander le dossier de sortie (optionnel)
    output_dir = input("\n📂 Dossier de sortie (laissez vide pour utiliser le même dossier): ").strip().strip('"\'')
    if not output_dir:
        output_dir = os.path.dirname(image_path)
        if not output_dir:
            output_dir = "."

    # Mode debug
    debug_choice = input("\n🐛 Activer le mode debug pour voir les étapes? (o/n): ").strip().lower()
    debug_mode = debug_choice in ['o', 'oui', 'y', 'yes']
    
    return image_path, output_dir, debug_mode


def main():
    """Fonction principale avec interface utilisateur."""
    try:
        # Vérifier les dépendances
        try:
            import scipy
        except ImportError:
            print("⚠️ scipy non trouvé. Certaines fonctionnalités avancées seront désactivées.")
            print("Pour installer: pip install scipy")
        
        # Obtenir les paramètres de l'utilisateur
        image_path, output_dir, debug_mode = get_user_input()
        
        print(f"\n🚀 Démarrage de l'extraction...")
        print(f"📷 Image: {image_path}")
        print(f"📁 Sortie: {output_dir}")
        print(f"🐛 Debug: {'Activé' if debug_mode else 'Désactivé'}")
        
        # Créer l'extracteur
        extractor = TableColumnExtractor(debug_mode=debug_mode)
        
        # Effectuer l'extraction
        colonnes = extractor.extract_columns(image_path, output_dir)
        
        if colonnes:
            print(f"\n✅ SUCCÈS! {len(colonnes)} colonnes extraites:")
            for i, colonne in enumerate(colonnes, 1):
                print(f"   {i}. {os.path.basename(colonne)}")
        else:
            print("\n❌ ÉCHEC: Aucune colonne n'a pu être extraite.")
            print("💡 Suggestions:")
            print("   • Vérifiez que l'image contient bien un tableau")
            print("   • Essayez le mode debug pour voir les étapes")
            print("   • Assurez-vous que le tableau a des lignes visibles")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Arrêt demandé par l'utilisateur.")
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {str(e)}")
        print("🔧 Vérifiez votre image et réessayez.")



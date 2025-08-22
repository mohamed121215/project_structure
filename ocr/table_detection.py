import cv2
import numpy as np
from doctr.models import ocr_predictor
from typing import Tuple, Optional
import math
from ocr.preprocessing import preprocess_image
model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31', pretrained=True)
def detecter_lignes_tableau(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Détection améliorée des lignes de tableau"""
    
    # 1. Détection des lignes horizontales avec des tailles variables
    horizontal_kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    ]
    
    horizontal_lines = np.zeros_like(image)
    for kernel in horizontal_kernels:
        temp = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        horizontal_lines = cv2.bitwise_or(horizontal_lines, temp)
    
    # 2. Détection des lignes verticales avec des tailles variables
    vertical_kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    ]
    
    vertical_lines = np.zeros_like(image)
    for kernel in vertical_kernels:
        temp = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        vertical_lines = cv2.bitwise_or(vertical_lines, temp)
    
    # 3. Combinaison pondérée des lignes
    table_structure = cv2.addWeighted(horizontal_lines, 0.6, vertical_lines, 0.4, 0)
    
    return horizontal_lines, vertical_lines, table_structure

def analyser_grille_tableau(contour, image_shape) -> dict:
    """Analyse si un contour correspond vraiment à un tableau"""
    height, width = image_shape[:2]
    aire = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    
    # Créer un masque pour ce contour
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    
    # Région d'intérêt
    roi_mask = mask[y:y+h, x:x+w]
    
    # 1. Vérifier la densité de lignes horizontales et verticales
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    
    # Appliquer sur la région découpée
    gray_roi = cv2.cvtColor(cv2.bitwise_and(
        cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR), 
        cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    ), cv2.COLOR_BGR2GRAY) if len(roi_mask.shape) == 3 else roi_mask
    
    horizontal_density = cv2.morphologyEx(gray_roi, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_density = cv2.morphologyEx(gray_roi, cv2.MORPH_OPEN, vertical_kernel)
    
    # Compter les pixels de lignes
    h_pixels = np.count_nonzero(horizontal_density)
    v_pixels = np.count_nonzero(vertical_density)
    total_pixels = w * h
    
    # Ratios de lignes
    ratio_h_lines = h_pixels / total_pixels if total_pixels > 0 else 0
    ratio_v_lines = v_pixels / total_pixels if total_pixels > 0 else 0
    
    return {
        'aire': aire,
        'ratio_h_lines': ratio_h_lines,
        'ratio_v_lines': ratio_v_lines,
        'bbox': (x, y, w, h),
        'has_grid_structure': ratio_h_lines > 0.01 and ratio_v_lines > 0.01
    }

def detecter_cellules_tableau(image_region: np.ndarray) -> int:
    """Compte le nombre approximatif de cellules dans une région"""
    
    # Détection des contours internes (cellules)
    edges = cv2.Canny(image_region, 50, 150)
    
    # Fermeture pour connecter les lignes brisées
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Trouver les contours des cellules
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours qui ressemblent à des cellules
    cellules = 0
    h, w = image_region.shape[:2]
    min_cell_area = (w * h) / 200  # Une cellule fait au minimum 1/200 de l'image
    max_cell_area = (w * h) / 4    # Et au maximum 1/4
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_cell_area < area < max_cell_area:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Une cellule a un ratio pas trop extrême
            if 0.2 < aspect_ratio < 5.0:
                cellules += 1
    
    return cellules

def calculer_score_tableau(contour, image_shape, enhanced_image) -> float:
    """Calcule un score de confiance pour un tableau potentiel"""
    
    height, width = image_shape[:2]
    aire_image = width * height
    
    # Analyse de base
    analyse = analyser_grille_tableau(contour, image_shape)
    aire = analyse['aire']
    x, y, w, h = analyse['bbox']
    
    # Critères de base
    ratio_aspect = w / h if h > 0 else 0
    ratio_aire = aire / aire_image
    ratio_remplissage = aire / (w * h) if (w * h) > 0 else 0
    
    # Score = 0 si ne respecte pas les critères de base
    if ratio_aire < 0.15:  # Trop petit (moins de 15% de l'image)
        return 0
    
    if ratio_aspect < 1.0 or ratio_aspect > 8.0:  # Ratio trop extrême
        return 0
    
    if not analyse['has_grid_structure']:  # Pas de structure de grille
        return 0
    
    # Extraire la région pour analyse approfondie
    roi = enhanced_image[y:y+h, x:x+w]
    nb_cellules = detecter_cellules_tableau(roi)
    
    # Calcul du score pondéré
    score = 0
    
    # 1. Score de taille (30% du total)
    score_taille = min(ratio_aire * 3, 1) * 30
    score += score_taille
    
    # 2. Score de forme (20% du total)
    score_forme = min(ratio_aspect / 2, 1) * 20 if ratio_aspect <= 4 else 20 * (1 - (ratio_aspect - 4) / 4)
    score += max(0, score_forme)
    
    # 3. Score de structure grille (25% du total)
    score_grille = (analyse['ratio_h_lines'] + analyse['ratio_v_lines']) * 500
    score += min(score_grille, 25)
    
    # 4. Score de cellules (20% du total)
    score_cellules = min(nb_cellules / 10, 1) * 20  # Plus de cellules = mieux
    score += score_cellules
    
    # 5. Score de position (5% du total)
    center_x, center_y = x + w//2, y + h//2
    distance_centre = math.sqrt((center_x - width//2)**2 + (center_y - height//2)**2)
    max_distance = math.sqrt((width//2)**2 + (height//2)**2)
    score_position = (1 - distance_centre / max_distance) * 5
    score += score_position
    
    return score

def detecter_vrai_tableau_ameliore(image_path: str, afficher_debug: bool = False) -> Optional[dict]:
    """
    Version améliorée de la détection de tableau avec analyse multicritères.
    """
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    
    height, width = image.shape[:2]
    
    # Préprocessing amélioré
    enhanced = preprocess_image(image)
    
    # Détection des lignes améliorée
    horizontal_lines, vertical_lines, table_structure = detecter_lignes_tableau(enhanced)
    
    # Amélioration de la détection des contours
    edges = cv2.Canny(table_structure, 50, 150)
    
    # Morphologie pour connecter les éléments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
    
    if afficher_debug:
        cv2.imshow('Enhanced', enhanced)
        cv2.imshow('Lignes horizontales', horizontal_lines)
        cv2.imshow('Lignes verticales', vertical_lines)
        cv2.imshow('Structure tableau', table_structure)
        cv2.imshow('Edges finaux', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Évaluer chaque contour
    candidats = []
    for contour in contours:
        score = calculer_score_tableau(contour, (height, width), enhanced)
        if score > 30:  # Seuil minimum plus élevé
            x, y, w, h = cv2.boundingRect(contour)
            candidats.append({
                'contour': contour,
                'score': score,
                'bbox': (x, y, w, h)
            })
    
    if not candidats:
        return None
    
    # Trier par score décroissant
    candidats.sort(key=lambda x: x['score'], reverse=True)
    meilleur = candidats[0]
    
    x, y, w, h = meilleur['bbox']
    aire = cv2.contourArea(meilleur['contour'])
    
    return {
        'haut': y,
        'bas': y + h,
        'gauche': x,
        'droite': x + w,
        'largeur': w,
        'hauteur': h,
        'aire': aire,
        'score': meilleur['score'],
        'ratio_aspect': w / h if h > 0 else 0,
        'ratio_remplissage': aire / (w * h) if (w * h) > 0 else 0
    }

def detecter_tableau_par_ocr_zones(image_path: str) -> Optional[dict]:
    """
    Détection basée sur les zones de texte organisé (alternative robuste).
    """
    
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Seuillage adaptatif pour isoler le texte
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Nettoyage du bruit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Dilatation horizontale pour connecter les mots d'une ligne
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    h_dilated = cv2.morphologyEx(binary, cv2.MORPH_DILATE, h_kernel)
    
    # Dilatation verticale légère pour connecter les lignes proches
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    combined = cv2.morphologyEx(h_dilated, cv2.MORPH_DILATE, v_kernel)
    
    # Trouver les zones de texte
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyser chaque zone pour détecter une structure tabulaire
    meilleur_zone = None
    meilleur_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < (width * height * 0.1):  # Trop petit
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extraire la région
        roi = binary[y:y+h, x:x+w]
        
        # Compter les lignes horizontales de texte
        h_proj = np.sum(roi, axis=1)  # Projection horizontale
        lignes_texte = np.where(h_proj > w * 0.1)[0]  # Lignes avec suffisamment de texte
        
        # Compter les "colonnes" approximatives
        v_proj = np.sum(roi, axis=0)  # Projection verticale
        colonnes = np.where(v_proj > h * 0.05)[0]
        
        nb_lignes = len(lignes_texte)
        nb_colonnes = len(colonnes)
        
        # Score basé sur la régularité (tableau = lignes et colonnes régulières)
        if nb_lignes >= 3 and nb_colonnes >= 2:
            ratio_aspect = w / h if h > 0 else 0
            score_organisation = min(nb_lignes * nb_colonnes, 50)
            score_taille = (area / (width * height)) * 100
            score_forme = 20 if 1.5 < ratio_aspect < 6 else 0
            
            score_total = score_organisation + score_taille + score_forme
            
            if score_total > meilleur_score:
                meilleur_score = score_total
                meilleur_zone = {
                    'haut': y,
                    'bas': y + h,
                    'gauche': x,
                    'droite': x + w,
                    'largeur': w,
                    'hauteur': h,
                    'aire': area,
                    'score': score_total,
                    'nb_lignes': nb_lignes,
                    'nb_colonnes': nb_colonnes
                }
    
    return meilleur_zone

def detecter_tableau_final(image_path: str, afficher_resultat: bool = True) -> Optional[str]:
    """
    Fonction principale améliorée avec multiple méthodes de détection.
    """
    
    print("🔍 Détection de tableau améliorée...")
    print("=" * 50)
    
    resultats = []
    
    # Méthode 1: Détection par structure de grille améliorée
    print("📊 Méthode 1: Analyse de structure de grille...")
    try:
        resultat1 = detecter_vrai_tableau_ameliore(image_path)
        if resultat1:
            resultats.append(('Grille améliorée', resultat1))
            print(f"   ✅ Score: {resultat1['score']:.1f}")
        else:
            print("   ❌ Aucun tableau détecté")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Méthode 2: Détection par zones de texte organisé
    print("📝 Méthode 2: Analyse des zones de texte...")
    try:
        resultat2 = detecter_tableau_par_ocr_zones(image_path)
        if resultat2:
            resultats.append(('Zones de texte', resultat2))
            print(f"   ✅ Score: {resultat2['score']:.1f}")
        else:
            print("   ❌ Aucun tableau détecté")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Sélection du meilleur résultat
    if not resultats:
        print("\n❌ ÉCHEC: Aucune méthode n'a détecté de tableau")
        return None
    
    # Trier par score décroissant
    resultats.sort(key=lambda x: x[1]['score'], reverse=True)
    meilleure_methode, meilleur_resultat = resultats[0]
    
    print(f"\n🏆 MEILLEUR RÉSULTAT:")
    print(f"   Méthode: {meilleure_methode}")
    print(f"   Score: {meilleur_resultat['score']:.1f}")
    print(f"   Dimensions: {meilleur_resultat['largeur']}x{meilleur_resultat['hauteur']}")
    print(f"   Position: ({meilleur_resultat['gauche']}, {meilleur_resultat['haut']})")
    
    # Charger l'image et découper
    image = cv2.imread(image_path)
    x = meilleur_resultat['gauche']
    y = meilleur_resultat['haut'] 
    w = meilleur_resultat['largeur']
    h = meilleur_resultat['hauteur']
    
    # Ajouter une petite marge pour éviter de couper les bordures
    marge = 5
    x = max(0, x - marge)
    y = max(0, y - marge)
    w = min(image.shape[1] - x, w + 2*marge)
    h = min(image.shape[0] - y, h + 2*marge)
    
    tableau_crop = image[y:y+h, x:x+w]
    
    # Sauvegarder
    output_path = "tableau_crop_ameliore.jpg"
    cv2.imwrite(output_path, tableau_crop)
    
    if afficher_resultat:
        # Afficher l'image originale avec le rectangle de détection
        image_debug = image.copy()
        cv2.rectangle(image_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_debug, f"Score: {meilleur_resultat['score']:.1f}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Détection du tableau", cv2.resize(image_debug, (800, 600)))
        cv2.imshow("Tableau extrait", tableau_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\n✅ Tableau extrait et sauvegardé dans: {output_path}")
    return output_path
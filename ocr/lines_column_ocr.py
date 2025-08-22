import cv2
import numpy as np
import tempfile
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from typing import List

model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31', pretrained=True)
def ocr_column_lines1(colonne_img: np.ndarray, max_line_gap: int = 15, max_word_gap: int = 30) -> List[str]:
    """
    Retourne la liste des lignes détectées dans l'image colonne avec docTR, en regroupant les mots proches.
    
    Args:
        colonne_img: Image de la colonne (numpy array).
        max_line_gap: Distance maximale (en pixels) entre deux lignes pour les regrouper.
        max_word_gap: Distance maximale (en pixels) entre deux mots dans une même ligne.
    
    Returns:
        Liste de chaînes, chaque chaîne représentant une ligne logique.
    """
    print("non cellule")
    # Sauvegarder l'image temporairement
    temp_file = "temp_column.png"
    cv2.imwrite(temp_file, colonne_img)
    doc = DocumentFile.from_images(temp_file)
    result = model(doc)
    print(result.render())
    lignes = []
    current_line = []
    last_y_max = None
    last_x_max = None
    
    for block in result.pages[0].blocks:
        for line in block.lines:
            # Trier les mots par position x pour respecter l'ordre de lecture
            words = [(word.value, word.geometry) for word in line.words]
            words.sort(key=lambda x: x[1][0][0])  # Tri par x_min
            
            for word, geometry in words:
                # Coordonnées : ((x_min, y_min), (x_max, y_max))
                y_max = geometry[1][1] * colonne_img.shape[0]  # Normaliser en pixels
                x_max = geometry[1][0] * colonne_img.shape[1]
                
                # Si c'est la première ligne ou si le mot est proche de la ligne précédente
                if last_y_max is None or abs(y_max - last_y_max) <= max_line_gap:
                    # Vérifier l'écart horizontal avec le mot précédent
                    if last_x_max is None or (geometry[0][0] * colonne_img.shape[1] - last_x_max) <= max_word_gap:
                        current_line.append(word)
                    else:
                        # Nouvelle ligne si l'écart horizontal est trop grand
                        if current_line:
                            lignes.append(" ".join(current_line))
                        current_line = [word]
                else:
                    # Terminer la ligne actuelle et commencer une nouvelle
                    if current_line:
                        lignes.append(" ".join(current_line))
                    current_line = [word]
                
                last_y_max = y_max
                last_x_max = x_max
            
    # Ajouter la dernière ligne
    if current_line:
        lignes.append(" ".join(current_line))
    
    os.remove(temp_file)  # Nettoyer le fichier temporaire
    return lignes


def ocr_column_lines2(chemin_image, seuil_hauteur_min=20):
    """
    Extrait le texte de chaque cellule d'une colonne à partir d'une image et affiche les ROI.
    
    Args:
        chemin_image (str): Chemin vers l'image de la colonne
        seuil_hauteur_min (int): Hauteur minimale pour considérer une ligne comme cellule
    
    Returns:
        list: Liste des textes extraits de chaque cellule
    """
    
    # Charger l'image
    image = cv2.imread(chemin_image)
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {chemin_image}")
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un seuillage pour améliorer la détection
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Détecter les contours pour identifier les cellules
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extraire les rectangles englobants et les trier par position verticale
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > seuil_hauteur_min and w > 10:  # Filtrer les petits éléments
            rectangles.append((x, y, w, h))
    
    # Trier par position verticale (y)
    rectangles.sort(key=lambda rect: rect[1])
    
    # Si pas de contours détectés, diviser l'image en sections égales
    if not rectangles:
        hauteur_image = image.shape[0]
        largeur_image = image.shape[1]
        
        # Essayer de détecter automatiquement le nombre de lignes
        projections = np.sum(thresh == 0, axis=1)  # Projection horizontale du texte
        
        # Trouver les zones avec du texte
        seuil_projection = np.max(projections) * 0.1
        zones_texte = projections > seuil_projection
        
        # Identifier les transitions (début/fin de zones de texte)
        transitions = np.diff(zones_texte.astype(int))
        debuts = np.where(transitions == 1)[0] + 1
        fins = np.where(transitions == -1)[0] + 1
        
        # Ajuster pour les cas limites
        if zones_texte[0]:
            debuts = np.concatenate([[0], debuts])
        if zones_texte[-1]:
            fins = np.concatenate([fins, [len(zones_texte)]])
        
        # Créer les rectangles à partir des zones détectées
        for debut, fin in zip(debuts, fins):
            if fin - debut > seuil_hauteur_min:
                rectangles.append((0, debut, largeur_image, fin - debut))
    
    # Extraire le texte de chaque rectangle et afficher le ROI
    textes_extraits = []
    
    for i, (x, y, w, h) in enumerate(rectangles, 1):
        # Extraire la région d'intérêt
        roi = image[y:y+h, x:x+w]  # Garder l'image couleur pour DocTR
       
        
        # Sauvegarde temporaire pour DocTR
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            cv2.imwrite(tmpfile.name, roi)
            doc = DocumentFile.from_images(tmpfile.name)
        
        try:
            # Extraction du texte avec le modèle OCR
            result = model(doc)
            
            cell_text = " ".join(
                " ".join(word.value for word in line.words)
                for block in result.pages[0].blocks
                for line in block.lines
            ).strip()
            
            # Nettoyer le texte extrait
            texte = ' '.join(cell_text.split())  # Supprimer les espaces multiples
            
            if texte:  # N'ajouter que si du texte a été trouvé
                textes_extraits.append(texte)
        
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.unlink(tmpfile.name)
            except:
                pass
    

    
    return textes_extraits
def linesoneortwo(full_img: np.ndarray, med_list: list) -> bool:
    """
    Vérifie s'il existe une ligne horizontale entre le 1er et le 2e médicament de med_list.
    - Si les deux médicaments sont différents : on prend leur 1ʳᵉ occurrence respective.
    - Si ce sont les mêmes : on prend la 1ʳᵉ et la 2ᵉ occurrence dans l'image.
    """
    if not med_list or len(med_list) < 2:
        return False

    # OCR complet
    temp_file = "temp_full.png"
    cv2.imwrite(temp_file, full_img)
    try:
        doc = DocumentFile.from_images(temp_file)
        result = model(doc)
    finally:
        try:
            os.remove(temp_file)
        except:
            pass

    H, W = full_img.shape[:2]

    # Récupérer toutes les occurrences (nom + y_center)
    occurrences = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            line_text = " ".join([w.value for w in line.words]).upper().strip()
            print(line_text)
            for med in med_list:
                med_name = med['nom_de_marque'].upper().strip()
                tokens = med_name.split()
                if (len(tokens) == 1 and tokens[0] in line_text) or \
                    (len(tokens) >= 2 and all(tok in line_text for tok in tokens[:2])):
                    y_min = min(w.geometry[0][1] for w in line.words) * H
                    y_max = max(w.geometry[1][1] for w in line.words) * H
                    y_center = (y_min + y_max) / 2
                    occurrences.append((med_name, y_center))


    # Supprimer les doublons proches
    occurrences_uniques = []
    tol = 2  # pixels de tolérance
    for name, y in occurrences:
        if not any(abs(y - y2) < tol and name == name2 for name2, y2 in occurrences_uniques):
            occurrences_uniques.append((name, y))
    print(occurrences_uniques)
    # Noms normalisés
    first_med = med_list[0]['nom_de_marque'].upper().strip()
    last_med  = med_list[1]['nom_de_marque'].upper().strip()

    y_first, y_last = None, None

    if first_med == last_med:
        # Filtrer les occurrences uniques pour ce médicament
        occs = [(name, y) for name, y in occurrences_uniques if name == first_med]

        # Vérifier qu'il y a bien au moins 2 occurrences
        if len(occs) >= 2:
            y_first = float(occs[0][1])  # première occurrence
            y_last  = float(occs[1][1])  # deuxième occurrence
        else:
            return False

    else:
        y_first = next((y for name, y in occurrences_uniques if name == first_med), None)
        y_last  = next((y for name, y in occurrences_uniques if name == last_med), None)
    
    print(y_first,y_last)
    if y_first is None or y_last is None:
        return False

    y_low, y_high = sorted([y_first, y_last])

    # Détection des lignes horizontales
    gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY) if len(full_img.shape) == 3 else full_img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=50, minLineLength=40, maxLineGap=10)

    if lines is None:
        return False

    min_length = 0.7 * W  # ligne doit couvrir au moins 70% de la largeur
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        length = np.hypot(x2 - x1, y2 - y1)
        y_line = (y1 + y2)/2
        if angle < 5 and length >= min_length and y_low < y_line < y_high:
            return True


    return False


def ocr_lines(colonne_img,chemin,b):
    if b is False:
        print("---->1")
        lignes = ocr_column_lines1(colonne_img)
    
        # Regrouper les lignes contiguës qui font partie d'une même entrée
        lignes_regroupees = []
        current_line = []
        
        for ligne in lignes:
            # Si la ligne est vide, passer à la suivante
            if not ligne.strip():
                continue
            
            # Ajouter la ligne à la ligne courante si elle semble faire partie de la même entrée
            # Critère : pas de nouvelle ligne vide ou de séparation claire (par exemple, nouvelle majuscule seule)
            if current_line and not (ligne.startswith(" ") or ligne.isupper() and not any(c.islower() for c in ligne)):
                current_line.append(ligne)
            else:
                # Sauvegarder la ligne précédente si elle existe
                if current_line:
                    lignes_regroupees.append(" ".join(current_line))
                current_line = [ligne]
        
        # Ajouter la dernière ligne si elle existe
        if current_line:
            lignes_regroupees.append(" ".join(current_line))
        return lignes_regroupees    
    else:
        print("----->2")
        lignes = ocr_column_lines2(chemin)  
        return lignes
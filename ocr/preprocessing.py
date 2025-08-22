import cv2
import re
import spacy
import numpy as np
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Préprocessing avancé de l'image pour améliorer la détection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Débruitage adaptatif
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return enhanced
def improve_column_segmentation(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger {image_path}")

    # Prétraitement pour renforcer les lignes verticales
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
    enhanced = cv2.addWeighted(gray, 0.7, vertical_lines, 0.3, 0.0)

    # Sauvegarder l'image améliorée pour vérification
    cv2.imwrite("enhanced_table.jpg", enhanced)
    return "enhanced_table.jpg"
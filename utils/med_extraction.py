from doctr.models import ocr_predictor
from rapidfuzz import fuzz
import pandas as pd
resultats_dosages_globaux = []
from .text_cleaning import enlever_accents
model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31', pretrained=True)
def extraire_medicaments_depuis_ocr(texte_ocr, chemin_csv, seuil=90):
    """
    Extrait les médicaments du texte OCR sans lemmatisation, en utilisant la correspondance floue.
    
    Args:
        texte_ocr (str): Texte nettoyé issu de l'OCR.
        chemin_csv (str): Chemin vers le fichier CSV contenant la base de médicaments.
        seuil (int): Seuil de similarité pour la correspondance floue (défaut: 90).
    
    Returns:
        list: Liste des informations des médicaments détectés (dictionnaires avec nom et autres colonnes du CSV).
    """
    # Nettoyage initial sans lemmatisation
    tokens_ocr = texte_ocr.split()  # Tokenisation simple par espaces

    texte_ocr_lower = ' '.join(tokens_ocr).lower()
    texte_ocr_lower = enlever_accents(texte_ocr_lower)

    # Charger la base de médicaments
    df = pd.read_csv(chemin_csv)
    noms_medicaments = df['NOM_DE_MARQUE'].dropna().unique().tolist()

    # Pré-calculer les noms nettoyés (sans lemmatisation)
    noms_medicaments_clean = []
    for nom in noms_medicaments:
        nom_clean = nom.lower().strip()
        nom_clean = enlever_accents(nom_clean)
        noms_medicaments_clean.append((nom, nom_clean))

    medicaments_detectes = []
    covered_positions = set()

    # Parcourir les tokens OCR séquentiellement
    tokens_length = len(tokens_ocr)
    i = 0
    while i < tokens_length:
        for window_size in [1, 2, 3]:  # Prioriser les tokens uniques d'abord
            if i + window_size <= tokens_length:
                candidate = ' '.join(tokens_ocr[i:i + window_size]).lower()
                candidate_start = len(' '.join(tokens_ocr[:i])) + (1 if i > 0 else 0)
                candidate_end = candidate_start + len(candidate)

                # Vérifier si la position est déjà couverte
                if any(start <= candidate_end and end >= candidate_start for start, end in covered_positions):
                    break

                # Comparer avec les noms de médicaments
                for nom, nom_clean in noms_medicaments_clean:
                    if abs(len(nom_clean) - len(candidate)) < 10:  # Tolérance augmentée
                        score = fuzz.ratio(enlever_accents(nom_clean), enlever_accents(candidate))
                        if score >= seuil:
                            row = df[df['NOM_DE_MARQUE'].str.lower().str.strip() == nom.lower().strip()].iloc[0]
                            info_medicament = {"nom_de_marque": nom}
                            for col in df.columns:
                                if col != 'NOM_DE_MARQUE':
                                    info_medicament[col.lower()] = row[col]
                            medicaments_detectes.append(info_medicament)
                            covered_positions.add((candidate_start, candidate_end))
                            i += window_size - 1
                            break
                else:
                    continue
                break
        i += 1

    return medicaments_detectes
# MediScan 3D - Extraction Automatique de Médicaments

MediScan 3D est une application web immersive construite avec Streamlit, conçue pour extraire automatiquement les informations sur les médicaments à partir d'images de documents médicaux (comme des ordonnances ou des tableaux). Elle utilise des techniques avancées d'OCR (Reconnaissance Optique de Caractères) avec docTR, de détection de tableaux via OpenCV, et d'extraction intelligente des médicaments via une correspondance floue contre une base de données CSV. L'interface utilisateur est stylisée en 3D holographique pour une expérience futuriste.

## Fonctionnalités Principales

- **Upload d'Images** : Chargez des images contenant des tableaux de médicaments (JPG, PNG, etc.).
- **Détection de Tableaux** : Identification automatique des zones de tableaux.
- **Extraction de Colonnes** : Découpage intelligent des colonnes pour un traitement ciblé.
- **OCR Avancé** : Extraction du texte avec gestion des lignes et cellules.
- **Extraction de Médicaments** : Correspondance floue avec une base CSV pour identifier les noms de marque et autres informations.
- **Nettoyage du Texte** : Normalisation, suppression des accents, et NLP avec spaCy.
- **Interface 3D Immersive** : Design holographique avec animations CSS et thèmes dynamiques.
- **Mode Debug** : Visualisation des étapes intermédiaires.


## Utilisation

1. Lancez l'application :
   ```
   streamlit run app.py
   ```

2. Accédez à `http://localhost:8501` dans votre navigateur.
3. Uploadez une image, fournissez le chemin du CSV, et extrayez les médicaments.


### Détails des Fichiers et Fonctions

#### `app.py`
- **`main()`**: Point d'entrée qui appelle la fonction principale de l'interface (`ui.main`).

#### `interface/ui.py`
- **`main()`**: Configure l'interface Streamlit avec un design 3D holographique, gère l'upload d'images, le choix du thème, et l'extraction des médicaments. Affiche les résultats et gère les erreurs avec un style visuel.

#### `ocr/preprocessing.py`
- **`preprocess_image(image: np.ndarray) -> np.ndarray`**: Convertit l'image en niveaux de gris, applique un débruitage et améliore le contraste avec CLAHE pour optimiser la détection.
- **`improve_column_segmentation(image_path: str) -> str`**: Renforce les lignes verticales pour améliorer la segmentation des colonnes, sauvegarde l'image traitée.

#### `ocr/table_detection.py`
- **`detecter_lignes_tableau(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`**: Détecte les lignes horizontales et verticales d'un tableau avec des noyaux morphologiques de tailles variables.
- **`analyser_grille_tableau(contour, image_shape) -> dict`**: Analyse un contour pour vérifier s'il correspond à un tableau en évaluant la densité des lignes.
- **`detecter_cellules_tableau(image_region: np.ndarray) -> int`**: Compte les cellules dans une région en détectant les contours internes.
- **`detecter_vrai_tableau_ameliore(image_path: str) -> Optional[dict]`**: Détecte les tableaux par analyse de grille et projections de texte, renvoie la meilleure zone détectée.
- **`detecter_tableau_final(image_path: str, afficher_resultat: bool) -> Optional[str]`**: Combine plusieurs méthodes de détection, sélectionne le meilleur tableau, et sauvegarde l'image cropée.

#### `ocr/lines_column_ocr.py`
- **`ocr_column_lines1(colonne_img: np.ndarray, max_line_gap: int, max_word_gap: int) -> List[str]`**: Extrait les lignes d'une colonne avec docTR, en regroupant les mots proches.
- **`ocr_column_lines2(chemin_image: str, seuil_hauteur_min: int) -> List[str]`**: Extrait le texte par cellule en utilisant la détection de contours.
- **`linesoneortwo(full_img: np.ndarray, med_list: list) -> bool`**: Vérifie l'existence d'une ligne horizontale entre deux médicaments pour déterminer la méthode OCR à utiliser(ocr_column_lines1 ou ocr_column_lines2).
- **`ocr_lines(colonne_img, chemin: str, b: bool) -> List[str]`**: Sélectionne la méthode d'OCR (`ocr_column_lines1` ou `ocr_column_lines2`) selon la présence de lignes.

#### `ocr/text_extraction.py`
- **`extract_text(temp_path: str) -> str`**: Extrait le texte brut d'une image avec docTR.

#### `utils/extract_column.py`
- **`TableColumnExtractor` (class)**:
  - **`__init__(debug_mode: bool)`**: Initialise l'extracteur avec option de debug.
  - **`preprocess_image(image: np.ndarray) -> np.ndarray`**: Prétraitement avec contraste et débruitage.
  - **`detect_table_lines(image: np.ndarray) -> Tuple[List[int], List[int]]`**: Détecte les lignes horizontales et verticales via HoughLinesP ou projection.
  - **`detect_columns_by_projection(gray: np.ndarray) -> List[int]`**: Détecte les colonnes par projection verticale.
  - **`merge_close_lines(lines: List[int], threshold: int) -> List[int]`**: Fusionne les lignes proches.
  - **`visualize_detected_lines(image: np.ndarray, horizontal_lines: List[int], vertical_lines: List[int])`**: Visualise les lignes détectées (debug).
  - **`extract_column_boundaries(image: np.ndarray) -> List[Tuple[int, int]]`**: Extrait les limites des colonnes.
  - **`extract_columns(image_path: str, output_dir: str) -> List[str]`**: Extrait et sauvegarde les colonnes.
- **`get_user_input() -> Tuple[str, str, bool]`**: Demande le chemin de l'image, le dossier de sortie.

#### `utils/text_cleaning.py`
- **`nettoyer_texte_avec_virgule_Depoint_autre(texte: str) -> str`**: Nettoie le texte OCR (remplace tirets, supprime caractères spéciaux, normalise avec spaCy).
- **`nettoyer_et_lemmatiser(texte: str) -> List[spacy.Token]`**: Nettoie et lemmatise le texte avec spaCy.
- **`enlever_accents(texte: str) -> str`**: Supprime les accents des caractères.
- **`enlever_accents_et_apostrophes(text: str) -> str`**: Supprime accents et apostrophes.

#### `utils/med_extraction.py`
- **`extraire_medicaments_depuis_ocr(texte_ocr: str, chemin_csv: str, seuil: int) -> List[dict]`**: Extrait les médicaments du texte OCR en utilisant une correspondance floue avec un CSV.

#### `utils/inclus.py`
- **`inclusion_score1(a: str, b: str) -> bool`**: Vérifie si la chaîne `a` est incluse dans `b` (case-insensitive) pour voir est ce que un médicament existe dans un ligne ou non.

## shema du programme

```mermaid
flowchart TD
    A["Uploader image<br><i>Utilise st.file_uploader pour charger une image (JPG/PNG) via l'interface Streamlit</i>"] --> 
    B["Sauvegarder image temporaire<br><i>Enregistre l'image uploadée dans un fichier temporaire avec tempfile.NamedTemporaryFile, stocke chemin dans temp_path</i>"]

    B --> C["Améliorer segmentation colonnes<br><i>Appelle preprocessing.improve_column_segmentation(temp_path): renforce lignes verticales avec morphologie, met à jour temp_path avec 'enhanced_table.jpg'</i>"]

    C --> D["Détecter tableau<br><i>Appelle table_detection.detecter_tableau_final(temp_path): détecte tableaux via contours/projections, croppe meilleure zone, met à jour temp_path avec 'tableau_crop_ameliore.jpg'</i>"]

    D --> E["Extraire colonnes<br><i>Crée TableColumnExtractor(debug_mode=False), appelle extract_columns(temp_path): détecte lignes verticales, découpe colonnes, retourne liste de chemins PNG</i>"]

    E --> F["Vérifier colonnes extraites<br><i>Si aucune colonne détectée, lève ValueError('Aucune colonne détectée dans le tableau')</i>"]

    F --> G["Extraire texte brut<br><i>Appelle text_extraction.extract_text(temp_path): extrait texte entier de l'image (tableau ou entière) avec docTR</i>"]

    G --> H["Nettoyer texte brut<br><i>Appelle text_cleaning.nettoyer_texte_avec_virgule_Depoint_autre(texte): supprime tirets, caractères spéciaux, normalise avec spaCy, affiche texte_propre</i>"]

    H --> I["Extraire médicaments<br><i>Charge CSV '../projet_structure/nom_de_marque_sans_doublons.csv', appelle med_extraction.extraire_medicaments_depuis_ocr(texte_propre, chemin_csv_medicaments): extrait médicaments via fuzzy matching</i>"]

    I --> J["Afficher médicaments<br><i>Affiche liste des médicaments (nom de marque, infos CSV) avec st.markdown dans cartes 3D</i>"]

    J --> K["Charger image tableau<br><i>Si temp_path est str, charge image avec cv2.imread(temp_path), gère erreurs d'ouverture</i>"]

    K --> L["Vérifier mode OCR<br><i>Appelle lines_column_ocr.linesoneortwo(im, medicaments): vérifie ligne horizontale entre 1ère/2ème occurrences de médicaments, retourne booléen b</i>"]

    L --> M["Traiter chaque colonne<br><i>Pour chaque colonne dans colonnes: vérifie si chemin (str) ou np.ndarray, charge image avec cv2.imread ou crée fichier temporaire, gère erreurs</i>"]

    M --> N["Vérifier validité image colonne<br><i>Vérifie si image.size > 0 et dimensions valides, sinon affiche erreur et passe à la colonne suivante</i>"]

    N --> O["Extraire lignes des colonnes<br><i>Appelle lines_column_ocr.ocr_lines(image, chemin, b): utilise ocr_column_lines1 (lignes regroupées) ou ocr_column_lines2 (cellules), ajoute lignes à lignes_global</i>"]

    O --> P["Filtrer lignes avec médicaments<br><i>Extrait noms_medicaments en minuscules, appelle inclus.inclusion_score1 pour chaque ligne dans lignes_global: vérifie inclusion de nom de médicament, ajoute à lignes_avec_medicaments</i>"]

    P --> Q["Afficher lignes filtrées<br><i>Affiche chaque ligne de lignes_avec_medicaments avec st.markdown dans interface, style 3D</i>"]

    E --> R["Capturer exceptions<br><i>Encapsule étapes depuis détection tableau jusqu'à affichage lignes dans try/except pour gérer erreurs</i>"]

    R --> S["Afficher message d'erreur si exception<br><i>Affiche carte 3D rouge avec st.markdown: détails erreur et suggestion de réessayer</i>"]

    Q --> T["Afficher footer 3D<br><i>Affiche footer stylisé avec st.markdown: thème dynamique, texte 'MediScan 3D - Technologie du Futur', labels Analyse/Médical/Rapide</i>"]
    S --> T

```

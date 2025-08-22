import cv2
import re
import spacy
import numpy as np
from PIL import Image
from transformers import pipeline
import tempfile
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import streamlit as st
from typing import Tuple, List, Optional,Dict
import math
import unicodedata
from rapidfuzz import fuzz
import pandas as pd
import time
from ocr.table_detection import detecter_tableau_final 
from ocr.preprocessing import improve_column_segmentation

from utils.text_cleaning import nettoyer_texte_avec_virgule_Depoint_autre
from utils.extract_column import TableColumnExtractor
from ocr.text_extraction import extract_text
from utils.med_extraction import extraire_medicaments_depuis_ocr
from ocr.lines_column_ocr import linesoneortwo,ocr_lines
from utils.inclus import inclusion_score1
model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31', pretrained=True)
def main():
   # Configuration de la page
    # Début chrono
   start_time = time.time()
   st.set_page_config(
       page_title="🏥 Interface 3D - Extraction Médicaments",
       page_icon="💊",
       layout="wide",
       initial_sidebar_state="expanded"
   )

   # CSS 3D avancé pour interface immersive
   st.markdown("""
   <style>
   @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
   
   .stApp {
       background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 100%);
       color: white;
   }
   
   /* Fond animé avec particules */
   .main::before {
       content: '';
       position: fixed;
       top: 0;
       left: 0;
       width: 100%;
       height: 100%;
       background: 
           radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
           radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
           radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
       animation: backgroundMove 20s ease-in-out infinite;
       z-index: -1;
       pointer-events: none;
   }
   
   @keyframes backgroundMove {
       0%, 100% { transform: scale(1) rotate(0deg); }
       50% { transform: scale(1.1) rotate(5deg); }
   }
   
   /* Header 3D holographique */
   .header-3d {
       background: rgba(255, 255, 255, 0.05);
       backdrop-filter: blur(30px);
       border-radius: 30px;
       border: 2px solid;
       border-image: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4) 1;
       padding: 40px;
       margin: 20px 0;
       text-align: center;
       transform: perspective(1000px) rotateX(10deg) translateZ(50px);
       box-shadow: 
           0 30px 60px rgba(0, 0, 0, 0.4),
           inset 0 1px 0 rgba(255, 255, 255, 0.2),
           0 0 50px rgba(77, 208, 225, 0.3);
       animation: headerFloat 8s ease-in-out infinite;
       position: relative;
       overflow: hidden;
   }
   
   .header-3d::before {
       content: '';
       position: absolute;
       top: 0;
       left: -100%;
       width: 100%;
       height: 100%;
       background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
       animation: shine 3s infinite;
   }
   
   @keyframes headerFloat {
       0%, 100% { transform: perspective(1000px) rotateX(10deg) translateZ(50px) translateY(0px); }
       50% { transform: perspective(1000px) rotateX(10deg) translateZ(50px) translateY(-10px); }
   }
   
   @keyframes shine {
       0% { left: -100%; }
       100% { left: 100%; }
   }
   
   .title-3d {
       font-family: 'Orbitron', monospace;
       font-size: 3.5rem;
       font-weight: 900;
       background: linear-gradient(45deg, #4dd0e1, #26c6da, #00bcd4, #0097a7);
       -webkit-background-clip: text;
       -webkit-text-fill-color: transparent;
       background-clip: text;
       text-shadow: 0 0 30px rgba(77, 208, 225, 0.5);
       margin-bottom: 20px;
       animation: glow 2s ease-in-out infinite alternate;
   }
   
   @keyframes glow {
       from { text-shadow: 0 0 30px rgba(77, 208, 225, 0.5); }
       to { text-shadow: 0 0 50px rgba(77, 208, 225, 0.8), 0 0 70px rgba(77, 208, 225, 0.6); }
   }
   
   /* Cartes 3D flottantes */
   .card-3d {
       background: rgba(255, 255, 255, 0.08);
       backdrop-filter: blur(25px);
       border-radius: 20px;
       border: 1px solid rgba(255, 255, 255, 0.15);
       padding: 30px;
       margin: 20px 0;
       transform: perspective(1000px) rotateX(5deg) rotateY(-2deg) translateZ(20px);
       transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
       box-shadow: 
           0 20px 40px rgba(0, 0, 0, 0.3),
           inset 0 1px 0 rgba(255, 255, 255, 0.1);
       position: relative;
       overflow: hidden;
   }
   
   .card-3d:hover {
       transform: perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(40px) scale(1.02);
       box-shadow: 
           0 30px 60px rgba(0, 0, 0, 0.4),
           0 0 50px rgba(77, 208, 225, 0.2);
   }
   
   .card-3d::after {
       content: '';
       position: absolute;
       top: 0;
       left: 0;
       right: 0;
       bottom: 0;
       background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
       opacity: 0;
       transition: opacity 0.3s ease;
   }
   
   .card-3d:hover::after {
       opacity: 1;
   }
   
   /* Zone d'upload 3D */
   .upload-zone-3d {
       background: rgba(255, 255, 255, 0.03);
       backdrop-filter: blur(20px);
       border: 3px dashed rgba(77, 208, 225, 0.5);
       border-radius: 25px;
       padding: 60px;
       text-align: center;
       transform: perspective(1000px) rotateX(8deg) translateZ(30px);
       transition: all 0.4s ease;
       position: relative;
       overflow: hidden;
   }
   
   .upload-zone-3d:hover {
       transform: perspective(1000px) rotateX(0deg) translateZ(50px);
       border-color: rgba(77, 208, 225, 0.8);
       background: rgba(77, 208, 225, 0.05);
       box-shadow: 0 0 50px rgba(77, 208, 225, 0.3);
   }
   
   .upload-icon {
       font-size: 4rem;
       color: #4dd0e1;
       margin-bottom: 20px;
       animation: pulse 2s ease-in-out infinite;
       text-shadow: 0 0 20px rgba(77, 208, 225, 0.5);
   }
   
   @keyframes pulse {
       0%, 100% { transform: scale(1); }
       50% { transform: scale(1.1); }
   }
   
   /* Médicament card 3D */
   .medication-card-3d {
       background: rgba(76, 175, 80, 0.1);
       backdrop-filter: blur(20px);
       border-radius: 15px;
       border: 2px solid rgba(76, 175, 80, 0.3);
       padding: 25px;
       margin: 15px 0;
       transform: perspective(800px) rotateY(-5deg) translateZ(25px);
       transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
       box-shadow: 
           0 15px 30px rgba(0, 0, 0, 0.2),
           0 0 30px rgba(76, 175, 80, 0.2);
       position: relative;
       overflow: hidden;
   }
   
   .medication-card-3d:hover {
       transform: perspective(800px) rotateY(0deg) translateZ(40px) scale(1.03);
       border-color: rgba(76, 175, 80, 0.6);
       box-shadow: 
           0 25px 50px rgba(0, 0, 0, 0.3),
           0 0 40px rgba(76, 175, 80, 0.4);
   }
   
   .medication-card-3d::before {
       content: '';
       position: absolute;
       top: 0;
       left: 0;
       right: 0;
       height: 3px;
       background: linear-gradient(90deg, #4caf50, #8bc34a, #cddc39);
       animation: progress 3s ease-in-out infinite;
   }
   
   @keyframes progress {
       0%, 100% { transform: translateX(-100%); }
       50% { transform: translateX(100%); }
   }
   
   /* Metrics 3D */
   .metric-3d {
       background: rgba(255, 255, 255, 0.05);
       backdrop-filter: blur(25px);
       border-radius: 20px;
       border: 2px solid rgba(255, 255, 255, 0.1);
       padding: 30px;
       text-align: center;
       transform: perspective(600px) rotateX(10deg) translateZ(30px);
       transition: all 0.4s ease;
       box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
       position: relative;
   }
   
   .metric-3d:hover {
       transform: perspective(600px) rotateX(0deg) translateZ(50px) scale(1.05);
       box-shadow: 0 30px 60px rgba(0, 0, 0, 0.3);
   }
   
   .metric-value {
       font-family: 'Orbitron', monospace;
       font-size: 3rem;
       font-weight: 700;
       color: #4dd0e1;
       text-shadow: 0 0 20px rgba(77, 208, 225, 0.5);
       animation: countUp 2s ease-out;
   }
   
   @keyframes countUp {
       from { transform: scale(0) rotate(180deg); }
       to { transform: scale(1) rotate(0deg); }
   }
   
   /* Sidebar 3D */
   .css-1d391kg {
       background: rgba(0, 0, 0, 0.8) !important;
       backdrop-filter: blur(20px) !important;
   }
   
   /* Boutons 3D */
   .stButton > button {
       background: linear-gradient(135deg, #4dd0e1, #26c6da) !important;
       color: white !important;
       border: none !important;
       border-radius: 25px !important;
       padding: 15px 30px !important;
       font-weight: bold !important;
       font-size: 1.1rem !important;
       transform: perspective(500px) translateZ(10px) !important;
       transition: all 0.3s ease !important;
       box-shadow: 0 10px 20px rgba(77, 208, 225, 0.3) !important;
   }
   
   .stButton > button:hover {
       transform: perspective(500px) translateZ(20px) scale(1.05) !important;
       box-shadow: 0 20px 40px rgba(77, 208, 225, 0.4) !important;
   }
   
   /* Animation de chargement 3D */
   .loading-3d {
       display: flex;
       justify-content: center;
       align-items: center;
       height: 100px;
   }
   
   .spinner-3d {
       width: 50px;
       height: 50px;
       border: 3px solid rgba(77, 208, 225, 0.3);
       border-top: 3px solid #4dd0e1;
       border-radius: 50%;
       animation: spin3d 1s linear infinite;
       box-shadow: 0 0 30px rgba(77, 208, 225, 0.5);
   }
   
   @keyframes spin3d {
       0% { transform: rotateY(0deg) rotateX(0deg); }
       100% { transform: rotateY(360deg) rotateX(360deg); }
   }
   
   /* Effets de texte */
   .neon-text {
       color: #4dd0e1;
       text-shadow: 
           0 0 5px #4dd0e1,
           0 0 10px #4dd0e1,
           0 0 20px #4dd0e1,
           0 0 40px #4dd0e1;
       animation: flicker 2s infinite alternate;
   }
   
   @keyframes flicker {
       0%, 100% { opacity: 1; }
       50% { opacity: 0.8; }
   }
   
   /* Scrollbar 3D */
   ::-webkit-scrollbar {
       width: 12px;
   }
   
   ::-webkit-scrollbar-track {
       background: rgba(255, 255, 255, 0.1);
       border-radius: 10px;
   }
   
   ::-webkit-scrollbar-thumb {
       background: linear-gradient(135deg, #4dd0e1, #26c6da);
       border-radius: 10px;
       box-shadow: 0 0 10px rgba(77, 208, 225, 0.5);
   }
   
   ::-webkit-scrollbar-thumb:hover {
       background: linear-gradient(135deg, #26c6da, #4dd0e1);
   }
   
   /* Toast notifications 3D */
   .stToast {
       background: rgba(255, 255, 255, 0.1) !important;
       backdrop-filter: blur(20px) !important;
       border-radius: 15px !important;
       border: 1px solid rgba(255, 255, 255, 0.2) !important;
   }

   /* Couleurs thématiques dynamiques */
   .theme-cyan { --primary-color: #4dd0e1; --secondary-color: #26c6da; }
   .theme-green { --primary-color: #4caf50; --secondary-color: #8bc34a; }
   .theme-violet { --primary-color: #9c27b0; --secondary-color: #e91e63; }
   .theme-orange { --primary-color: #ff9800; --secondary-color: #ff5722; }
   </style>
   """, unsafe_allow_html=True)

   # ===================== SIDEBAR 3D OPTIMISÉ =====================
   with st.sidebar:
       st.markdown("""
       <div class="card-3d">
           <h2 class="neon-text">⚙️ Contrôles 3D</h2>
       </div>
       """, unsafe_allow_html=True)
       
       # ========== Options d'affichage ==========
       st.markdown("### 📊 Affichage")
       show_metrics = st.checkbox("Métriques 3D", True)
       show_columns = st.checkbox("Détail des colonnes", True)
       show_raw_text = st.checkbox("Texte brut", False)
       
       # ========== Thème visuel ==========
       st.markdown("### 🎨 Thème")
       theme_colors = {
           "Cyan": {"primary": "#4dd0e1", "secondary": "#26c6da"},
           "Vert": {"primary": "#4caf50", "secondary": "#8bc34a"},
           "Violet": {"primary": "#9c27b0", "secondary": "#e91e63"},
           "Orange": {"primary": "#ff9800", "secondary": "#ff5722"}
       }
       
       theme_color = st.selectbox("Couleur principale", list(theme_colors.keys()))
       selected_theme = theme_colors[theme_color]
       
       # Application dynamique du thème
       st.markdown(f"""
       <style>
       :root {{
           --theme-primary: {selected_theme["primary"]};
           --theme-secondary: {selected_theme["secondary"]};
       }}
       .title-3d {{
           background: linear-gradient(45deg, {selected_theme["primary"]}, {selected_theme["secondary"]}) !important;
           -webkit-background-clip: text !important;
           -webkit-text-fill-color: transparent !important;
       }}
       .neon-text {{
           color: {selected_theme["primary"]} !important;
           text-shadow: 0 0 20px {selected_theme["primary"]} !important;
       }}
       </style>
       """, unsafe_allow_html=True)
       
       # ========== Informations d'aide ==========
       st.markdown("""
       <div class="card-3d" style="margin-top: 30px;">
           <h4 class="neon-text">💡 Guide d'utilisation</h4>
           <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem; line-height: 1.6;">
               <p><strong>1. Upload :</strong> Glissez votre ordonnance</p>
               <p><strong>2. Visualisation :</strong> Activez/désactivez les options</p>
               <p><strong>3. Thème :</strong> Personnalisez l'interface</p>
           </div>
       </div>
       """, unsafe_allow_html=True)

   # Header principal 3D
   st.markdown("""
   <div class="header-3d">
       <div class="title-3d">🏥 MediScan 3D</div>
       <p style="font-size: 1.3rem; color: rgba(255, 255, 255, 0.8); margin-bottom: 10px;">
           Intelligence Artificielle • Reconnaissance Optique • Analyse Médicale
       </p>
       <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
           <span style="padding: 8px 16px; background: rgba(76, 175, 80, 0.2); border-radius: 20px; border: 1px solid rgba(76, 175, 80, 0.5);">
               ✅ OCR Avancé
           </span>
           <span style="padding: 8px 16px; background: rgba(33, 150, 243, 0.2); border-radius: 20px; border: 1px solid rgba(33, 150, 243, 0.5);">
               🧠 IA Médicale
           </span>
           <span style="padding: 8px 16px; background: rgba(255, 152, 0, 0.2); border-radius: 20px; border: 1px solid rgba(255, 152, 0, 0.5);">
               🔬 Analyse 3D
           </span>
       </div>
   </div>
   """, unsafe_allow_html=True)

   # Zone d'upload 3D interactive
   #---------------------------------------------------------------
   col1, col2, col3 = st.columns([1, 2, 1])
   with col2:
        st.markdown(f"""
        <div class="upload-zone-3d" style="
            border: 2px dashed {selected_theme['primary']};
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            font-family: 'Orbitron', monospace;
            color: {selected_theme['primary']};
            margin: 20px auto;  /* marge top/bottom + centrer horizontalement */
            texte-align:center;
            width: 600px;
        ">
            <div class="upload-icon" style="font-size: 48px;">📤</div>
            <h3 style="margin-bottom: 15px;">Zone de Téléchargement 3D</h3>
            <p style="color: rgba(255, 255, 255, 0.7);">
                Glissez votre ordonnance ici
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            label="", 
            type=['jpg', 'jpeg', 'webp', 'png'], 
            label_visibility="collapsed",
            key="uploader3d"
        )

        # Appliquer les mêmes marges au uploader et fixer la largeur à 600px
        st.markdown("""
        <style>
        /* cibler la div contenant le file uploader */
        div[data-testid="stFileUploader"] > div:first-child {
            width: 600px !important;
            margin: 20px auto !important;  /* même margin que la zone */
            display: block;
        }
        div[data-testid="stFileUploader"] {
            margin-left: 2px !important;
            width:590px;
        }
        </style>
        """, unsafe_allow_html=True)


   if uploaded_file:
       # Sauvegarde et traitement initial
       temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)


       if os.path.exists(temp_path):
            os.remove(temp_path)

       with open(temp_path, "wb") as f:
           f.write(uploaded_file.getbuffer())

       # Affichage de l'image avec effet 3D
       st.markdown("---")
       col1, col2, col3 = st.columns([1, 3, 1])
       with col2:
           st.markdown(f"""
           <div class="card-3d" style="border-color: {selected_theme['primary']}30;">
               <h3 class="neon-text" style="text-align: center; margin-bottom: 20px; color: {selected_theme['primary']};">
                   📋 Ordonnance Analysée
               </h3>
           </div>
           """, unsafe_allow_html=True)
           
           image = Image.open(uploaded_file)
           st.image(image, use_container_width=True, caption="Image en cours d'analyse...")

       # Animation de chargement 3D
       progress_container = st.empty()
       status_container = st.empty()
       
       with progress_container.container():
           st.markdown(f"""
           <div class="loading-3d">
               <div class="spinner-3d" style="border-top-color: {selected_theme['primary']};"></div>
           </div>
           """, unsafe_allow_html=True)
       
       # Simulation du processus
       steps = [
           ("🔍 Initialisation de l'OCR...", 0.25),
           ("📝 Extraction du texte...", 0.5),
           ("🧹 Nettoyage des données...", 0.75),
           ("💊 Identification des médicaments...", 1.0)
       ]
       
       progress_bar = st.progress(0)
       for step_text, progress in steps:
           status_container.info(step_text)
           progress_bar.progress(progress)
           time.sleep(0.6)
       
       progress_container.empty()
       status_container.empty()

       # Traitement réel
       try:
           temp_path = improve_column_segmentation(temp_path)
           temp_path=detecter_tableau_final(temp_path)
           # Étape 2 : Extraire les colonnes avec TableColumnExtractor
           extractor = TableColumnExtractor(debug_mode=False)  # Activer debug_mode=True pour visualisations
           colonnes = extractor.extract_columns(temp_path)
           if not colonnes:
                raise ValueError("Aucune colonne détectée dans le tableau.")
           lignes_global=[]
           texte=extract_text(temp_path)
           texte_propre = nettoyer_texte_avec_virgule_Depoint_autre(texte)
           print(texte_propre)
           # Extraction des médicaments
           chemin_csv_medicaments = "../projet_structure/nom_de_marque_sans_doublons.csv"
           medicaments = extraire_medicaments_depuis_ocr(texte_propre, chemin_csv_medicaments)
           st.markdown(medicaments)
           if isinstance(temp_path, str):
                    try:
                        im = cv2.imread(temp_path)
                    except Exception as e:
                        print(f"Erreur lors du chargement de l'image de la colonne ")
           b=linesoneortwo(im,medicaments)
           for i,col_img in enumerate(colonnes, 1):
                # Vérifier si col_img est une chaîne (chemin de fichier)
                if isinstance(col_img, str):
                    try:
                        image = cv2.imread(col_img)
                        if image is None:
                            print(f"Erreur: Impossible de charger l'image de la colonne #{i} à partir de {col_img}")
                            continue
                    except Exception as e:
                        print(f"Erreur lors du chargement de l'image de la colonne #{i}: {e}")
                        continue
                    chemin = col_img  # Utiliser le chemin original
                elif isinstance(col_img, np.ndarray):
                    image = col_img
                    # Générer un chemin temporaire si nécessaire
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        cv2.imwrite(tmpfile.name, image)
                        chemin = tmpfile.name
                else:
                    print(f"Erreur: col_img #{i} n'est ni un chemin de fichier ni un tableau NumPy, type: {type(col_img)}")
                    continue

                # Vérifier que l'image est valide
                if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                    print(f"Erreur: Image de la colonne #{i} est vide ou invalide: {image.shape}")
                    continue

                # Appeler ocr_lines avec l'image et le chemin
                lignes = ocr_lines(image, chemin,b)
                if lignes:
                    for ligne in lignes:
                        lignes_global.append(ligne)
                    
           
           noms_medicaments = [med["nom_de_marque"].lower() for med in medicaments]
           # Vérification des lignes contenant un médicament
           print(lignes_global)
           lignes_avec_medicaments = []
           for ligne in lignes_global:
                ligne_lower = ligne.lower()
                for nom in noms_medicaments:
                    if inclusion_score1(nom,ligne_lower)==True:  # Seuil de correspondance floue
                        lignes_avec_medicaments.append((ligne))
                        break
           for l in lignes_avec_medicaments:
                st.markdown(f"{l}")       
       except Exception as e:
           st.markdown(f"""
           <div class="card-3d" style="border-color: rgba(244, 67, 54, 0.5);">
               <h3 style="color: #f44336;">❌ Erreur de Traitement</h3>
               <p style="color: rgba(255,255,255,0.8);">
                   Une erreur s'est produite lors de l'analyse: {str(e)}
               </p>
               <div style="margin-top: 20px; padding: 15px; background: rgba(244, 67, 54, 0.1); border-radius: 10px;">
                   <p style="color: rgba(255,255,255,0.7);">
                       Veuillez réessayer avec une autre image.
                   </p>
               </div>
           </div>
           """, unsafe_allow_html=True)

   # Footer 3D avec thème dynamique
   st.markdown("---")
   st.markdown(f"""
   <div class="card-3d" style="text-align: center; margin-top: 50px; border-color: {selected_theme['primary']}20;">
       <h4 class="neon-text" style="color: {selected_theme['primary']};">🚀 MediScan 3D - Technologie du Futur</h4>
       <p style="color: rgba(255,255,255,0.6); margin-top: 15px;">
           Propulsé par l'Intelligence Artificielle • OCR Avancé • Interface 3D Immersive
       </p>
       <div style="margin-top: 20px;">
           <span style="margin: 0 10px; color: {selected_theme['primary']};">🔬 Analyse</span>
           <span style="margin: 0 10px; color: {selected_theme['secondary']};">🏥 Médical</span>
           <span style="margin: 0 10px; color: #ff9800;">⚡ Rapide</span>
       </div>
       <div style="margin-top: 15px; font-size: 0.9rem; color: rgba(255,255,255,0.5);">
           Thème actuel: {theme_color}
       </div>
   </div>
   """, unsafe_allow_html=True)

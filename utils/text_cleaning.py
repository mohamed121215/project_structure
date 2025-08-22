import re
import spacy
from doctr.models import ocr_predictor
import unicodedata
model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31', pretrained=True)
nlp = spacy.load("fr_core_news_sm")
def nettoyer_texte_avec_virgule_Depoint_autre(texte):
    # Nettoyage OCR de base
    texte = texte.replace("–", "-").replace("—", "-")
    texte=re.sub(r"[-_]"," ",texte)
    texte = re.sub(r"[@#;]+", " ", texte)
    texte = re.sub(r"[°]+", "*", texte)
    # On garde uniquement les caractères ASCII + lettres accentuées + espaces + , et :
    texte = re.sub(r"[^\x00-\x7FÀ-ÿ\s,:*/%]", "", texte)
    texte = re.sub(r"\s+", " ", texte)
    # Ne pas supprimer les espaces avant , et :
    texte = re.sub(r"\s+([*.,;:!?/%'])", r"\1", texte)

    # Nettoyage final
    # On supprime tout sauf lettres, chiffres, espaces, virgules et deux-points
    texte = re.sub(r"[^\wÀ-ÿ\s,:*/%']", "", texte)
    texte = texte.lower()

    # NLP spaCy
    doc = nlp(texte)
    texte_final = " ".join([token.text for token in doc])
    return texte_final
def nettoyer_et_lemmatiser(texte):
    doc = nlp(texte.lower())
    return [token for token in doc if not token.is_stop and not token.is_punct]


def enlever_accents(texte):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texte)
        if unicodedata.category(c) != 'Mn'
    )


def enlever_accents_et_apostrophes(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r"[’']", "", text)
    return text
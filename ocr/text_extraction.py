
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model = ocr_predictor(det_arch='db_resnet50', reco_arch='sar_resnet31', pretrained=True)
def extract_text(temp_path):
        img=DocumentFile.from_images(temp_path)
        result=model(img)
        texte = "\n".join(
            " ".join(word.value for word in line.words)
            for block in result.pages[0].blocks
            for line in block.lines
        )
        return texte
import io
import os
import zlib
import base64
import zipfile
from typing import List
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

# ==========================================
# 1. TON ARCHITECTURE AUTOENCODEUR
# ==========================================
class Encodeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.reseau(x)

class Decodeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.reseau(z)

class AutoencodeurCompressif(nn.Module):
    def __init__(self):
        super().__init__()
        self.encodeur = Encodeur()
        self.decodeur = Decodeur()

# Chargement du modèle au démarrage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modele = AutoencodeurCompressif().to(device)
if os.path.exists("komprssor_weights.pth"):
    modele.load_state_dict(torch.load("komprssor_weights.pth", map_location=device))
modele.eval()

# ==========================================
# 2. CONFIGURATION DE L'API WEB
# ==========================================
app = FastAPI(title="Komprssor-img")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def lire_accueil():
    return FileResponse("static/index.html")

@app.post("/api/compress")
async def compresser_images(files: List[UploadFile] = File(...)):
    resultats = []

    for fichier in files:
        contenu = await fichier.read()
        img_pil = Image.open(io.BytesIO(contenu)).convert("RGB")
        largeur_orig, hauteur_orig = img_pil.size

        # Calcul du poids brut non compressé (référence BMP)
        buf_bmp = io.BytesIO()
        img_pil.save(buf_bmp, format="BMP")
        taille_brute = len(buf_bmp.getvalue())

        # Préparation du tenseur et padding (multiple de 4)
        tenseur = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
        pad_h = (4 - hauteur_orig % 4) % 4
        pad_w = (4 - largeur_orig % 4) % 4
        tenseur_padded = F.pad(tenseur, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            # Encodage + Quantification uint8 + Zlib
            z = modele.encodeur(tenseur_padded)
            z_entiers = torch.round(z * 255.0).to(torch.uint8)
            octets_bruts = z_entiers.cpu().numpy().tobytes()
            octets_bin = zlib.compress(octets_bruts, level=9)

            # Décodage pour aperçu et téléchargement image
            z_q = z_entiers.float() / 255.0
            reconst_padded = modele.decodeur(z_q)
            reconst = reconst_padded[:, :, :hauteur_orig, :largeur_orig]

        # Conversion de l'image reconstruite en PNG (optimisé) pour l'affichage et téléchargement
        img_out = transforms.ToPILImage()(reconst.squeeze(0).cpu())
        buf_png = io.BytesIO()
        img_out.save(buf_png, format="PNG", optimize=True)
        png_bytes = buf_png.getvalue()

        taille_bin = len(octets_bin)
        ratio = round(taille_brute / max(taille_bin, 1), 2)

        resultats.append({
            "filename": fichier.filename,
            "dimensions": f"{largeur_orig}×{hauteur_orig}",
            "raw_size_kb": round(taille_brute / 1024, 1),
            "bin_size_kb": round(taille_bin / 1024, 1),
            "ratio": ratio,
            "preview_base64": "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8"),
            "bin_base64": base64.b64encode(octets_bin).decode("utf-8")
        })

    return {"results": resultats}

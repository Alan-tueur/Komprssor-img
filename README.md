# Komprssor-img
compresseur d'image base sur la technologie d'auto-encodage utilisee en Deep-Learning


**Komprssor-img** est une application web moderne permettant de compresser des images (PNG, JPG, WEBP) grâce à un **autoencodeur convolutif profond** couplé à une quantification 8 bits et un codage entropique.

---

## ✨ Fonctionnalités

- **Compression par Intelligence Artificielle** : Réduction spatiale et représentation latente apprise par réseau de neurones convolutif (PyTorch).
- **Compression réelle en octets** : Quantification en entiers `uint8` (avec *Straight-Through Estimator* lors de l'entraînement) et compression binaire sans perte via `zlib`.
- **Support haute résolution** : Architecture entièrement convolutive capable de traiter des images de n'importe quelle dimension grâce à un remplissage (*padding*) dynamique.
- **Interface web épurée** : Zone *Glisser-Déposer* intuitive, aperçu avant/après, calcul du ratio de compression en temps réel.
- **Export flexible** : Téléchargement individuel (image reconstruite `.png` ou flux binaire `.bin`) ou groupé en archive `.zip`.

---

## 🧠 Architecture & Fonctionnement

Le pipeline de compression suit quatre étapes :

```text
Image (3 × H × W)
       │
       ▼
┌──────────────────┐
│     ENCODEUR     │  Convolutions stride=2 -> Réduction spatiale (16 × H/4 × W/4)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  QUANTIFICATION  │  Conversion des flottants [0, 1] en entiers 8 bits (uint8)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│ CODAGE ENTROPIQUE│  Compression binaire sans perte (zlib) -> Fichier .bin
└──────────────────┘
       │
       ▼
┌──────────────────┐
│     DÉCODEUR     │  Convolutions transposées -> Image reconstruite (3 × H × W)
└──────────────────┘
```

---

## 📂 Structure du projet

```text
Komprssor-img/
├── app.py                  # API FastAPI et moteur d'inférence PyTorch
├── komprssor_weights.pth   # Poids du modèle autoencodeur entraîné
├── requirements.txt        # Dépendances Python
├── README.md               # Documentation du projet
└── static/
    └── index.html          # Interface utilisateur (HTML5, CSS3, JS + JSZip)
```

---

## 🚀 Installation et exécution en local

### 1. Cloner le dépôt
```bash
git clone https://github.com
cd Komprssor-img
```

### 2. Créer un environnement virtuel et installer les dépendances
```bash
python -m venv venv
source venv/bin/activate  # Sous Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Lancer le serveur web
```bash
uvicorn app:app --reload
```
Ouvrez ensuite votre navigateur à l'adresse : **`http://127.0.0.1:8000`**

---

## 🛠️ Technologies utilisées

- **Deep Learning** : PyTorch, Torchvision
- **Backend & API** : FastAPI, Uvicorn, Pillow
- **Frontend** : HTML5, CSS3 (Google Fonts *Plus Jakarta Sans*), JavaScript Vanilla, JSZip

---

## 👤 Auteur

Développé par **Kabrel** — Projet d'application pratique des réseaux de neurones autoencodeurs à la compression d'images.

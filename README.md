# 📸 Local AI Photo Sorter

Automatically classify and sort your photos into folders using CLIP — no cloud, no subscriptions, runs fully local on your GPU.

![Python](https://img.shields.io/badge/python-3.11-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## ✨ Features

- 🔒 **100% local** — your photos never leave your machine
- ⚡ **GPU accelerated** — uses CUDA for fast inference
- 📂 **8 categories** — people, documents, memes, screenshots, animals, landscapes, food, other
- 🔁 **Resume support** — skips already processed files automatically
- 🗑️ **Duplicate removal** — finds and removes dupes by file hash, not just filename
- 🖼️ **HEIC support** — handles iPhone photos natively

---

## 🗂️ Categories

| Folder | What goes there |
|---|---|
| `people` | Portraits, selfies, group photos |
| `docs` | Documents, receipts, passports, text screenshots |
| `memes` | Memes, funny images with text |
| `screenshots` | App, game, desktop, chat screenshots |
| `animals` | Cats, dogs, wildlife, pets |
| `landscapes` | Nature, mountains, sea, sky |
| `food` | Meals, drinks, food photos |
| `other` | Everything else |

---

## 🛠️ Requirements

- Python 3.11
- NVIDIA GPU with 4GB+ VRAM (CPU works too, but slow)
- CUDA 12.1

---

## 🚀 Installation

**1. Clone the repo**
```bash
git clone https://github.com/danarieel/local-photo-sorter
cd local-photo-sorter
```

**2. Install PyTorch with CUDA**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Install other dependencies**
```bash
pip install open-clip-torch pillow-heif Pillow tqdm
```

---

## 📖 Usage

```bash
python main.py
```

An interactive menu will appear:

```
========================================
   Local AI Photo Sorter
   CLIP ViT-L/14 | fully local
========================================

  1. Sort photos
  2. Find missing files
  3. Remove duplicates
  0. Exit
```

**1. Sort photos** — classifies all photos from input folder and copies them into categorized subfolders. Originals are untouched.

~6 photos/sec on RTX 2060 Super. 5000 photos ≈ 15 minutes.

**2. Find missing files** — if some files failed on the first run, this option finds and processes them.

**3. Remove duplicates** — finds exact duplicates by MD5 hash, shows what it found, then asks for confirmation before deleting anything.

---

## 🧠 How it works

The project uses **OpenAI CLIP ViT-L/14** — a vision-language model that understands images by comparing them to text descriptions.

For each photo:
1. Image is encoded into a feature vector by CLIP's image encoder
2. Each category has multiple text prompts (e.g. *"a selfie"*, *"a portrait of a human"*, *"a group of people"*)
3. Prompts are averaged into a single category vector
4. The photo is assigned to the most similar category by cosine similarity

Photos are processed in batches of 16 for maximum GPU throughput.

---

## 📁 Project Structure

```
local-photo-sorter/
├── main.py       # classifier, missing file finder, duplicate remover
├── README.md
└── LICENSE
```

---

## ⚙️ Tested on

- Windows 10 LTSC
- Ryzen 5 5600 + RTX 2060 Super (8GB VRAM)
- Python 3.11 / CUDA 12.1
- ~5000 photos including JPG, PNG, HEIC, WebP

---

## 📝 Notes

- CLIP works best with real photos. Very abstract or ambiguous images may land in `other`
- The model (~1.7GB) is downloaded automatically on first run and cached locally
- To run on CPU only, just remove the CUDA torch install step — it will fall back automatically

---

## 📄 License

MIT
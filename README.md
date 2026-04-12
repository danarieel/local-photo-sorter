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
- 🔁 **Resume support** — retry scripts skip already processed files
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
git clone https://github.com/yourusername/local-photo-sorter
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

### Step 1 — Sort your photos
```bash
python sort_photos_clip_large.py --input "C:/Photos" --output "C:/Sorted"
```
Copies photos from `--input` into categorized subfolders in `--output`. Originals are untouched.

~6 photos/sec on RTX 2060 Super. 5000 photos ≈ 15 minutes.

---

### Step 2 — Fix any failed files (optional)
If some files failed (e.g. HEIC not supported), run:
```bash
python find_missing.py --input "C:/Photos" --output "C:/Sorted"
```
Finds files that didn't make it into the output and processes them.

---

### Step 3 — Remove duplicates (optional)
```bash
python remove_dupes.py --folder "C:/Sorted"
```
Finds exact duplicates by MD5 hash, shows you what it found, then asks for confirmation before deleting anything.

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
├── sort_photos_clip_large.py   # Main classifier
├── find_missing.py             # Retry failed files
├── remove_dupes.py             # Remove duplicate files
├── requirements.txt
└── README.md
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

"""
Local AI Photo Sorter
Classify and sort photos using CLIP ViT-L/14, fully local on GPU.

Usage:
    python main.py
"""

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
import pillow_heif
from tqdm import tqdm
import torch
import open_clip

pillow_heif.register_heif_opener()

# ─── Config ───────────────────────────────────────────────────────────────────

CATEGORIES = {
    "people": [
        "a photo of a person",
        "a selfie",
        "a portrait of a human",
        "a group of people",
        "a photo of a man or woman",
        "a photo of a face",
        "people at a party",
        "a photo of friends",
    ],
    "docs": [
        "a photo of a document",
        "a scanned document",
        "a photo of a receipt",
        "a photo of a passport or ID",
        "a screenshot of text",
        "a photo of a contract or paper",
        "a photo of a certificate",
        "handwritten notes",
    ],
    "memes": [
        "an internet meme",
        "a funny image with text",
        "a meme with a caption",
        "a humorous image macro",
        "a reaction meme",
        "a funny screenshot with text overlay",
        "a viral meme image",
    ],
    "screenshots": [
        "a screenshot of a mobile app",
        "a screenshot of a phone screen",
        "a screenshot of a computer desktop",
        "a screenshot of a video game",
        "a screenshot of a chat or messenger",
        "a screenshot of a website",
        "a screenshot of social media",
    ],
    "animals": [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of an animal",
        "a photo of a pet",
        "wildlife photography",
        "a photo of a bird",
        "animals in nature",
    ],
    "landscapes": [
        "a landscape photo",
        "a photo of nature",
        "a photo of mountains",
        "a photo of the sea or ocean",
        "a photo of a forest",
        "a photo of the sky",
        "a scenic outdoor view",
        "a sunset or sunrise photo",
    ],
    "food": [
        "a photo of food",
        "a photo of a meal",
        "a photo of a dish",
        "a photo of a drink",
        "food photography",
        "a photo of a restaurant meal",
        "a photo of street food",
    ],
    "other": [
        "an abstract image",
        "a random object",
        "a photo of a building",
        "a photo of a car",
        "a photo of furniture",
        "a miscellaneous photo",
        "an unidentified image",
    ],
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".heif"}
BATCH_SIZE = 16

# ─── CLIP ─────────────────────────────────────────────────────────────────────

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")
    print("[*] Loading CLIP ViT-L/14...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, preprocess, tokenizer, device


def encode_prompts(model, tokenizer, device):
    all_prompts, prompt_to_cat = [], []
    for cat, prompts in CATEGORIES.items():
        for p in prompts:
            all_prompts.append(p)
            prompt_to_cat.append(cat)

    tokens = tokenizer(all_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    cat_names = list(CATEGORIES.keys())
    cat_features = []
    for cat in cat_names:
        indices = [i for i, c in enumerate(prompt_to_cat) if c == cat]
        feat = text_features[indices].mean(dim=0)
        feat /= feat.norm()
        cat_features.append(feat)

    return torch.stack(cat_features), cat_names


def classify_batch(tensors, model, cat_features, device):
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        image_features = model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarities = image_features @ cat_features.T
    return similarities.argmax(dim=-1).cpu().tolist()


def run_classification(input_dir, output_dir, model, preprocess, cat_features, cat_names, device, files):
    for cat in cat_names:
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    stats  = {cat: 0 for cat in cat_names}
    errors = 0
    batch_files, batch_tensors = [], []

    def flush(batch_files, batch_tensors):
        nonlocal errors
        indices = classify_batch(batch_tensors, model, cat_features, device)
        for file, idx in zip(batch_files, indices):
            category = cat_names[idx]
            dest = output_dir / category / file.name
            if dest.exists():
                dest = output_dir / category / f"{file.stem}_{file.stat().st_ino}{file.suffix}"
            try:
                shutil.copy2(file, dest)
                stats[category] += 1
            except Exception:
                errors += 1

    with tqdm(total=len(files), unit="photo") as pbar:
        for file in files:
            try:
                img = preprocess(Image.open(file).convert("RGB"))
                batch_files.append(file)
                batch_tensors.append(img)
            except Exception:
                errors += 1
                pbar.update(1)
                continue

            if len(batch_tensors) == BATCH_SIZE:
                flush(batch_files, batch_tensors)
                pbar.update(len(batch_files))
                batch_files, batch_tensors = [], []

        if batch_tensors:
            flush(batch_files, batch_tensors)
            pbar.update(len(batch_files))

    print("\n=== Done ===")
    for cat, count in stats.items():
        print(f"  {cat:15s}: {count}")
    if errors:
        print(f"  Failed: {errors}")

# ─── Menu actions ─────────────────────────────────────────────────────────────

def action_sort():
    input_dir  = Path(input("Input folder (with photos): ").strip())
    output_dir = Path(input("Output folder: ").strip())

    if not input_dir.exists():
        print(f"[!] Not found: {input_dir}")
        return

    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    print(f"[*] Found: {len(files)} files")

    model, preprocess, tokenizer, device = load_model()
    cat_features, cat_names = encode_prompts(model, tokenizer, device)
    run_classification(input_dir, output_dir, model, preprocess, cat_features, cat_names, device, files)


def action_find_missing():
    input_dir  = Path(input("Input folder (original photos): ").strip())
    output_dir = Path(input("Output folder (sorted): ").strip())

    all_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    existing = set()
    for f in output_dir.rglob("*"):
        if f.is_file():
            stem = f.stem
            if "_" in stem:
                parts = stem.rsplit("_", 1)
                if parts[1].isdigit():
                    stem = parts[0]
            existing.add(stem)

    missing = [f for f in all_files if f.stem not in existing]
    print(f"[*] Total: {len(all_files)} | Copied: {len(all_files)-len(missing)} | Missing: {len(missing)}")

    if not missing:
        print("[*] Nothing to do!")
        return

    model, preprocess, tokenizer, device = load_model()
    cat_features, cat_names = encode_prompts(model, tokenizer, device)
    run_classification(input_dir, output_dir, model, preprocess, cat_features, cat_names, device, missing)


def action_remove_dupes():
    folder = Path(input("Folder to deduplicate: ").strip())

    files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    print(f"[*] Scanning {len(files)} files...")

    hashes  = defaultdict(list)
    errors  = 0

    for f in tqdm(files, unit="file"):
        try:
            h = hashlib.md5(f.read_bytes()).hexdigest()
            hashes[h].append(f)
        except Exception:
            errors += 1

    dupes = {h: paths for h, paths in hashes.items() if len(paths) > 1}

    if not dupes:
        print("[*] No duplicates found!")
        return

    total_dupes = sum(len(v) - 1 for v in dupes.values())
    total_size  = sum(p.stat().st_size for paths in dupes.values() for p in paths[1:])

    print(f"\n[*] Duplicate groups: {len(dupes)}")
    print(f"[*] Files to delete:  {total_dupes}")
    print(f"[*] Space to free:    {total_size / 1024 / 1024:.1f} MB")
    print("\nExamples:")
    for h, paths in list(dupes.items())[:3]:
        print(f"  original: {paths[0].name}")
        for p in paths[1:]:
            print(f"  dupe:     {p.name}")
        print()

    confirm = input("Delete all duplicates? (yes/no): ").strip().lower()
    if confirm not in ("yes", "y"):
        print("[*] Cancelled.")
        return

    deleted = 0
    for paths in dupes.values():
        for p in paths[1:]:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                print(f"[!] Could not delete {p.name}: {e}")

    print(f"[*] Deleted: {deleted} files, freed {total_size / 1024 / 1024:.1f} MB")

# ─── Main menu ────────────────────────────────────────────────────────────────

MENU = {
    "1": ("Sort photos",        action_sort),
    "2": ("Find missing files", action_find_missing),
    "3": ("Remove duplicates",  action_remove_dupes),
}

def main():
    print("=" * 40)
    print("   Local AI Photo Sorter")
    print("   CLIP ViT-L/14 | fully local")
    print("=" * 40)

    while True:
        print()
        for key, (label, _) in MENU.items():
            print(f"  {key}. {label}")
        print("  0. Exit")
        print()

        choice = input("Select: ").strip()

        if choice == "0":
            print("Bye!")
            break
        elif choice in MENU:
            print()
            MENU[choice][1]()
        else:
            print("[!] Invalid option")

if __name__ == "__main__":
    main()

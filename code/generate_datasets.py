import shutil
import random
from pathlib import Path
from PIL import Image

ORIGIN_PATH = # YOUR ORIGIN PATH HERE

# Source folders
ORIGINS = {
    "lightness": ORIGIN_PATH / "Frames",
    "v1": ORIGIN_PATH / "Frames_v1",
    "v2": ORIGIN_PATH / "Frames_v2",
    "v3": ORIGIN_PATH / "Frames_v3",
}

# Destination folder
DEST_ROOT = # YOUR DESTINATION PATH HERE

# Combinations
COMBINATIONS = {
    "dataset_lightness_v1": ["lightness", "v1"],
    "dataset_lightness_v2": ["lightness", "v2"],
    "dataset_lightness_v3": ["lightness", "v3"],
    "dataset_lightness_v1_v2": ["lightness", "v1", "v2"],
    "dataset_lightness_v1_v3": ["lightness", "v1", "v3"],
    "dataset_lightness_v2_v3": ["lightness", "v2", "v3"],
    "dataset_lightness_v1_v2_v3": ["lightness", "v1", "v2", "v3"],
}

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def copy_image_convert_to_jpg(src_path: Path, dst_path: Path):
    """
    Copy an image converting it to JPG, regardless of the original extension.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try opening the image with PIL
    try:
        with Image.open(src_path) as img:
            # Convert to RGB (e.g., PNG with alpha channel)
            rgb_img = img.convert('RGB')
            # Save as JPG in the destination
            rgb_img.save(dst_path.with_suffix('.jpg'), format='JPEG', quality=95)
    except Exception as e:
        print(f"⚠️ Erro ao converter {src_path} para JPG: {e}")

def sortear_stems_coloridos(total=414, train_ratio=0.8, seed=42):
    # Generate numbers from 1 to total
    indices = list(range(1, total + 1))
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(indices)

    # Split according to the ratio
    split_index = int(total * train_ratio)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    # Convert to "frame_001" format
    train_stems = [f"frame_{i:03}" for i in train_indices]
    val_stems = [f"frame_{i:03}" for i in val_indices]

    print(f"🎯 Sorteio finalizado:")
    print(f"🟢 Treino: {len(train_stems)} imagens")
    print(f"🔵 Validação: {len(val_stems)} imagens")

    return train_stems, val_stems

def criar_dataset(nome_combo, versoes, treino_stems, val_stems):
    print(f"\n📁 Criando {nome_combo} com versões {versoes}")
    destino = DEST_ROOT / nome_combo

    # For each split and each stem, copy all versions
    for split, split_stems in [("train", treino_stems), ("val", val_stems)]:
        for stem in split_stems:
            for versao in versoes:
                pasta_origem = ORIGINS[versao]
                suffix = "" if versao == "lightness" else f"_{versao}"

                # First try jpg
                img_name_jpg = f"{stem}{suffix}.jpg"
                img_name_png = f"{stem}{suffix}.png"

                src_img_jpg = pasta_origem / img_name_jpg
                src_img_png = pasta_origem / img_name_png

                # Labels are txt only (same name)
                label_name = f"{stem}{suffix}.txt"
                src_lbl = pasta_origem / label_name

                # Destinations: always jpg for image, txt for label
                dst_img = destino / "images" / split / f"{stem}{suffix}.jpg"
                dst_lbl = destino / "labels" / split / label_name

                # Check which image exists
                if src_img_jpg.exists():
                    # Copy jpg image directly
                    copy_file(src_img_jpg, dst_img)
                elif src_img_png.exists():
                    # Convert png to jpg
                    copy_image_convert_to_jpg(src_img_png, dst_img)
                else:
                    print(f"⚠️ Imagem não encontrada: {img_name_jpg} nem {img_name_png}")

                # Copy label if it exists
                if src_lbl.exists():
                    copy_file(src_lbl, dst_lbl)
                else:
                    print(f"⚠️ Label não encontrado: {src_lbl}")


    # Create YAML file
    yaml = destino / "data.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    yaml.write_text(f"""train: {str((destino / 'images' / 'train').resolve())}
val:   {str((destino / 'images' / 'val').resolve())}

nc: 1
names: ['objeto']
""", encoding="utf-8")

    print(f"✅ {nome_combo} criado — {len(treino_stems)} treinos / {len(val_stems)} val.")

def main():
    # Draw stems once for all datasets
    treino_stems, val_stems = sortear_stems_coloridos(total=414, train_ratio=0.8, seed=42)
    
    for nome, versoes in COMBINATIONS.items():
        criar_dataset(nome, versoes, treino_stems, val_stems)

if __name__ == "__main__":
    main()

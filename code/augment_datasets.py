import os
from pathlib import Path
import cv2
import numpy as np
import shutil
import random
import textwrap

# ========== CONFIG ==========
DATASET_PATH = # YOUR DATASET PATH HERE
BACKGROUND_PATH = # YOUR BACKGROUND PATH HERE

BACKGROUND_PATHS = {
    "lightness": BACKGROUND_PATH / "Frames",
    "v1": BACKGROUND_PATH / "Frames_v1",
    "v2": BACKGROUND_PATH / "Frames_v2",
    "v3": BACKGROUND_PATH / "Frames_v3",
}

DATASETS = [
    "dataset_lightness",
    "dataset_lightness_v1"
]

# ==============================
# Augment functions (light)
# ==============================
def flip_horizontal(img, boxes):
    img_flipped = cv2.flip(img, 1)
    new_boxes = []
    for box in boxes:
        cls, x, y, bw, bh = box
        new_x = 1.0 - x
        new_boxes.append([cls, new_x, y, bw, bh])
    return img_flipped, new_boxes

def rotate(img, boxes, angle):
    h, w = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated, boxes  # approximation

def blur(img, boxes, ksize=(3,3)):
    return cv2.GaussianBlur(img, ksize, 0), boxes

def vignette(img, boxes):
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vign = img.copy().astype(np.float32)
    for c in range(3):
        vign[:,:,c] = vign[:,:,c] * mask
    vign = np.clip(vign, 0, 255).astype(np.uint8)
    return vign, boxes

# ==============================
# Label helpers
# ==============================
def read_labels(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x,y,w,h = map(float, parts[1:])
            boxes.append([cls, x, y, w, h])
    return boxes

def save_labels(path, boxes):
    with open(path, "w") as f:
        for box in boxes:
            cls, x, y, w, h = box
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# ==============================
# Ensure the image is saved as .jpg
# ==============================
def ensure_jpg(src_path, dst_path):
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"⚠️  Não abriu a imagem: {src_path}")
        return None
    dst_path = dst_path.with_suffix(".jpg")
    cv2.imwrite(str(dst_path), img)
    return dst_path

# ==============================
# Augment + save
# ==============================
def augment_and_save(img_path, label_path, out_img_dir, out_lbl_dir, prefix):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Não abriu a imagem: {img_path}")
        return
    boxes = read_labels(label_path)

    # flip
    img_f, boxes_f = flip_horizontal(img, boxes)
    cv2.imwrite(str(Path(out_img_dir) / f"{prefix}_flip.jpg"), img_f)
    save_labels(str(Path(out_lbl_dir) / f"{prefix}_flip.txt"), boxes_f)

    # mild rotation
    angle = random.choice([-15, -10, -5, 5, 10, 15])
    img_r, boxes_r = rotate(img, boxes, angle)
    cv2.imwrite(str(Path(out_img_dir) / f"{prefix}_rot.jpg"), img_r)
    save_labels(str(Path(out_lbl_dir) / f"{prefix}_rot.txt"), boxes_r)

    # blur
    img_b, boxes_b = blur(img, boxes)
    cv2.imwrite(str(Path(out_img_dir) / f"{prefix}_blur.jpg"), img_b)
    save_labels(str(Path(out_lbl_dir) / f"{prefix}_blur.txt"), boxes_b)

    # vignette
    img_v, boxes_v = vignette(img, boxes)
    cv2.imwrite(str(Path(out_img_dir) / f"{prefix}_vignette.jpg"), img_v)
    save_labels(str(Path(out_lbl_dir) / f"{prefix}_vignette.txt"), boxes_v)

# ==============================
# Create augmented dataset
# ==============================
def create_augmented_dataset(base_dir: Path, new_dir: Path, background_dirs):
    print(f"\n▶️  Processando dataset: {base_dir.name}")
    if not base_dir.exists():
        print(f"    ❌ Pasta base não encontrada: {base_dir}")
        return

    if new_dir.exists():
        shutil.rmtree(new_dir)
    (new_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (new_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (new_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (new_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 1) copy val
    for f in (base_dir / "images" / "val").glob("*"):
        if f.suffix.lower() not in [".jpg", ".png", ".jpeg"]: continue
        dest_img = ensure_jpg(f, new_dir / "images" / "val" / f.stem)
        lbl_name = f.stem + ".txt"
        src_lbl = base_dir / "labels" / "val" / lbl_name
        dest_lbl = new_dir / "labels" / "val" / lbl_name
        if src_lbl.exists():
            shutil.copy2(src_lbl, dest_lbl)
        else:
            open(dest_lbl, "w").close()

    # 2) copy train + augmentations
    for f in (base_dir / "images" / "train").glob("*"):
        if f.suffix.lower() not in [".jpg", ".png", ".jpeg"]: continue
        dest_img = ensure_jpg(f, new_dir / "images" / "train" / f.stem)
        lbl_name = f.stem + ".txt"
        src_lbl = base_dir / "labels" / "train" / lbl_name
        dest_lbl = new_dir / "labels" / "train" / lbl_name
        if src_lbl.exists():
            shutil.copy2(src_lbl, dest_lbl)
        else:
            open(dest_lbl, "w").close()

        augment_and_save(str(dest_img), str(dest_lbl),
                         str(new_dir / "images" / "train"),
                         str(new_dir / "labels" / "train"),
                         f.stem)

    # 3) backgrounds
    for bg_dir in background_dirs:
        for f in bg_dir.glob("*"):
            if f.suffix.lower() not in [".jpg", ".png", ".jpeg"]: continue
            dest_img = ensure_jpg(f, new_dir / "images" / "train" / f.stem)
            lbl_file = bg_dir / (f.stem + ".txt")
            dest_lbl = new_dir / "labels" / "train" / (f.stem + ".txt")
            if lbl_file.exists():
                shutil.copy2(lbl_file, dest_lbl)
            else:
                open(dest_lbl, "w").close()

            augment_and_save(str(dest_img), str(dest_lbl),
                             str(new_dir / "images" / "train"),
                             str(new_dir / "labels" / "train"),
                             f.stem)

    # 4) create yaml
    yaml_path = new_dir / f"{new_dir.name}.yaml"
    yaml_text = textwrap.dedent(f"""
        path: {str(new_dir)}
        train: images/train
        val: images/val

        nc: 1
        names: ['person']
    """)
    yaml_path.write_text(yaml_text)
    print(f"✅ Dataset aumentado criado em: {new_dir} (yaml: {yaml_path})")

# ==============================
# Execution
# ==============================
if __name__ == "__main__":
    random.seed(42)
    for ds in DATASETS:
        base_ds = BASE_PATH / ds
        new_ds = BASE_PATH / f"{ds}_aug"
        bg_dirs = []
        if "lightness" in ds: bg_dirs.append(BACKGROUND_PATHS["lightness"])
        if "v1" in ds: bg_dirs.append(BACKGROUND_PATHS["v1"])
        if "v2" in ds: bg_dirs.append(BACKGROUND_PATHS["v2"])
        if "v3" in ds: bg_dirs.append(BACKGROUND_PATHS["v3"])
        create_augmented_dataset(base_ds, new_ds, bg_dirs)
    print("\nFim do processamento.")

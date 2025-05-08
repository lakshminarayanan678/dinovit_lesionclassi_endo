import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# === USER CONFIGURATION ===
original_root = Path("/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/data/capsulevision")  # <-- REPLACE THIS with your actual root
output_root = Path("/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data")  # Output directory
split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
random_seed = 42

# === STEP 1: Collect all images grouped by (lesion_class, dataset) ===
all_images = {}  # {(lesion_class, dataset): [list of image paths]}

for subdir in original_root.iterdir():
    if not subdir.is_dir():
        continue
    for lesion_class in subdir.iterdir():
        if not lesion_class.is_dir():
            continue
        for dataset in lesion_class.iterdir():
            if not dataset.is_dir():
                continue
            key = (lesion_class.name, dataset.name)
            all_images.setdefault(key, [])
            for img_file in dataset.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    all_images[key].append(img_file)

# === STEP 2: Split and copy ===
for (lesion_class, dataset), images in all_images.items():
    # Shuffle and split
    train_imgs, temp_imgs = train_test_split(
        images, train_size=split_ratios["train"], random_state=random_seed
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=split_ratios["test"] / (split_ratios["val"] + split_ratios["test"]),
        random_state=random_seed,
    )

    split_map = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    # Copy files to new structure
    for split_name, img_list in split_map.items():
        out_dir = output_root / split_name / lesion_class / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in img_list:
            shutil.copy(img_path, out_dir / img_path.name)

print(f"✅ Splitting complete. Output saved in: {output_root.resolve()}")

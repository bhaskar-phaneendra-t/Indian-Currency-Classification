import re
from pathlib import Path

DATA_ROOT = Path("data/raw")

# Match UUID in filename
UUID_PATTERN = re.compile(
    r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
    re.IGNORECASE
)

def rename_images_in_class(class_dir: Path):
    for img_path in class_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        filename = img_path.name
        match = UUID_PATTERN.search(filename)

        if not match:
            print(f"Skipping (no UUID found): {filename}")
            continue

        uuid = match.group(1)
        new_name = f"img_{uuid}{img_path.suffix.lower()}"
        new_path = class_dir / new_name

        if new_path.exists():
            print(f"Already exists, skipping: {new_name}")
            continue

        img_path.rename(new_path)
        print(f"Renamed: {filename} → {new_name}")

def main():
    for class_folder in DATA_ROOT.iterdir():
        if class_folder.is_dir():
            print(f"\nProcessing folder: {class_folder.name}")
            rename_images_in_class(class_folder)

    print("\n All  images renamed successfully.")

if __name__ == "__main__":
    main()

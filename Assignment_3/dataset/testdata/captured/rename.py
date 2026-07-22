from pathlib import Path

folder = Path(".")

extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

images = []

for ext in extensions:
    images.extend(folder.glob(f"*{ext}"))
    images.extend(folder.glob(f"*{ext.upper()}"))

images = sorted(images)

print(f"Found {len(images)} image(s).")

for i, img in enumerate(images, start=1):
    new_name = folder / f"img-{i:04d}{img.suffix.lower()}"
    print(f"{img.name} -> {new_name.name}")
    img.rename(new_name)

print("Done!")
from pathlib import Path
import shutil

root = Path("data/imagenet")
train_dir = root / "train"
val_dir = root / "val"
gt_file = root / "ILSVRC2012_validation_ground_truth.txt"

classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
labels = [int(x.strip()) for x in gt_file.read_text().splitlines() if x.strip()]
images = sorted(val_dir.glob("ILSVRC2012_val_*.JPEG"))

print(f"classes={len(classes)} labels={len(labels)} images={len(images)}")
assert len(classes) == 1000, "train/ should contain 1000 class folders"
assert len(labels) == len(images), "val image count and label count must match"

for img, label in zip(images, labels):
    cls = classes[label - 1]  # labels are 1-based
    dst_dir = val_dir / cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img), str(dst_dir / img.name))

print("Done.")

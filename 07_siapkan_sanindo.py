import json
import shutil
from pathlib import Path

BASE = Path.home() / "Downloads" / "SANINDO.v1i.coco"
OUT = Path("dataset_sanindo_yolo")

SPLITS = {"train": "train", "valid": "val", "test": "test"}

sample_path = BASE / "train" / "_annotations.coco.json"
sample = json.loads(sample_path.read_text(encoding="utf-8"))

categories = [
    c for c in sample["categories"]
    if c["name"].strip().lower() == "person"
]

if len(categories) != 1:
    raise ValueError(f"Kategori person tidak ditemukan dengan tepat: {categories}")

person_id = categories[0]["id"]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "classes.txt").write_text("person\n", encoding="utf-8")

for source_split, target_split in SPLITS.items():
    source_dir = BASE / source_split
    data = json.loads(
        (source_dir / "_annotations.coco.json").read_text(encoding="utf-8")
    )

    image_dir = OUT / "images" / target_split
    label_dir = OUT / "labels" / target_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    annotations = {}
    for ann in data["annotations"]:
        if ann["category_id"] == person_id:
            annotations.setdefault(ann["image_id"], []).append(ann)

    empty_count = 0

    for image in data["images"]:
        src = source_dir / image["file_name"]
        if not src.exists():
            raise FileNotFoundError(src)

        shutil.copy2(src, image_dir / src.name)

        w, h = image["width"], image["height"]
        rows = []

        for ann in annotations.get(image["id"], []):
            x, y, bw, bh = ann["bbox"]

            if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
                raise ValueError(f"Bbox tidak valid: {src.name}")

            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            if not (
                0 <= cx <= 1 and 0 <= cy <= 1
                and 0 < nw <= 1 and 0 < nh <= 1
            ):
                raise ValueError(f"Bbox di luar batas: {src.name}")

            rows.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = label_dir / f"{src.stem}.txt"
        label_path.write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )

        if not rows:
            empty_count += 1

    print(
        f"{target_split}: {len(data['images'])} gambar, "
        f"{empty_count} label kosong"
    )

yaml = (
    f"path: {OUT.resolve().as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: person\n"
)
(OUT / "data.yaml").write_text(yaml, encoding="utf-8")

print(f"\nSelesai. Dataset tersimpan di: {OUT.resolve()}")
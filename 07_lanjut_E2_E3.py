from pathlib import Path
import time
import csv
from datetime import datetime
from ultralytics import YOLO

DATA = Path("dataset_sanindo_v1/data.yaml").resolve()
PROJECT = Path("hasil_praktik").resolve()
NIM = "4222401048"
NAMA = "STIVEN RAIN HEART"
EPOCHS = 10
IMGSZ = 416
BATCH = 4
DEVICE = 0
MODEL = "yolo26n"

if not DATA.exists():
    raise SystemExit(f"Dataset tidak ditemukan: {DATA}")

E1_BEST = PROJECT / "E1_pretrained" / "weights" / "best.pt"
if not E1_BEST.exists():
    raise SystemExit(f"Bobot E1 tidak ditemukan: {E1_BEST}")

def evaluasi(bobot, imgsz, split):
    return YOLO(str(bobot)).val(
        data=str(DATA),
        split=split,
        imgsz=imgsz,
        batch=BATCH,
        plots=False,
        verbose=False,
        workers=0,
        device=DEVICE
    )

def buat_baris(nama, perubahan, pretrained, metrik, menit, imgsz, bobot):
    return {
        "nim": NIM,
        "nama": NAMA,
        "eksperimen": nama,
        "perubahan": perubahan,
        "pretrained": pretrained,
        "mAP50": round(metrik.box.map50, 4),
        "mAP50-95": round(metrik.box.map, 4),
        "precision": round(metrik.box.mp, 4),
        "recall": round(metrik.box.mr, 4),
        "waktu_latih_menit": menit,
        "imgsz": imgsz,
        "bobot": str(bobot)
    }

def latih(nama, model_file, pretrained, perubahan, **ubah):
    cfg = dict(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=0,
        deterministic=True,
        project=str(PROJECT),
        name=nama,
        exist_ok=True,
        pretrained=pretrained,
        plots=True,
        verbose=False,
        workers=0,
        device=DEVICE
    )
    cfg.update(ubah)
    print(f"\n=== {nama} ===", flush=True)
    model = YOLO(model_file)
    mulai = time.time()
    model.train(**cfg)
    menit = round((time.time() - mulai) / 60, 2)
    bobot = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not bobot.exists():
        raise SystemExit(f"Bobot terbaik tidak ditemukan: {bobot}")
    metrik = evaluasi(bobot, cfg["imgsz"], "val")
    return buat_baris(
        nama, perubahan, pretrained, metrik, menit, cfg["imgsz"], bobot
    )

print("\n=== Mengevaluasi E1 yang sudah tersimpan; tidak melatih ulang ===",
      flush=True)
m1 = evaluasi(E1_BEST, IMGSZ, "val")
hasil = [
    buat_baris(
        "E1_pretrained", "baseline; bobot yang sudah tersimpan",
        True, m1, "tidak dilatih ulang", IMGSZ, E1_BEST
    )
]

hasil.append(latih(
    "E2_scratch", f"{MODEL}.yaml", False,
    "inisialisasi bobot acak"
))

hasil.append(latih(
    "E3_tuning", f"{MODEL}.pt", True,
    "imgsz=640", imgsz=640
))

terbaik = max(hasil, key=lambda baris: baris["mAP50-95"])

print(f"\n=== Model terbaik pada validasi: {terbaik['eksperimen']} ===",
      flush=True)
uji = evaluasi(terbaik["bobot"], terbaik["imgsz"], "test")

kolom = [
    "nim", "nama", "eksperimen", "perubahan", "pretrained",
    "mAP50", "mAP50-95", "precision", "recall",
    "waktu_latih_menit", "imgsz", "bobot"
]
csv_path = PROJECT / f"hasil_praktik_{NIM}.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    penulis = csv.DictWriter(f, fieldnames=kolom)
    penulis.writeheader()
    penulis.writerows(hasil)

txt_path = PROJECT / f"hasil_praktik_{NIM}.txt"
with txt_path.open("w", encoding="utf-8") as f:
    f.write("PRAKTIK TERPADU RET503 - PERBANDINGAN MODEL YOLO\n")
    f.write(f"NIM / Nama: {NIM} / {NAMA}\n")
    f.write(f"Waktu laporan: {datetime.now():%d-%m-%Y %H:%M:%S}\n")
    f.write(f"Dataset: {DATA}\n")
    f.write(f"Epoch E2/E3: {EPOCHS}; imgsz dasar: {IMGSZ}; batch: {BATCH}\n\n")
    for r in hasil:
        f.write(f"{r['eksperimen']}\n")
        f.write(f"  Perubahan: {r['perubahan']}\n")
        f.write(f"  Precision: {r['precision']}\n")
        f.write(f"  Recall: {r['recall']}\n")
        f.write(f"  mAP50: {r['mAP50']}\n")
        f.write(f"  mAP50-95: {r['mAP50-95']}\n")
        f.write(f"  Bobot: {r['bobot']}\n\n")
    f.write(f"Model terbaik berdasarkan validasi: {terbaik['eksperimen']}\n")
    f.write("Hasil evaluasi TEST model terbaik:\n")
    f.write(f"  Precision: {round(uji.box.mp, 4)}\n")
    f.write(f"  Recall: {round(uji.box.mr, 4)}\n")
    f.write(f"  mAP50: {round(uji.box.map50, 4)}\n")
    f.write(f"  mAP50-95: {round(uji.box.map, 4)}\n")

print("\n=== PERBANDINGAN VALIDASI ===")
for r in hasil:
    print(
        f"{r['eksperimen']}: mAP50={r['mAP50']}, "
        f"mAP50-95={r['mAP50-95']}, precision={r['precision']}, "
        f"recall={r['recall']}"
    )
print(f"\nModel terbaik: {terbaik['eksperimen']}")
print("Hasil TEST:")
print(f"Precision={round(uji.box.mp, 4)}")
print(f"Recall={round(uji.box.mr, 4)}")
print(f"mAP50={round(uji.box.map50, 4)}")
print(f"mAP50-95={round(uji.box.map, 4)}")
print(f"\nLaporan CSV: {csv_path.resolve()}")
print(f"Laporan TXT: {txt_path.resolve()}")

from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hasil_praktik"
REPORT = OUT / "hasil_praktik_4222401048.csv"
EXPERIMENTS = [
    ("E1_pretrained", "E1 — Pretrained"),
    ("E2_scratch", "E2 — Scratch"),
    ("E3_tuning", "E3 — Tuning"),
]

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [
            {str(k).strip(): (v or "").strip() for k, v in row.items()
             if k is not None}
            for row in csv.DictReader(f)
        ]

report_rows = read_csv(REPORT)
report = {r["eksperimen"]: r for r in report_rows}

for key, _ in EXPERIMENTS:
    if key not in report:
        raise SystemExit(f"Hasil {key} tidak ditemukan dalam {REPORT}")
    result_file = OUT / key / "results.csv"
    if not result_file.exists():
        raise SystemExit(f"Grafik training sumber tidak ditemukan: {result_file}")

def value(exp, metric):
    return float(report[exp][metric]) * 100

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "Deteksi Person pada Dataset SANINDO — Perbandingan YOLO",
    fontsize=16, fontweight="bold"
)

labels = [label for _, label in EXPERIMENTS]
x = list(range(len(labels)))
width = 0.36

# Panel 1: mAP50 dan mAP50-95
ax = axes[0, 0]
map50 = [value(key, "mAP50") for key, _ in EXPERIMENTS]
map5095 = [value(key, "mAP50-95") for key, _ in EXPERIMENTS]
b1 = ax.bar([i - width/2 for i in x], map50, width, label="mAP50")
b2 = ax.bar([i + width/2 for i in x], map5095, width, label="mAP50-95")
ax.set_title("Metrik mAP — Validasi")
ax.set_ylabel("Persentase (%)")
ax.set_xticks(x, labels)
ax.set_ylim(0, 105)
ax.legend()
ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)

# Panel 2: Precision dan Recall
ax = axes[0, 1]
precision = [value(key, "precision") for key, _ in EXPERIMENTS]
recall = [value(key, "recall") for key, _ in EXPERIMENTS]
b1 = ax.bar([i - width/2 for i in x], precision, width, label="Precision")
b2 = ax.bar([i + width/2 for i in x], recall, width, label="Recall")
ax.set_title("Precision dan Recall — Validasi")
ax.set_ylabel("Persentase (%)")
ax.set_xticks(x, labels)
ax.set_ylim(0, 105)
ax.legend()
ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)

# Panel 3: mAP50-95 per epoch dari results.csv
ax = axes[1, 0]
for key, label in EXPERIMENTS:
    rows = read_csv(OUT / key / "results.csv")
    if not rows:
        continue
    keys = list(rows[0].keys())
    metric_key = next(
        (k for k in keys if "map50-95" in k.lower()),
        None
    )
    epoch_key = next((k for k in keys if k.lower() == "epoch"), None)
    if not metric_key:
        ax.text(0.1, 0.5, f"Kolom mAP50-95 tidak ditemukan untuk {key}")
        continue
    epochs = [
        int(float(r[epoch_key])) + 1 if epoch_key and r.get(epoch_key) else i + 1
        for i, r in enumerate(rows)
    ]
    values = [float(r[metric_key]) * 100 for r in rows]
    ax.plot(epochs, values, marker=".", markersize=3, label=label)
ax.set_title("Perkembangan mAP50-95 per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("mAP50-95 (%)")
ax.grid(True, alpha=0.3)
ax.legend()

# Panel 4: box loss training
ax = axes[1, 1]
for key, label in EXPERIMENTS:
    rows = read_csv(OUT / key / "results.csv")
    if not rows:
        continue
    keys = list(rows[0].keys())
    loss_key = next(
        (k for k in keys if "train/box_loss" in k.lower()),
        None
    )
    epoch_key = next((k for k in keys if k.lower() == "epoch"), None)
    if not loss_key:
        continue
    epochs = [
        int(float(r[epoch_key])) + 1 if epoch_key and r.get(epoch_key) else i + 1
        for i, r in enumerate(rows)
    ]
    losses = [float(r[loss_key]) for r in rows]
    ax.plot(epochs, losses, marker=".", markersize=3, label=label)
ax.set_title("Perkembangan Box Loss Training")
ax.set_xlabel("Epoch")
ax.set_ylabel("Box loss")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout(rect=[0, 0, 1, 0.95])
target = OUT / "ringkasan_perbandingan.png"
fig.savefig(target, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Grafik ringkasan berhasil dibuat: {target}")

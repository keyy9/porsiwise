"""Generate ml/train_yolo26seg.ipynb from helper sources (run: python3 ml/build_notebook.py)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HELPERS = (HERE / "ml_helpers.py").read_text(encoding="utf-8")
KELAS = (ROOT / "data" / "kelas_v1.csv").read_text(encoding="utf-8")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Porsiwise: training YOLO26-seg makanan Indonesia (v2)

Notebook ini kamu jalankan sendiri dari atas ke bawah. Hasil akhirnya:

- model segmentasi `best.pt`
- metrik per kelas dan per sumber dataset
- `MODEL_CARD.md` (dokumentasi training)
- file export untuk app Flutter: `.tflite` (Android) dan `.mlpackage.zip` (iOS)

**Yang baru di v2** (berdasarkan hasil v1):

1. **Semua sumber wajib lengkap.** Di v1 cuma 1 dari 3 dataset yang kepakai tanpa ketahuan; sekarang notebook berhenti dengan pesan jelas.
2. **Salinan augmentasi Roboflow dibuang.** Jumlah data jadi jujur, epoch ±3× lebih cepat. Augmentasi tetap dilakukan Ultralytics saat training.
3. **Cek foto duplikat antar split.** Foto yang sama di train dan test bikin skor test terlalu bagus.
4. **Mask SAM dirapikan:** dipotong ke kotak (nggak bocor ke piring), serpihan kecil dibuang, mask yang gagal diganti elips. Bug urutan mask yang bisa tertukar juga sudah diperbaiki.
5. **A7b: uji kualitas mask** terhadap polygon buatan manusia.
6. **A9: profil cara pelabelan** per kelas, untuk mendeteksi label per buah vs per kelompok yang campur.
7. **A12: evaluasi per sumber**, plus banding dengan v1 di test set yang sama.

**Sebelum mulai:**

1. **GPU:** pilih `Runtime → Change runtime type → T4 GPU`.
2. **API key Roboflow:** buka ikon 🔑 (Secrets) di sidebar kiri, tambahkan `ROBOFLOW_API_KEY`, lalu nyalakan *Notebook access*.
3. **Google Drive:** semua hasil disimpan di `MyDrive/gizi-ml/`. Kalau Colab putus, kerjaan nggak hilang.
""")

md("""
## A1. Setup
""")
code("""
!nvidia-smi -L || echo "⚠️ GPU tidak terdeteksi: ganti runtime ke T4 GPU dulu"
%pip install -q "ultralytics>=8.4.142" roboflow
""")
code("""
import ultralytics, torch
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| CUDA", torch.cuda.is_available())
""")

md("""
## A2. Google Drive, folder kerja, dan setting

Dataset dikerjakan di disk lokal Colab (`/content`) karena jauh lebih cepat dari Drive. Hasil penting disalin ke Drive.

Nama dataset dan run diberi versi. Jadi hasil v1 di Drive tetap aman, dan v2 dibangun dari awal.
""")
code("""
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

DATASET_NAME = "makanan-v2"
RUN_NAME = "yolo26n-seg-v2"
PREV_RUN = "yolo26n-seg-v1"        # untuk perbandingan di A12 (boleh tidak ada)

DRIVE_ROOT = Path("/content/drive/MyDrive/gizi-ml")
DS_DRIVE = DRIVE_ROOT / "datasets" / DATASET_NAME            # laporan dataset (json/csv)
DATASET_ZIP = DRIVE_ROOT / "datasets" / f"{DATASET_NAME}-seg.zip"
RAW = Path("/content/datasets/raw")                           # hasil download Roboflow
MERGED = Path(f"/content/datasets/{DATASET_NAME}")           # dataset gabungan (lokal, cepat)
RUNS = DRIVE_ROOT / "runs"

SAM_WEIGHTS = "sam2.1_b.pt"   # model SAM untuk box -> mask (lihat A7)
BASE_MODEL = "yolo26n-seg.pt"
EPOCHS, PATIENCE, IMGSZ = 100, 20, 640

for p in (DS_DRIVE, RUNS, RAW):
    p.mkdir(parents=True, exist_ok=True)
print("Hasil akan disimpan di", DRIVE_ROOT)
""")

md("""
## A3. Helper dan daftar kelas

Sel pertama menulis file helper. Sel kedua menulis `kelas_v1.csv`. Kalau daftar kelasnya kamu ubah, ganti isi sel kedua lalu jalankan ulang dari sini.
""")
code("%%writefile /content/ml_helpers.py\n" + HELPERS)
code("%%writefile /content/kelas_v1.csv\n" + KELAS)
code("""
import sys, importlib, json
sys.path.insert(0, "/content")
import ml_helpers as H
importlib.reload(H)

CLASS_ROWS, NAMES, REMAP = H.load_classes("/content/kelas_v1.csv")
SLUG2ID = {v: k for k, v in NAMES.items()}
print(f"{len(NAMES)} kelas; prioritas A: {sum(r['prioritas'] == 'A' for r in CLASS_ROWS)}")
""")

md("""
## A4. Dataset sumber

Nilai `tag` harus sama dengan nama dataset di kolom `remap_dari` (`CalorEase`, `Combination`, `Calory`).

**CalorEase dan Calory belum punya versi di Roboflow**, jadi harus di-fork dulu (sekali saja):

1. Buka link dataset di komentar bawah → klik **Fork Dataset** → pilih workspace kamu.
2. Di project hasil fork, buat versi baru (**Generate** / *Create New Version*):
   - Preprocessing: *Auto-Orient* + *Resize → Fit within 640×640* (jangan *Stretch*, biar bentuk makanan nggak gepeng).
   - Augmentation: **tidak ada**. Augmentasi dilakukan Ultralytics saat training.
   - Kalau semua gambar masuk *Train*, pakai *Rebalance* jadi 70/20/10 dulu.
3. Lihat URL-nya: `app.roboflow.com/<workspace>/<project>/<versi>`. Isi tiga nilai itu di bawah.

Kalau ternyata dataset aslinya sudah punya versi, boleh langsung pakai workspace + versi aslinya.

**Combination tetap pakai v13**, supaya test set-nya sama dengan v1 dan hasilnya bisa dibandingkan. Salinan augmentasinya dibuang otomatis.

**Foto sendiri ikut training (mulai v3):** tambahkan project Roboflow-mu sebagai sumber dengan `tag="FotoSendiri"`. Nama kelas yang sama dengan `kelas_v1` (mis. `nasi_putih`) dikenali otomatis. Foto di split **Test** tetap tidak ikut training, dan bisa dinilai terpisah di Bagian C.
""")
code("""
SOURCES = [
    # Asli: https://universe.roboflow.com/gulakupedia/calorease-lps84  (CC0, ±4.2k gambar) -> fork dulu
    dict(tag="CalorEase", workspace="ISI_WORKSPACE_KAMU", project="ISI_PROJECT_HASIL_FORK", version=None,
         license="CC0 1.0"),
    # Asli: https://universe.roboflow.com/machinelearning-4tyun/indonesian-combination-food  (CC BY 4.0)
    dict(tag="Combination", workspace="machinelearning-4tyun", project="indonesian-combination-food", version=13,
         license="CC BY 4.0"),
    # Asli: https://universe.roboflow.com/ayu-asipq/calory  (CC BY 4.0) -> fork dulu kalau belum ada versi
    dict(tag="Calory", workspace="ISI_WORKSPACE_KAMU", project="ISI_PROJECT_HASIL_FORK", version=None,
         license="CC BY 4.0"),
]

DEDUPE_RF_AUG = True           # buang salinan augmentasi Roboflow (..rf.<hash>)
ALLOW_MISSING_SOURCES = False  # True = boleh lanjut walau ada sumber yang belum siap (tidak disarankan)
REBUILD_DATASET = False        # True = hapus dataset gabungan + zip di Drive lalu bangun ulang dari nol

import shutil
if REBUILD_DATASET:
    shutil.rmtree(MERGED, ignore_errors=True)
    shutil.rmtree(RAW, ignore_errors=True)
    DATASET_ZIP.unlink(missing_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    print("Dataset lama dihapus, akan dibangun ulang.")
""")

md("""
## A5. Download dari Roboflow
""")
code("""
missing = [s["tag"] for s in SOURCES if not s.get("version") or str(s.get("workspace", "")).startswith("ISI_")]
if DATASET_ZIP.exists() or MERGED.exists():
    ready = [s["tag"] for s in SOURCES if s["tag"] not in missing]
    print("Dataset gabungan sudah ada (lokal atau zip di Drive): download dilewati.")
    print("Kalau mau bangun ulang, set REBUILD_DATASET = True di A4.")
else:
    if missing and not ALLOW_MISSING_SOURCES:
        raise RuntimeError(f"⛔ Sumber belum siap: {missing}. Fork + generate versi dulu (lihat A4), "
                           "lalu isi workspace/project/version di SOURCES.")
    from roboflow import Roboflow
    rf = Roboflow(api_key=H.roboflow_api_key())
    ready = []
    for s in SOURCES:
        if s["tag"] in missing:
            print(f"⏭ {s['tag']}: dilewati (belum siap)")
            continue
        dest = RAW / s["tag"]
        if not (dest / "data.yaml").exists():
            rf.workspace(s["workspace"]).project(s["project"]).version(s["version"]).download(
                "yolov8", location=str(dest), overwrite=False)
        ready.append(s["tag"])
print("Siap digabung:", ready)
""")

md("""
## A6. Gabung, remap kelas, dan cek duplikat

- Split asli tiap dataset dipertahankan (train → train, valid → val, test → test).
- Salinan augmentasi Roboflow dibuang (`DEDUPE_RF_AUG`). Polygon 4 titik yang sebenarnya kotak diubah jadi box supaya ikut diproses SAM.
- Kelas yang tidak ada di `kelas_v1` dibuang. Gambar yang jadi kosong ikut dibuang.
- **Duplikat lintas split:** foto val/test yang (hampir) identik dengan foto di split lain dihapus dari val/test. Ini penting karena dataset Roboflow Universe sering saling menyalin.

Kalau zip dataset sudah ada di Drive dari sesi sebelumnya, sel ini cukup mengekstraknya.
""")
code("""
import zipfile

if DATASET_ZIP.exists() and not MERGED.exists():
    print("Memulihkan dataset dari Drive:", DATASET_ZIP)
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(MERGED)
elif not MERGED.exists():
    MERGE_REPORT = H.merge_sources(ready, RAW, MERGED, REMAP, dedupe_rf_aug=DEDUPE_RF_AUG, rows=CLASS_ROWS)
    for tag, rep in MERGE_REPORT.items():
        print(f"\\n== {tag}")
        print("  gambar:", rep["gambar_masuk"], "→", rep["gambar_keluar"],
              "| salinan augmentasi dibuang:", rep["salinan_augmentasi_dibuang"],
              "| dibuang kosong:", rep["gambar_dibuang_kosong"])
        print("  instance:", rep["instance_masuk"], "→", rep["instance_keluar"],
              "| polygon asli:", rep["instance_polygon_asli"], "| polygon kotak → box:", rep.get("polygon_kotak_jadi_box", 0))
        print("  per split:", rep["per_split"])
        if rep["kelas_tidak_dipetakan"]:
            print("  kelas dibuang:", rep["kelas_tidak_dipetakan"])
    (DS_DRIVE / "merge_report.json").write_text(json.dumps(MERGE_REPORT, indent=2, ensure_ascii=False))
else:
    print("Dataset gabungan sudah ada di", MERGED)
""")
code("""
# Cek duplikat lintas split (dHash). Jalankan sebelum SAM supaya SAM tidak memproses foto yang akan dihapus.
import matplotlib.pyplot as plt
from PIL import Image

HAPUS_DUPLIKAT = True
dup_file = DS_DRIVE / "duplicates.json"
if dup_file.exists():
    DUP_INFO = json.loads(dup_file.read_text())
    print("Duplikat sudah dicek sebelumnya:", DUP_INFO["dihapus"])
else:
    dups = H.find_cross_split_duplicates(MERGED, max_dist=4)
    print(f"{len(dups)} pasangan foto mirip lintas split")
    for d in dups[:8]:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(Image.open(MERGED / "images" / d["split"] / d["file"])); ax[0].set_title(f"{d['split']}", fontsize=8)
        ax[1].imshow(Image.open(MERGED / "images" / d["mirip_dengan_split"] / d["mirip_dengan"]))
        ax[1].set_title(f"{d['mirip_dengan_split']} (jarak {d['jarak']})", fontsize=8)
        for a in ax: a.axis("off")
        plt.show()
    removed = H.remove_duplicates(MERGED, dups) if HAPUS_DUPLIKAT else {}
    DUP_INFO = {"pasangan": len(dups), "dihapus": removed, "contoh": dups[:50]}
    dup_file.write_text(json.dumps(DUP_INFO, indent=2, ensure_ascii=False))
    print("Dihapus dari val/test:", removed)

DATA_YAML = H.write_data_yaml(MERGED, NAMES)
print(open(DATA_YAML).read()[:300])
""")

md("""
## A7. Ubah box jadi mask pakai SAM 2.1

Box dipakai sebagai *prompt* ke SAM, lalu SAM yang menggambar mask-nya. Di v2 mask-nya dirapikan:

- **Dipotong ke kotak.** Mask yang bocor keluar kotak (ikut piring atau daun pisang) pasti salah.
- **Serpihan kecil dibuang** (< 10% dari bagian terbesar).
- **Mask yang gagal** (kosong atau < 8% luas kotak) diganti **elips di dalam kotak**. Elips lebih mirip bentuk makanan daripada persegi.

Sel ini bisa dilanjutkan: kalau Colab putus, jalankan ulang, dan file yang sudah jadi polygon dilewati.
""")
code("""
import pandas as pd

predict = H.make_sam_predict_fn(SAM_WEIGHTS, device=0)
SAM_STATS, SUSPICIOUS = H.convert_boxes_to_masks(MERGED, predict)
print(SAM_STATS)
print("Instance mencurigakan:", len(SUSPICIOUS))

if SAM_STATS.get("gambar_dikonversi", 0) or not DATASET_ZIP.exists():
    (DS_DRIVE / "sam_stats.json").write_text(json.dumps({**SAM_STATS, **predict.config}, indent=2))
    pd.DataFrame(SUSPICIOUS).to_csv(DS_DRIVE / "sam_suspicious.csv", index=False)
    # simpan dataset hasil SAM ke Drive supaya sesi berikutnya tidak perlu SAM ulang
    shutil.make_archive(str(DATASET_ZIP.with_suffix("")), "zip", root_dir=MERGED)
    print("Tersimpan:", DATASET_ZIP)
else:
    print("Semua label sudah polygon; zip di Drive sudah terbaru.")
""")

md("""
## A7b. Uji kualitas mask (opsional, ±5 menit)

Sebagian label dari sumber sudah berupa **polygon buatan manusia** (di v1 ada 143 gambar). Itu bisa dipakai sebagai kunci jawaban. Kotaknya dijadikan prompt, lalu hasilnya dibandingkan pakai IoU (0 = meleset total, 1 = persis sama) untuk 4 cara:

| cara | arti |
|---|---|
| `kotak` | mask = persegi kotaknya (batas bawah) |
| `elips` | mask = elips di dalam kotak |
| `sam_mentah` | SAM apa adanya (cara v1) |
| `sam_rapi` | SAM yang dirapikan (cara v2, yang dipakai training) |

Kalau `sam_rapi` jelas lebih tinggi dari `elips`, SAM memang berguna. Kalau ada kelas dengan IoU rendah, mask kelas itu perlu dicek manual.
""")
code("""
import pandas as pd

if "predict" not in globals():
    predict = H.make_sam_predict_fn(SAM_WEIGHTS, device=0)
Q_ROWS = H.sam_quality_check(MERGED, predict, max_images=200)
Q_OVERALL, Q_PER_CLASS = H.summarize_quality(Q_ROWS, NAMES)
print(f"{len(Q_ROWS)} polygon manusia diuji")
print("IoU rata-rata:", Q_OVERALL)
(DS_DRIVE / "sam_quality.json").write_text(json.dumps({"keseluruhan": Q_OVERALL, "n_instance": len(Q_ROWS),
                                                         "per_kelas": Q_PER_CLASS}, indent=2))
pd.DataFrame(Q_PER_CLASS).sort_values("sam_rapi") if Q_PER_CLASS else "Tidak ada polygon manusia di dataset ini."
""")

md("""
## A8. Review visual

Langkah ini paling sering dilewati, padahal paling menentukan kualitas model. Cek apakah mask-nya pas di lauk, tidak ikut piring, dan labelnya benar.

Di v2 cek terutama **lauk satuan**, yang di v1 paling lemah: `sate`, `telur_rebus`, `bakso`, `tahu_goreng`, `tempe_goreng`, `kerupuk`, `nugget`. Aturannya **satu mask per buah**. Kalau kamu lihat satu kotak besar menutupi sekelompok sate atau telur, catat sumbernya. Nama file diawali nama sumber.
""")
code("""
import matplotlib.pyplot as plt
import pandas as pd

CEK_KELAS = "telur_rebus"   # None = acak semua kelas; atau nama kelas, mis. "sate"
cid = SLUG2ID.get(CEK_KELAS) if CEK_KELAS else None
files = H.sample_label_files(MERGED, "train", k=12, class_id=cid)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for ax, lbl in zip(axes.ravel(), files):
    img = H.find_image(MERGED / "images" / "train", lbl.stem)
    ax.imshow(H.draw_labels(img, lbl, NAMES))
    ax.set_title(lbl.stem[:30], fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout(); plt.show()
""")
code("""
# Lihat instance yang rasio luas mask/kotaknya ekstrem
import pandas as pd
import matplotlib.pyplot as plt

sus_csv = DS_DRIVE / "sam_suspicious.csv"
if "SUSPICIOUS" not in globals():  # sesi baru: ambil dari Drive
    SUSPICIOUS = pd.read_csv(sus_csv).to_dict("records") if sus_csv.exists() and sus_csv.stat().st_size > 1 else []
if SUSPICIOUS:
    sus = pd.DataFrame(SUSPICIOUS).sample(min(8, len(SUSPICIOUS)), random_state=0)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (_, row) in zip(axes.ravel(), sus.iterrows()):
        lbl = MERGED / "labels" / row["split"] / row["label"]
        img = H.find_image(MERGED / "images" / row["split"], lbl.stem)
        if img is not None and lbl.exists():
            ax.imshow(H.draw_labels(img, lbl, NAMES))
        ax.set_title(f"{NAMES[row['class_id']]} rasio={row['rasio_luas']}", fontsize=8)
        ax.axis("off")
    plt.tight_layout(); plt.show()
""")

md("""
## A9. Statistik per kelas dan profil cara pelabelan

**Jumlah instance** di tabel pertama sudah dihitung dari foto unik, jadi tidak ada lagi salinan augmentasi yang ikut terhitung. Target minimal 150 instance train per kelas.

**Tabel profil** membandingkan cara pelabelan lauk satuan di tiap sumber:

- *instance_per_gambar* tinggi dan kotaknya kecil berarti dilabel per buah ✅.
- Kalau sumber lain untuk kelas yang sama ±1 instance per gambar dengan kotak besar, kemungkinan dilabel per kelompok. Labelnya tidak konsisten, dan kelas itu kandidat untuk dirapikan.
""")
code("""
import pandas as pd

COUNTS, N_IMAGES = H.count_instances(MERGED, NAMES)
print("Foto unik per split:", dict(N_IMAGES))
table = pd.DataFrame(H.class_table(CLASS_ROWS, COUNTS))
table.to_csv(DS_DRIVE / "class_counts.csv", index=False)
table.sort_values("train")
""")
code("""
import pandas as pd

COUNTABLE = ["ayam_goreng", "ikan_goreng", "sate", "telur_ceplok", "telur_rebus", "bakso", "tempe_goreng",
             "tahu_goreng", "perkedel", "kerupuk", "nugget", "donat"]
profile = pd.DataFrame(H.instance_profile(MERGED, NAMES, class_ids={SLUG2ID[c] for c in COUNTABLE}))
profile.to_csv(DS_DRIVE / "instance_profile.csv", index=False)
profile
""")

md("""
## A10. (Opsional) Logging ke Weights & Biases
""")
code("""
USE_WANDB = False
if USE_WANDB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)
    import wandb
    wandb.login()
    from ultralytics import settings
    settings.update({"wandb": True})
""")

md("""
## A11. Training YOLO26n-seg

- **`patience=20`:** training berhenti otomatis kalau validasi tidak membaik 20 epoch. Di v1 kurvanya sudah datar sejak epoch ±70.
- **`batch=-1`:** ukuran batch dipilih otomatis sesuai memori GPU.
- **Hasil per epoch** tersimpan di Drive, termasuk `last.pt` untuk melanjutkan.

Dengan 3 sumber, training bisa makan 2–4 jam di T4. Kalau kuota GPU habis di tengah jalan, lanjutkan besok pakai sel resume di bawah.
""")
code("""
from ultralytics import YOLO

(DRIVE_ROOT / "LAST_RUN.txt").write_text(RUN_NAME)   # dibaca bagian export (B)
model = YOLO(BASE_MODEL)   # BASE_MODEL, EPOCHS, PATIENCE, IMGSZ diatur di A2
model.train(
    data=str(DATA_YAML), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ, batch=-1,
    workers=2, seed=0, plots=True, project=str(RUNS), name=RUN_NAME, exist_ok=True,
)
""")
md("""
**Kalau Colab putus di tengah training**, jalankan lagi A1–A6. A5 otomatis melewati download, dan A6 memulihkan dataset dari zip Drive. Setelah itu jalankan sel ini (hapus tanda `#`), bukan sel training di atas:
""")
code("""
# from ultralytics import YOLO
# YOLO(str(RUNS / RUN_NAME / "weights" / "last.pt")).train(resume=True)
""")

md("""
## A12. Evaluasi akhir

Evaluasi dilakukan di split **test**, yaitu gambar yang tidak pernah dipakai buat memilih model. Kalau test belum ada, dipakai val.

Cara membaca hasilnya:

- **(B)** = metrik bounding box. GT-nya kotak buatan manusia, jadi ini angka paling bisa dipercaya untuk *akurasi deteksi*.
- **(M)** = metrik mask. GT mask-nya kebanyakan buatan SAM, jadi angka ini mengukur seberapa mirip dengan mask SAM. Kualitas mask yang sebenarnya dilihat di A7b.
- **Per kelas:** kelas dengan < 10 instance di test angkanya masih untung-untungan.
""")
code("""
import pandas as pd
from ultralytics import YOLO

BEST = RUNS / RUN_NAME / "weights" / "best.pt"
EVAL_SPLIT = "test" if (MERGED / "images" / "test").exists() and any((MERGED / "images" / "test").iterdir()) else "val"

def metrics_row(m):
    return {"box_mAP50": round(float(m.box.map50), 4), "box_mAP50-95": round(float(m.box.map), 4),
            "mask_mAP50": round(float(m.seg.map50), 4), "mask_mAP50-95": round(float(m.seg.map), 4)}

best_model = YOLO(str(BEST))
metrics = best_model.val(data=str(DATA_YAML), split=EVAL_SPLIT, imgsz=IMGSZ, plots=True,
                         project=str(RUNS), name=f"{RUN_NAME}-eval-{EVAL_SPLIT}", exist_ok=True)
FINAL = metrics_row(metrics)
print(EVAL_SPLIT, FINAL)

per_class = pd.DataFrame(metrics.summary())
per_class.to_csv(RUNS / RUN_NAME / f"per_class_{EVAL_SPLIT}.csv", index=False)
per_class
""")
code("""
from IPython.display import Image as IPImage, display
cm = Path(metrics.save_dir) / "confusion_matrix_normalized.png"
display(IPImage(filename=str(cm))) if cm.exists() else print("confusion matrix tidak ditemukan di", metrics.save_dir)
""")
code("""
# Evaluasi per sumber dataset (+ banding dengan v1 kalau best.pt v1 ada di Drive)
PREV_BEST = RUNS / PREV_RUN / "weights" / "best.pt"
prev_model = YOLO(str(PREV_BEST)) if PREV_BEST.exists() else None
PER_SOURCE = []
for tag in ready:
    yp, n = H.write_subset_yaml(MERGED, NAMES, EVAL_SPLIT, tag)
    if not yp:
        continue
    for label, mdl in (("v2", best_model), ("v1", prev_model)):
        if mdl is None:
            continue
        m = mdl.val(data=str(yp), split=EVAL_SPLIT, imgsz=IMGSZ, plots=False, verbose=False,
                    project=str(RUNS), name=f"{RUN_NAME}-eval-{tag}-{label}", exist_ok=True)
        PER_SOURCE.append({"model": label, "sumber": tag, "gambar": n, **metrics_row(m)})
pd.DataFrame(PER_SOURCE)
""")
md("""
**Membaca perbandingan v1 vs v2:** yang adil hanya baris sumber **Combination**, karena v1 dulu dilatih dari Combination saja dan test set-nya sama. Bandingkan kolom **box**. Kolom mask ikut berubah karena GT mask-nya sekarang dari SAM yang dirapikan, tapi tetap adil antar model karena keduanya dinilai dengan GT yang sama.
""")

md("""
## A13. Model card
""")
code("""
import pandas as pd

def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None

if "N_IMAGES" not in globals():
    _, N_IMAGES = H.count_instances(MERGED, NAMES)
merge_rep = load_json(DS_DRIVE / "merge_report.json") or {}
cleaning = {
    "salinan_augmentasi_dibuang": {t: r.get("salinan_augmentasi_dibuang", 0) for t, r in merge_rep.items()},
    "polygon_kotak_jadi_box": {t: r.get("polygon_kotak_jadi_box", 0) for t, r in merge_rep.items()},
    "duplikat_lintas_split_dihapus": (load_json(DS_DRIVE / "duplicates.json") or {}).get("dihapus"),
}
sam_q = load_json(DS_DRIVE / "sam_quality.json")
card = H.write_model_card(RUNS / RUN_NAME / "MODEL_CARD.md", {
    "run_name": RUN_NAME, "ultralytics_version": ultralytics.__version__, "base_model": BASE_MODEL,
    "imgsz": IMGSZ, "class_list": "kelas_v1.csv", "n_classes": len(NAMES),
    "mask_method": f"SAM ({SAM_WEIGHTS}) dari bounding box, dirapikan (potong ke kotak, buang serpihan, fallback elips)",
    "sources": [s for s in SOURCES if s["tag"] in ready],
    "n_images": dict(N_IMAGES), "cleaning": cleaning,
    "sam_stats": load_json(DS_DRIVE / "sam_stats.json"),
    "sam_quality": {"keseluruhan": sam_q["keseluruhan"], "n_instance": sam_q["n_instance"]} if sam_q else None,
    "best_epoch": H.best_epoch_info(RUNS / RUN_NAME),
    "final_split": EVAL_SPLIT, "final_metrics": FINAL,
    "per_source": [r for r in PER_SOURCE if r["model"] == "v2"],
})
shutil.copy("/content/kelas_v1.csv", RUNS / RUN_NAME / "kelas_v1.csv")
pd.DataFrame(PER_SOURCE).to_csv(RUNS / RUN_NAME / "per_source.csv", index=False)
print(card.read_text())
""")

md("""
## A14. Coba prediksi

Coba pakai beberapa gambar test, lalu foto makanan kamu sendiri. Foto asli seperti nampan makan siang itu ujian yang sebenarnya.
""")
code("""
import random
import matplotlib.pyplot as plt

test_imgs = sorted((MERGED / "images" / EVAL_SPLIT).iterdir())
for r in best_model.predict([str(p) for p in random.sample(test_imgs, min(4, len(test_imgs)))], imgsz=IMGSZ, conf=0.25):
    plt.figure(figsize=(7, 7)); plt.imshow(r.plot()[..., ::-1]); plt.axis("off"); plt.show()
""")
code("""
from google.colab import files
uploaded = files.upload()   # pilih foto makanan dari laptop/HP
for name in uploaded:
    r = best_model.predict(name, imgsz=IMGSZ, conf=0.25)[0]
    plt.figure(figsize=(7, 7)); plt.imshow(r.plot()[..., ::-1]); plt.axis("off"); plt.show()
""")

md("""
---
# Bagian B: export untuk app Flutter

Paket export bisa mengubah versi library yang dipakai training, jadi kerjakan di sesi terpisah:

1. Jalankan sel instalasi di bawah.
2. **Runtime → Restart session.**
3. Lanjut dari B2. Bagian ini nggak butuh GPU.

Setting export-nya mengikuti aset resmi plugin `ultralytics_yolo`:

- **Android:** LiteRT `w8a32` (bobot int8, aktivasi FP32, tanpa kalibrasi), `nms=None`
- **iOS:** Core ML int8, `nms=False`
""")
code("""
# numpy<=2.3.5 diminta exporter Core ML Ultralytics; dipasang sekarang supaya tidak di-update otomatis saat export
%pip install -q "ultralytics[export-litert,export-coreml]>=8.4.142" "numpy<=2.3.5"
""")
code("""
# B2. setelah restart session
from pathlib import Path
import shutil
from google.colab import drive
from ultralytics import YOLO

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/gizi-ml")
RUN_NAME = (DRIVE_ROOT / "LAST_RUN.txt").read_text().strip()   # ditulis A11; ganti manual kalau perlu
BEST = DRIVE_ROOT / "runs" / RUN_NAME / "weights" / "best.pt"
EXPORTS = DRIVE_ROOT / "exports" / RUN_NAME
EXPORTS.mkdir(parents=True, exist_ok=True)
assert BEST.exists(), f"best.pt tidak ditemukan di {BEST}"
print("Export run:", RUN_NAME)
""")
code("""
# B3. Android (LiteRT / TFLite)
tflite_path = Path(YOLO(str(BEST)).export(format="litert", quantize="w8a32", nms=None, imgsz=640))
android_name = EXPORTS / f"makanan-{RUN_NAME}_w8a32.tflite"
shutil.copy(tflite_path, android_name)
print("Android:", android_name)
""")
code("""
# B4. iOS (Core ML int8), di-zip jadi .mlpackage.zip untuk Flutter assets
mlpackage = Path(YOLO(str(BEST)).export(format="coreml", quantize=8, nms=False, imgsz=640))
ios_zip = shutil.make_archive(str(EXPORTS / f"makanan-{RUN_NAME}.mlpackage"), "zip",
                              root_dir=mlpackage.parent, base_dir=mlpackage.name)
print("iOS:", ios_zip)
""")

md("""
## Setelah selesai

Kirim ke chat file-file ini dari `MyDrive/gizi-ml/`:

- `runs/<run>/MODEL_CARD.md`
- `runs/<run>/per_class_test.csv`
- `runs/<run>/per_source.csv`
- `runs/<run>/results.csv`
- `datasets/makanan-v2/instance_profile.csv`

Nanti kita bedah bareng.

### Ide eksperimen berikutnya

| Eksperimen | Yang diuji |
|---|---|
| `yolo26s-seg.pt` vs `yolo26n-seg.pt` | Akurasi vs kecepatan di HP |
| `imgsz=800` | Lauk kecil (sambal, kerupuk) lebih terdeteksi? |
| Rapikan label lauk satuan di sumber yang tidak konsisten (A9) | Efek konsistensi label vs jumlah data |
| Test set dari foto sendiri (nampan, warung, rumah) | Performa di dunia nyata |

Ubah satu hal saja per run, dan beri `RUN_NAME` berbeda, supaya hasilnya bisa dibandingkan.
""")

md("""
---
# Bagian C: uji dengan foto sendiri

Bagian ini bisa dijalankan kapan saja, terpisah dari training. **Di sesi baru, jalankan A1, A2, A3 dulu, lalu langsung lompat ke C1.**

Hasilnya:

- skor tiap model di Drive pada foto asli kamu
- rincian per makanan: benar / salah kelas / kelewat
- kesalahan yang paling sering
- makanan yang belum punya kelas (kandidat kelas_v2)

**Cara menyiapkan foto (sekali di awal, lalu tinggal tambah):**

1. Buat project baru di Roboflow, tipe **Instance Segmentation**, misalnya `foto-makanan-sendiri`.
2. **Nama kelas** pakai nama di `kelas_v1` (daftarnya dicetak di C1), mis. `nasi_putih`, `ikan_goreng`, `sambal`. Makanan yang belum ada di daftar boleh pakai nama baru (mis. `tumis_terong`); notebook mencatatnya sebagai kandidat kelas baru.
3. **Label pakai polygon.** Smart Polygon di Roboflow bikin ini cepat. Polygon buatanmu sekalian jadi kunci jawaban asli untuk kualitas mask.
4. **Aturannya sama dengan dataset:** lauk satuan per buah, makanan curah per gundukan. Lauk bersambal (mis. lele sambel) dilabel sebagai lauknya.
5. **Masukkan foto uji ke split Test.** Bagian C hanya menilai split Test, dan foto di split Test tidak pernah dipakai training.
6. **Generate versi** tanpa augmentasi, lalu isi workspace/project/version di C1. Tiap kali foto bertambah, generate versi baru dan naikkan nomor versinya.

Kalau labelnya pakai alat lain, export format YOLO, upload foldernya ke `MyDrive/gizi-ml/foto-sendiri/`, lalu pakai mode folder di C1. Folder datar (`images/` + `labels/` + `data.yaml`) dinilai semua.
""")
code("""
from pathlib import Path

REAL_SOURCE = dict(type="roboflow", workspace="ISI_WORKSPACE_KAMU", project="foto-makanan-sendiri", version=None)
# REAL_SOURCE = dict(type="folder", path=str(DRIVE_ROOT / "foto-sendiri"))

REAL_MODELS = None   # None = semua run di Drive yang punya best.pt; atau daftar nama run, mis. ["yolo26n-seg-v2"]
REAL_CONF = 0.25     # ambang keyakinan, samakan dengan yang dipakai app
REAL_IOU = 0.5       # makanan dianggap ketemu kalau kotak tebakan overlap >= 50% dengan label
REAL_DIR = Path("/content/datasets/foto-sendiri")
REAL_RAW = Path("/content/datasets/raw-foto-sendiri")

print("Nama kelas untuk dipakai di Roboflow:")
print(", ".join(NAMES[i] for i in sorted(NAMES)))
""")
code("""
# C2. Ambil foto + label, siapkan set uji
import shutil

shutil.rmtree(REAL_DIR, ignore_errors=True)
if REAL_SOURCE["type"] == "roboflow":
    if not REAL_SOURCE.get("version") or str(REAL_SOURCE.get("workspace", "")).startswith("ISI_"):
        raise RuntimeError("⛔ Isi workspace/project/version project foto sendiri di C1 dulu.")
    from roboflow import Roboflow
    shutil.rmtree(REAL_RAW, ignore_errors=True)
    Roboflow(api_key=H.roboflow_api_key()).workspace(REAL_SOURCE["workspace"]).project(REAL_SOURCE["project"]) \\
        .version(REAL_SOURCE["version"]).download("yolov8", location=str(REAL_RAW), overwrite=True)
    real_src = REAL_RAW
else:
    real_src = Path(REAL_SOURCE["path"])

REAL_REP = H.build_real_testset(real_src, REAL_DIR, CLASS_ROWS, splits=("test",))
print("Foto per split di sumber:", REAL_REP["split_tersedia"], "| dinilai:", REAL_REP["split_dipakai"])
if REAL_REP["gambar"] == 0:
    raise RuntimeError("⛔ Belum ada foto di split Test. Di Roboflow, pindahkan foto uji ke split Test lalu generate versi baru.")
print(f"{REAL_REP['gambar']} foto | {REAL_REP['instance']} makanan dikenal "
      f"(polygon {REAL_REP['polygon']}, kotak {REAL_REP['box']}) | {REAL_REP['objek_kelas_belum_ada']} makanan kelas belum ada")
if REAL_REP["kelas_belum_ada"]:
    print("Kandidat kelas baru:", REAL_REP["kelas_belum_ada"])
REAL_YAML = H.write_real_yaml(REAL_DIR, NAMES)
""")
code("""
# C3. Label kotak -> mask pakai SAM (dilewati kalau semua label sudah polygon)
import torch

DEV = 0 if torch.cuda.is_available() else "cpu"
if REAL_REP["box"]:
    print(f"{REAL_REP['box']} makanan masih berupa kotak -> dibuat mask pakai SAM")
    real_predict = H.make_sam_predict_fn(SAM_WEIGHTS, device=DEV)
    print(H.convert_boxes_to_masks(REAL_DIR, real_predict, splits=(H.REAL_SPLIT,), log_every=0)[0])
else:
    print("Semua label polygon buatanmu: metrik mask di bawah mengukur kualitas mask yang sebenarnya.")
""")
md("""
## C4. Nilai semua model

Dua cara menilai:

- **mAP (box & mask):** sama seperti di A12, supaya bisa dibandingkan dengan angka test set dataset.
- **Per makanan:** cara yang paling mirip pengalaman di app. Tiap makanan berlabel dicek: ditebak **benar**, **salah kelas** (ketemu tapi namanya salah), atau **kelewat**. Tebakan yang tidak cocok dengan makanan mana pun dihitung **deteksi palsu**, kecuali kalau jatuh di makanan yang kelasnya belum ada.
""")
code("""
import json
from datetime import datetime
import pandas as pd
from ultralytics import YOLO

run_dirs = [RUNS / r for r in REAL_MODELS] if REAL_MODELS else sorted(p.parent.parent for p in RUNS.glob("*/weights/best.pt"))
REAL_OUT = DRIVE_ROOT / "real_eval" / datetime.now().strftime("%Y%m%d-%H%M%S")
REAL_OUT.mkdir(parents=True, exist_ok=True)
REAL_SUMMARY, ITEMS, EXTRA, REAL_MODELS_LOADED = [], {}, {}, {}
for run in run_dirs:
    mdl = YOLO(str(run / "weights" / "best.pt"))
    if len(mdl.names) != len(NAMES):
        print(f"⏭ {run.name}: dilatih dengan {len(mdl.names)} kelas, daftar kelas sekarang {len(NAMES)}. Dilewati.")
        continue
    m = mdl.val(data=str(REAL_YAML), split="test", imgsz=IMGSZ, plots=True, verbose=False, device=DEV,
                project=str(REAL_OUT), name=run.name, exist_ok=True)
    rows, extra = H.item_level_report(mdl, REAL_DIR, NAMES, conf=REAL_CONF, iou_thr=REAL_IOU, imgsz=IMGSZ, device=DEV)
    ITEMS[run.name], EXTRA[run.name], REAL_MODELS_LOADED[run.name] = rows, extra, mdl
    REAL_SUMMARY.append({"model": run.name, "box_mAP50": round(float(m.box.map50), 4),
                         "mask_mAP50": round(float(m.seg.map50), 4), "mask_mAP50-95": round(float(m.seg.map), 4),
                         **H.summarize_items(rows, extra)})
    pd.DataFrame(rows).to_csv(REAL_OUT / f"per_makanan_{run.name}.csv", index=False)
    pd.DataFrame(extra).to_csv(REAL_OUT / f"tebakan_ekstra_{run.name}.csv", index=False)
    pd.DataFrame(m.summary()).to_csv(REAL_OUT / f"per_class_{run.name}.csv", index=False)
pd.DataFrame(REAL_SUMMARY).to_csv(REAL_OUT / "ringkasan.csv", index=False)
pd.DataFrame(REAL_SUMMARY)
""")
code("""
# C5. Detail satu model: per kelas + kesalahan yang paling sering
from IPython.display import display

DETAIL_MODEL = None   # None = run terakhir (urut nama); atau nama run, mis. "yolo26n-seg-v1"
detail = DETAIL_MODEL or REAL_SUMMARY[-1]["model"]
print("Model:", detail)
display(pd.DataFrame(H.per_class_items(ITEMS[detail])))
print("Kesalahan paling sering:")
display(pd.DataFrame(H.top_confusions(ITEMS[detail], EXTRA[detail])))
""")
code("""
# C6. Contoh: label asli vs tebakan tiap model
import matplotlib.pyplot as plt

N_FOTO = 6
imgs = sorted(p for p in (REAL_DIR / "images" / H.REAL_SPLIT).iterdir())[:N_FOTO]
cols = 1 + len(REAL_MODELS_LOADED)
fig, axes = plt.subplots(len(imgs), cols, figsize=(4 * cols, 4 * len(imgs)), squeeze=False)
for i, img in enumerate(imgs):
    lbl = REAL_DIR / "labels" / H.REAL_SPLIT / f"{img.stem}.txt"
    axes[i][0].imshow(H.draw_labels(img, lbl, NAMES, extra_boxes=H.read_unknown(REAL_DIR, img.stem)))
    axes[i][0].set_title("label asli", fontsize=9)
    for k, (run_name, mdl) in enumerate(REAL_MODELS_LOADED.items(), start=1):
        r = mdl.predict(str(img), imgsz=IMGSZ, conf=REAL_CONF, verbose=False, device=DEV)[0]
        axes[i][k].imshow(r.plot()[..., ::-1])
        axes[i][k].set_title(run_name, fontsize=9)
    for a in axes[i]:
        a.axis("off")
plt.tight_layout()
fig.savefig(REAL_OUT / "contoh_prediksi.png", dpi=80)
plt.show()
""")
code("""
# C7. Simpan ringkasan
(REAL_OUT / "ringkasan.json").write_text(json.dumps({
    "sumber": {k: v for k, v in REAL_SOURCE.items()}, "foto": REAL_REP["gambar"], "makanan_dikenal": REAL_REP["instance"],
    "label_polygon": REAL_REP["polygon"], "label_kotak": REAL_REP["box"],
    "kelas_belum_ada": REAL_REP["kelas_belum_ada"], "conf": REAL_CONF, "iou": REAL_IOU, "model": REAL_SUMMARY,
}, indent=2, ensure_ascii=False, default=str))
print("Tersimpan di", REAL_OUT)
""")
md("""
Kirim ke chat `ringkasan.csv`, `ringkasan.json`, dan `per_makanan_<model>.csv` dari folder `real_eval/<tanggal>` di Drive. Dari situ kita tentukan kelas apa yang perlu ditambah dan makanan apa yang paling sering salah.

Dengan foto sedikit (puluhan), angka per kelas masih kasar. Yang paling berguna justru daftar kesalahan tersering dan kandidat kelas baru.
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = HERE / "train_yolo26seg.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Wrote", out, "with", len(cells), "cells")

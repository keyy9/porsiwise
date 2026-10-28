# Porsiwise

App Flutter (Android & iOS) untuk menghitung gizi makanan secara real-time dari kamera, lalu mencatatnya ke target harian. Dikembangkan dari dua project lama:

- **Project grup lama:** deteksi makanan pakai YOLO.
- **App web lama:** chatbot, meal planning, dan BMR/TDEE.

## Keputusan yang sudah dikunci

- **App:** Flutter, dengan plugin resmi Ultralytics (LiteRT di Android, Core ML di iOS).
- **Model:** YOLO26 segmentasi, dilatih ulang dari dataset baru. Model lama tidak dipakai.
- **Gizi:** TKPI sebagai sumber utama, dibundel offline di app (`gizi.sqlite`).
- **Alur kamera:** deteksi + mask → estimasi gram (bisa dikoreksi user lewat slider) → lookup TKPI → catat.

## Struktur

```
data/kelas_v1.csv                  daftar kelas deteksi + mapping TKPI + porsi default
data/README.md                     kamus data, sumber dataset, aturan anotasi
tools/tkpi_ingest.py               builder database gizi dari CSV TKPI
ml/train_yolo26seg.ipynb           notebook Colab: A dataset → SAM → training → evaluasi, B export, C uji foto sendiri
ml/ml_helpers.py                   fungsi yang dipakai notebook (gabung dataset, SAM, model card)
ml/build_notebook.py               generator notebook (edit di sini, lalu jalankan ulang)
ml/EVAL_yolo26n-seg-v1.md          evaluasi run pertama + daftar perubahan untuk v2
app/                               app Flutter minimal: kamera live + daftar deteksi (model v1 dibundel)
```

## Status

| Fase | Isi | Status |
|---|---|---|
| 1. Fondasi data | database gizi TKPI, daftar kelas v1 | daftar kelas siap; database menunggu CSV TKPI |
| 2. Dataset & model | gabung dataset terbuka, box→mask (SAM), foto tambahan, latih YOLO26n-seg, export | v1 selesai (test box mAP50 0,78 / mask 0,76, hanya Combination); notebook v2 siap; uji foto sendiri (Bagian C) siap, ditunda dulu |
| 3. App Flutter | kamera + deteksi on-device, estimasi porsi, log harian, target dari BMR/TDEE | deteksi live siap dicoba di HP (`app/`); porsi, log, target belum |
| 4. Backend & chatbot | akun & sinkronisasi log, chatbot dengan tool calling ke data TKPI + log user | belum (pilihan backend belum diputuskan) |

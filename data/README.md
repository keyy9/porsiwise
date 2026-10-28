# Data Porsiwise

Folder ini berisi dua fondasi yang dipakai bersama oleh model kamera, app Flutter, dan backend:

1. **Database gizi** (`gizi.sqlite`), dibangun dari TKPI dengan `tools/tkpi_ingest.py`.
2. **Daftar kelas deteksi** (`kelas_v1.csv`). Ini kontrak antara model YOLO dan database gizi.

## 1. `kelas_v1.csv`

| Kolom | Arti |
|---|---|
| `class_id` | Indeks kelas YOLO (0..n-1). Model selalu memakai semua id, supaya nomor kelas tidak berubah waktu data kelas B ditambahkan. Kelas yang belum punya data cuma punya 0 instance. |
| `slug` | Nama label di `data.yaml` YOLO. |
| `nama` | Nama tampilan di app. |
| `kelompok` | Golongan pangan untuk ringkasan gizi (karbohidrat, lauk_hewani, lauk_nabati, sayur, kuah, pelengkap, buah, kue, minuman). |
| `tipe_label` | `komponen` = dilabel per lauk di piring (gaya nasi campur). `hidangan` = satu mask untuk satu porsi utuh (mie ayam, soto, nasi goreng). |
| `prioritas` | `A` = masuk model v1 (sudah ada data terbuka). `B` = backlog, perlu foto/anotasi tambahan. |
| `sumber_terbuka` | Dataset terbuka yang punya kelas ini. |
| `remap_dari` | Nama kelas di dataset sumber yang digabung ke kelas ini (format `Dataset:nama kelas`). Dipakai skrip penggabung dataset. |
| `tkpi_kata_kunci` | Kata kunci untuk mencari item TKPI (alternatif dipisah `\|`). |
| `porsi_default_g` / `porsi_urt` | Porsi awal sebelum user menggeser slider. **Estimasi awal**, perlu divalidasi dengan daftar URT/bahan makanan penukar Kemenkes. |
| `catatan_anotasi` | Aturan khusus saat melabel kelas tersebut. |

Kalori **tidak** ditanam di nama kelas, beda dengan model lama (`Nasi -180 kkal-`). Nilai gizi diambil lewat `class_food_map`, jadi angkanya bisa dikoreksi tanpa training ulang.

## 2. Membangun database gizi

```bash
python3 tools/tkpi_ingest.py --tkpi path/ke/tkpi.csv --kelas data/kelas_v1.csv --out build
```

Hasil di `build/`:

- `gizi.sqlite`: tabel `foods` (per 100 g), `classes`, `class_food_map`, dan view `class_nutrition_100g`.
- `mapping_suggestions.csv`: 3 kandidat item TKPI per kelas. Kandidat dengan skor ≥ 0.75 otomatis dicentang di kolom `pilih`.
- `quality_report.md`: baris yang dibuang, duplikat, dan data yang energinya tidak cocok dengan hitungan Atwater (4P + 4KH + 9L).
- `foods_clean.csv`: data pangan yang sudah dibersihkan.

Alur review mapping:

1. Buka `mapping_suggestions.csv`.
2. Centang (`pilih` = 1) item yang benar. Untuk hidangan komposit, tambahkan baris dengan kolom `porsi_fraksi` yang totalnya 1 (contoh mie ayam = mi 0,6 + ayam 0,4).
3. Jalankan ulang dengan `--mapping mapping_suggestions.csv`. Status mapping berubah dari `auto` menjadi `confirmed`.

Format CSV yang dikenali:

- Dataset Kaggle *Indonesian Food and Drink Nutrition Dataset*: `id, calories, proteins, fat, carbohydrate, name, image`.
- Export lain berkolom Indonesia, mis. `Nama Bahan; Energi (Kal); Protein (g); Lemak (g); Karbohidrat (g); Serat (g)`, termasuk desimal koma.

## 3. Sumber dataset gambar (untuk dikumpulkan)

| Dataset | Isi | Anotasi | Lisensi |
|---|---|---|---|
| [CalorEase](https://universe.roboflow.com/gulakupedia/calorease-lps84) | 4.241 gambar, 10 kelas (ayam goreng, mie ayam, nasi putih, pempek, rendang, sambal, sate, soto, tahu goreng, tempe goreng) | Bounding box | CC0 |
| [Indonesian Combination Food](https://universe.roboflow.com/machinelearning-4tyun/indonesian-combination-food) | 777 gambar asli, 50 kelas komponen piring | Bounding box | CC BY 4.0 |
| [Calory](https://universe.roboflow.com/ayu-asipq/calory) | 988 gambar, 13 kelas (kalori di nama kelas, buang kelas `food-z7P4`) | Bounding box | CC BY 4.0 |
| [FoodSeg103](https://datasetninja.com/food-seg-103) | 7.118 gambar, 103 kelas bahan (umumnya masakan Barat) | Mask piksel | Apache 2.0 |
| [Indonesian Food Image (ITS)](https://data.mendeley.com/datasets/vtjd68bmwt) | 10 kelas hidangan (bakso, gado-gado, gudeg, rawon, dll.) | Klasifikasi saja | CC BY 4.0, gambar hasil scraping Google |

Catatan:

- **Belum ketemu dataset segmentasi makanan Indonesia yang terbuka.** Box dari dataset di atas diubah jadi mask memakai SAM (box sebagai prompt), lalu dicek manual.
- **FoodSeg103 opsional**, untuk pra-latih segmentasi bahan umum (nasi, telur, mi).
- **CC BY 4.0 wajib atribusi.** Catat sumber tiap gambar di `ATTRIBUTION.md` saat dataset digabung.
- **Kelas yang dibuang saat penggabungan:** `kari kentang` (Combination) dan `food-z7P4` (Calory). Semua kelas lain dari ketiga dataset deteksi sudah dipetakan di kolom `remap_dari`.

## 4. Aturan anotasi v1

1. **Format:** polygon segmentasi YOLO-seg.
   - **Lauk satuan dilabel per buah:** tiap potong ayam, tempe, tahu, telur, kerupuk, perkedel, nugget, tiap tusuk sate, dan tiap butir bakso. App nanti menghitung porsinya dari jumlah × berat per buah.
   - **Makanan curah dilabel satu mask per gundukan:** nasi, mie, sayur, sambal.
   - Hasil training v1 menunjukkan label yang campur (kadang per buah, kadang per kelompok) bikin model bingung, jadi aturan ini perlu dijaga konsisten di semua sumber.
2. **Ikuti `tipe_label`:** lauk di piring dilabel per komponen, hidangan semangkuk dilabel utuh.
3. **Yang tidak dilabel:** piring, sendok, tangan, dan kuah yang tidak masuk kelas. Objek yang terlihat < 30% dilewati.
4. **Variasi foto:**
   - sudut atas (90°) dan miring (±45°)
   - cahaya warung, siang, redup
   - piring putih/bermotif, styrofoam, daun pisang
   - satu menu dan piring campur
5. **Target v1:** ≥ 150 instance per kelas prioritas A (idealnya 300+). Split 70/20/10 per gambar. Set test sebaiknya foto sendiri dari dunia nyata, bukan potongan dataset yang sama.
6. **Data porsi (penting untuk estimasi gram):** untuk sebagian foto (target ±200 piring), timbang tiap lauk dan catat beratnya. Data foto + berat ini dipakai untuk melatih dan menguji estimasi porsi.

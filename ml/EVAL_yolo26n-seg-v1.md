# Evaluasi run yolo26n-seg-v1

Run pertama dari notebook training, dijalankan user di Colab T4 pada 24–25 Okt 2026 (Ultralytics 8.4.187).

## Setup

- Data: hanya **Combination v13** (CalorEase dan Calory terlewat karena `version` belum diisi).
- Jumlah foto unik: 769 (615 train, 77 val, 77 test). Train v13 sudah diaugmentasi Roboflow 3×, jadi tercatat 1.845 gambar train.
- 42 dari 51 kelas punya data. Tanpa data: mie_ayam, sayur_bayam, sayur_sop, soto, pempek, gado_gado, apel, risol_mayo, klepon.
- Mask: SAM 2.1-b dari box. 5.526 mask SAM, 6 fallback, 310 instance (5,6%) dengan rasio luas janggal.
- Training: 135 epoch (early stopping, patience 30), best.pt dari epoch 105, 2,16 jam.

## Hasil

| split | box mAP50 | box mAP50-95 | mask mAP50 | mask mAP50-95 |
|---|---|---|---|---|
| val (77 gambar, 172 instance) | 0,680 | 0,557 | 0,641 | 0,526 |
| test (77 gambar, 158 instance) | 0,781 | 0,617 | 0,762 | 0,590 |

- Val mask mAP50-95 0,48 di epoch 30, lalu datar 0,51–0,53 sejak epoch ±70 (sd 0,009). Model terbatas data, bukan jumlah epoch.
- Kelas kuat (mask mAP50 rata-rata val+test, kelas dengan ≥ 11 instance evaluasi): capcay 0,99, donat 0,98, nasi_putih 0,90, rendang 0,90, telur_ceplok 0,82, ayam_goreng 0,76.
- Kelas lemah: sate 0,28, telur_rebus 0,30, kerupuk 0,37, ikan_goreng 0,40, sambal 0,42, lalapan 0,45, tahu_goreng 0,50.
- Lemahnya kelas tidak sejalan dengan jumlah data: telur_rebus (131 instance unik), tahu (166), dan lalapan (199) tetap rendah.
- Confusion: nugget lebih sering ditebak ayam_goreng; ±1/3 telur_rebus tidak terdeteksi; kerupuk, tahu, dan lalapan sering terlewat.

## Diagnosis

1. **Label lauk satuan tidak konsisten.** Sate, telur rebus, bakso, dan nugget kadang dilabel per buah, kadang satu kotak untuk sekelompok. Contoh prediksi: 9 potong telur jadi 1 mask, bakso 1 dari ±7 terdeteksi.
   - Keputusan: **per buah** untuk semua lauk satuan (kelas_v1 diperbarui). App menghitung porsinya dari jumlah × berat per buah.
2. **Lalapan heterogen.** 8 kelas sayur mentah digabung jadi satu kelas. Prioritas rendah karena kalorinya kecil.
3. **Mask SAM.** Ada yang bocor keluar kotak, ada yang cuma secuil, ada yang pecah-pecah. Bug: urutan mask bisa tertukar kalau SAM membuang mask berskor < 0,25 (dampak kecil, ≤ 6 gambar).
   - Mask mAP mengukur kemiripan dengan mask SAM, bukan mask manusia.
4. **Domain gap.** Foto asli user (nampan stainless) beda dengan dataset (foto internet). Isi nampan menurut user: lele goreng sambel, nasi, tumis terong, sambel, ditambah mie di kompartemen kiri atas.
   - Nasi → nasi_putih 0,87 ✅
   - Sambel → salah satu deteksi sambal ✅
   - Lele goreng sambel (kanan atas) → ayam_goreng 0,41 ❌. Seharusnya ikan_goreng: kelas lemah, hanya 17 contoh unik, dan lelenya tertutup sambal.
   - Tumis terong → sambal ❌. Kelasnya belum ada di kelas_v1.
   - Mie dan 2 pisang tidak terdeteksi.
   - Hasil: 2 dari 5 makanan benar. Jauh di bawah angka test set.
   - Rencana: kumpulkan foto makan sendiri sebagai test set dunia nyata dan sumber training. Kelas baru (mis. tumis terong) masuk kelas_v2 berdasarkan frekuensi di foto sendiri, bukan ditambahkan sebelum ada datanya.

## Export

- **TFLite `w8a32`** (3,2 MB):
  - Metadata: task segment, 51 nama kelas, imgsz 640, end2end False.
  - Input [1,3,640,640], output [1,87,8400] + proto [1,32,160,160].
  - Sudah dijalankan di Linux: deteksi masuk akal.
- **Core ML int8** (`.mlpackage.zip`, 2,6 MB):
  - Struktur: top-level `best.mlpackage`, STORED/DEFLATE, tanpa ZIP64, ada Manifest.json.
  - mlProgram dengan input image 640, output [1,300,38] + proto, end2end True.
  - Belum dites di perangkat iOS. Ada warning `overflow encountered in cast` saat konversi.

## Perubahan untuk v2

- Ketiga sumber wajib lengkap (notebook berhenti kalau ada yang belum siap).
- Salinan augmentasi Roboflow dibuang, sehingga jumlah data jujur dan epoch ±3× lebih cepat.
- Foto duplikat lintas split dideteksi dengan dHash dan dihapus dari val/test.
- Mask SAM dipotong ke kotak, serpihan < 10% dibuang, dan fallback elips untuk mask < 8% kotak. Bug urutan mask diperbaiki.
- Uji kualitas mask (IoU) terhadap polygon buatan manusia: kotak vs elips vs SAM mentah vs SAM dirapikan.
- Profil cara pelabelan per kelas per sumber.
- Evaluasi per sumber + perbandingan v1 vs v2 di test Combination.
- Model card mengambil epoch terbaik dari best.pt. Di v1, model card menulis epoch 84 karena memakai mask mAP saja, sedangkan Ultralytics memilih epoch 105.

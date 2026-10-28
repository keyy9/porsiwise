# Porsiwise: uji deteksi di HP

Kamera live + deteksi makanan. Model v1 sudah ada di `assets/models/`, jadi tinggal build.

## 1. Siapkan laptop (sekali saja)

- Install **Flutter stable terbaru** + Android Studio: https://docs.flutter.dev/get-started
  Sudah punya Flutter? Jalankan `flutter upgrade`.
- Jalankan `flutter doctor --android-licenses` lalu jawab `y` semua.

## 2. Samsung A33 (paling gampang, bisa dari Windows/Mac/Linux)

1. Di HP: Settings → About phone → Software information → ketuk **Build number** 7×.
   Lalu Settings → Developer options → nyalakan **USB debugging**. Colok kabel, pilih "Allow".
2. Buka terminal di folder ini (yang ada `pubspec.yaml`), lalu:

   ```
   flutter create --org com.porsiwise --platforms android --platforms ios .
   flutter run --release
   ```

   Build pertama bisa 10–20 menit. Itu normal.
3. Di HP, izinkan akses kamera, lalu arahkan ke makanan.

## 3. iPhone 15 (butuh Mac + Xcode)

1. Jalankan perintah `flutter create` yang sama, lalu:

   ```
   plutil -replace NSCameraUsageDescription -string "Kamera dipakai untuk mendeteksi makanan." ios/Runner/Info.plist
   open ios/Runner.xcworkspace
   ```

2. Di Xcode: klik **Runner** → tab **Signing & Capabilities** → **Team** → pilih Apple ID kamu (Personal Team).
3. Di iPhone: Settings → Privacy & Security → **Developer Mode** → On. Colok kabel ke Mac.
4. `flutter run --release`
   Kalau muncul "Untrusted Developer": Settings → General → VPN & Device Management → Trust.

Pakai Apple ID gratis, app-nya aktif 7 hari. Setelah itu tinggal `flutter run --release` lagi.

## Ganti model (nanti, waktu v2 jadi)

Taruh file baru di `assets/models/` dengan **nama baru** (misalnya `...-v2...`), lalu ubah 3 baris di atas `lib/main.dart`.
Jangan menimpa file dengan nama yang sama: plugin menyimpan salinan model lama di HP dan tidak akan memperbaruinya.

## Kalau ada masalah

- Error soal `minSdk` atau `compileSdk`: jalankan `flutter upgrade`, lalu ulangi.
- HP tidak muncul di `flutter devices` (Windows): install Samsung USB Driver, colok ulang.
- Panel bawah menampilkan tulisan merah "Model gagal dimuat": kirim teksnya ke Claude.

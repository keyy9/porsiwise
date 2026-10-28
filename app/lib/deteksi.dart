import 'dart:math' as math;

import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Satu baris di daftar deteksi. Deteksi dengan kelas yang sama digabung.
class ItemDeteksi {
  const ItemDeteksi({
    required this.slug,
    required this.jumlah,
    required this.confidence,
  });

  final String slug;
  final int jumlah;

  /// Confidence tertinggi di antara deteksi kelas ini (0–1).
  final double confidence;

  String get nama => namaKelas[slug] ?? slug.replaceAll('_', ' ');
}

/// Gabungkan hasil deteksi per kelas, urut dari confidence tertinggi.
List<ItemDeteksi> ringkas(List<YOLOResult> hasil) {
  final jumlah = <String, int>{};
  final conf = <String, double>{};
  for (final r in hasil) {
    jumlah[r.className] = (jumlah[r.className] ?? 0) + 1;
    conf[r.className] = math.max(conf[r.className] ?? 0.0, r.confidence);
  }
  return [
    for (final slug in jumlah.keys)
      ItemDeteksi(slug: slug, jumlah: jumlah[slug]!, confidence: conf[slug]!),
  ]..sort((a, b) => b.confidence.compareTo(a.confidence));
}

/// Nama tampilan per kelas model (dari data/kelas_v1.csv: slug → nama).
const namaKelas = <String, String>{
  'nasi_putih': 'Nasi putih',
  'nasi_goreng': 'Nasi goreng',
  'nasi_kuning': 'Nasi kuning',
  'mie_goreng': 'Mie goreng',
  'mie_ayam': 'Mie ayam',
  'kentang_goreng': 'Kentang goreng',
  'ayam_goreng': 'Ayam goreng',
  'ayam_bakar': 'Ayam bakar',
  'rendang': 'Rendang',
  'sate': 'Sate',
  'ikan_goreng': 'Ikan goreng',
  'telur_ceplok': 'Telur ceplok',
  'telur_dadar': 'Telur dadar',
  'telur_rebus': 'Telur rebus',
  'bakso': 'Bakso (butir)',
  'tempe_goreng': 'Tempe goreng',
  'tahu_goreng': 'Tahu goreng',
  'perkedel': 'Perkedel',
  'sayur_bayam': 'Sayur bayam',
  'sayur_sop': 'Sayur sop',
  'capcay': 'Capcay',
  'tumis_kangkung': 'Tumis kangkung',
  'tumis_buncis': 'Tumis buncis',
  'lalapan': 'Lalapan / sayur rebus',
  'soto': 'Soto',
  'pempek': 'Pempek',
  'sambal': 'Sambal',
  'kerupuk': 'Kerupuk',
  'pisang': 'Pisang',
  'lontong': 'Lontong',
  'bihun': 'Bihun goreng',
  'roti_tawar': 'Roti tawar',
  'ayam_suwir': 'Ayam suwir',
  'ayam_asam_manis': 'Ayam asam manis',
  'ikan_bakar': 'Ikan bakar',
  'telur_balado': 'Telur balado',
  'nugget': 'Nugget',
  'sosis': 'Sosis',
  'kering_tempe': 'Kering tempe',
  'sambal_goreng_kentang': 'Sambal goreng kentang',
  'gado_gado': 'Gado-gado',
  'sambal_kacang': 'Bumbu kacang',
  'serundeng': 'Serundeng',
  'apel': 'Apel',
  'risol_mayo': 'Risol mayo',
  'klepon': 'Klepon',
  'donat': 'Donat',
  'martabak_manis': 'Martabak manis',
  'teh_manis': 'Teh manis',
  'kopi': 'Kopi',
  'air_putih': 'Air putih',
};

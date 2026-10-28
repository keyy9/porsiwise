import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:porsiwise/deteksi.dart';
import 'package:porsiwise/main.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

YOLOResult _hasil(String slug, double confidence) => YOLOResult(
  classIndex: 0,
  className: slug,
  confidence: confidence,
  boundingBox: Rect.zero,
  normalizedBox: Rect.zero,
);

void main() {
  test('nama kelas lengkap untuk 51 kelas model v1', () {
    expect(namaKelas.length, 51);
    expect(namaKelas['nasi_putih'], 'Nasi putih');
  });

  test('ringkas menggabungkan kelas yang sama', () {
    final items = ringkas([
      _hasil('sate', 0.6),
      _hasil('nasi_putih', 0.9),
      _hasil('sate', 0.8),
    ]);
    expect(items.map((i) => i.nama), ['Nasi putih', 'Sate']);
    expect(items[1].jumlah, 2);
    expect(items[1].confidence, 0.8);
  });

  testWidgets('panel menampilkan daftar deteksi', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: PanelDeteksi(
            items: ringkas([_hasil('sate', 0.8), _hasil('sate', 0.7)]),
            fps: 12,
            jeda: false,
            onJeda: () {},
          ),
        ),
      ),
    );
    expect(find.text('Sate ×2'), findsOneWidget);
    expect(find.text('80%'), findsOneWidget);
    expect(find.text('model v1 · 12 FPS'), findsOneWidget);
  });
}

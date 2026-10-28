import 'dart:async' show unawaited;
import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

import 'deteksi.dart';

// Model dari notebook (Bagian B). Kalau ganti model, pakai NAMA FILE BARU (mis. ...-v2...):
// plugin menyalin model ke memori HP per nama file dan tidak menimpa salinan lama.
const versiModel = 'v1';
const modelAndroid = 'assets/models/makanan-yolo26n-seg-v1_w8a32.tflite';
const modelIos = 'assets/models/makanan-yolo26n-seg-v1.mlpackage.zip';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const PorsiwiseApp());
}

class PorsiwiseApp extends StatelessWidget {
  const PorsiwiseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Porsiwise',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorSchemeSeed: const Color(0xFF4CAF50),
      ),
      home: const KameraPage(),
    );
  }
}

/// Kamera live: kotak + mask digambar oleh plugin, daftar makanan di panel bawah.
class KameraPage extends StatefulWidget {
  const KameraPage({super.key});

  @override
  State<KameraPage> createState() => _KameraPageState();
}

class _KameraPageState extends State<KameraPage> {
  final _kamera = YOLOViewController();
  List<ItemDeteksi> _items = const [];
  double _fps = 0;
  bool _jeda = false;
  String? _error;

  @override
  void dispose() {
    _kamera.dispose();
    super.dispose();
  }

  void _onResult(List<YOLOResult> hasil) {
    if (_jeda || !mounted) return;
    setState(() => _items = ringkas(hasil));
  }

  void _onMetrics(YOLOPerformanceMetrics metrics) {
    if (_jeda || !mounted) return;
    setState(() => _fps = metrics.fps);
  }

  void _onModelError(Object error, String modelPath, YOLOTask? task) {
    if (!mounted) return;
    setState(() => _error = '$error');
  }

  void _toggleJeda() {
    setState(() => _jeda = !_jeda);
    unawaited(_jeda ? _kamera.pause() : _kamera.resume());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: YOLOView(
                modelPath: Platform.isIOS ? modelIos : modelAndroid,
                task: YOLOTask.segment,
                controller: _kamera,
                onResult: _onResult,
                onPerformanceMetrics: _onMetrics,
                onModelError: _onModelError,
              ),
            ),
            PanelDeteksi(
              items: _items,
              fps: _fps,
              jeda: _jeda,
              error: _error,
              onJeda: _toggleJeda,
            ),
          ],
        ),
      ),
    );
  }
}

/// Daftar makanan yang sedang terdeteksi + tombol jeda.
class PanelDeteksi extends StatelessWidget {
  const PanelDeteksi({
    super.key,
    required this.items,
    required this.fps,
    required this.jeda,
    required this.onJeda,
    this.error,
  });

  final List<ItemDeteksi> items;
  final double fps;
  final bool jeda;
  final VoidCallback onJeda;
  final String? error;

  @override
  Widget build(BuildContext context) {
    final teks = Theme.of(context).textTheme;
    return Material(
      color: const Color(0xFF151A15),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Text('Terdeteksi', style: teks.titleMedium),
                const Spacer(),
                Text(
                  fps > 0
                      ? 'model $versiModel · ${fps.toStringAsFixed(0)} FPS'
                      : 'model $versiModel',
                  style: teks.bodySmall,
                ),
              ],
            ),
            const SizedBox(height: 4),
            SizedBox(height: 150, child: _daftar(teks)),
            const SizedBox(height: 8),
            FilledButton.icon(
              onPressed: onJeda,
              icon: Icon(jeda ? Icons.play_arrow : Icons.pause),
              label: Text(jeda ? 'Lanjut' : 'Jeda'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _daftar(TextTheme teks) {
    final pesan = error;
    if (pesan != null) {
      return SingleChildScrollView(
        child: Text(
          'Model gagal dimuat:\n$pesan',
          style: teks.bodyMedium?.copyWith(color: Colors.redAccent),
        ),
      );
    }
    if (items.isEmpty) {
      return Center(
        child: Text(
          jeda ? 'Dijeda' : 'Arahkan kamera ke makanan…',
          style: teks.bodyMedium,
        ),
      );
    }
    return ListView(
      padding: EdgeInsets.zero,
      children: [
        for (final item in items)
          ListTile(
            dense: true,
            contentPadding: EdgeInsets.zero,
            title: Text(
              item.jumlah > 1 ? '${item.nama} ×${item.jumlah}' : item.nama,
            ),
            trailing: Text('${(item.confidence * 100).round()}%'),
          ),
      ],
    );
  }
}

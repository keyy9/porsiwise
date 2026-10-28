# Helper untuk pipeline training Porsiwise (dipakai di notebook Colab).
# Semua fungsi menerima path sebagai argumen supaya bisa dites di luar Colab.

import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
SPLITS = ("train", "val", "test")
RF_AUG_RE = re.compile(r"\.rf\.[0-9a-fA-F]+$")  # Roboflow: <nama>_jpg.rf.<hash>.jpg


# ---------------------------------------------------------------- kelas & label
def norm_name(s):
    """Seragamkan nama kelas: buang '[129 kal per 100gr]' / '-180 kkal-', huruf kecil, spasi tunggal."""
    s = re.sub(r"\[.*?\]", " ", str(s))
    s = re.sub(r"-\s*\d+\s*k?kal\s*-?", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def load_classes(csv_path):
    """Baca kelas_v1.csv -> (rows, names {id: slug}, remap {(dataset, nama_norm): id})."""
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    names = {int(r["class_id"]): r["slug"] for r in rows}
    assert sorted(names) == list(range(len(names))), "class_id harus berurutan 0..n-1"
    remap = {}
    for r in rows:
        for part in (r.get("remap_dari") or "").split("|"):
            if ":" in part:
                ds, nm = part.split(":", 1)
                remap[(ds.strip(), norm_name(nm))] = int(r["class_id"])
    return rows, names, remap


def read_source_names(src_dir):
    """Ambil daftar nama kelas dari data.yaml export Roboflow (list atau dict)."""
    with open(Path(src_dir) / "data.yaml", encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    names = d["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k: int(k))]
    return [str(n) for n in names]


def parse_label_line(line):
    """'cls cx cy w h' -> (cls, 'box', [4 nilai]); 'cls x1 y1 x2 y2 ...' -> (cls, 'poly', [...])."""
    p = line.split()
    if len(p) < 5:
        return None
    try:
        cls = int(float(p[0]))
        vals = [float(x) for x in p[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        return cls, "box", vals
    if len(vals) >= 6 and len(vals) % 2 == 0:
        return cls, "poly", vals
    return None


def read_labels(lbl_path):
    if not Path(lbl_path).exists():
        return []
    return [p for p in (parse_label_line(x) for x in Path(lbl_path).read_text(encoding="utf-8").splitlines()) if p]


def format_line(cls, vals):
    return f"{cls} " + " ".join(f"{v:.6f}".rstrip("0").rstrip(".") if v else "0" for v in vals)


def find_image(img_dir, stem):
    for ext in IMG_EXTS:
        for cand in (Path(img_dir) / f"{stem}{ext}", Path(img_dir) / f"{stem}{ext.upper()}"):
            if cand.exists():
                return cand
    return None


def source_key(stem):
    """Kunci foto asli: salinan augmentasi Roboflow ('..._jpg.rf.<hash>') punya kunci yang sama."""
    return RF_AUG_RE.sub("", stem)


def source_tag(stem):
    """Nama dataset sumber dari nama file gabungan '<tag>__<nama asli>'."""
    return stem.split("__", 1)[0] if "__" in stem else "?"


def label_bounds(kind, vals):
    """Kotak (x1, y1, x2, y2) ternormalisasi dari baris box atau polygon."""
    if kind == "box":
        cx, cy, w, h = vals
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    pts = np.asarray(vals, dtype=float).reshape(-1, 2)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def is_rectangle(vals, tol=0.97):
    """Polygon <= 5 titik yang luasnya ~ luas kotaknya = sebenarnya bounding box, bukan bentuk objek."""
    pts = np.asarray(vals, dtype=float).reshape(-1, 2)
    if len(pts) > 5:
        return False
    x1, y1, x2, y2 = label_bounds("poly", vals)
    box_area = (x2 - x1) * (y2 - y1)
    return box_area > 0 and polygon_area(pts) / box_area >= tol


def match_class_names(source_names, rows, tag=None, remap=None):
    """Petakan nama kelas sumber -> class_id kelas_v1.

    Urutan: kolom remap_dari (untuk dataset publik), lalu cocok langsung ke slug atau nama kelas
    (untuk project milik sendiri yang memakai nama seperti 'nasi_putih'). Tidak cocok -> None.
    """
    direct = {}
    for r in rows or []:
        cid = int(r["class_id"])
        for key in (r["slug"], r.get("nama", "")):
            if key:
                direct.setdefault(norm_name(key), cid)
    out = {}
    for i, n in enumerate(source_names):
        cid = remap.get((tag, norm_name(n))) if remap and tag else None
        out[i] = cid if cid is not None else direct.get(norm_name(n))
    return out


def roboflow_api_key():
    """API key Roboflow dari Colab Secrets (ROBOFLOW_API_KEY), atau minta diketik (tidak ditampilkan)."""
    try:
        from google.colab import userdata

        key = userdata.get("ROBOFLOW_API_KEY")
        if key:
            return key
    except Exception:
        pass
    from getpass import getpass

    return getpass("Tempel API key Roboflow (tidak akan ditampilkan): ")


# ---------------------------------------------------------------- gabung dataset
def merge_sources(source_tags, raw_root, out_root, remap, dedupe_rf_aug=True, rows=None):
    """Gabung beberapa export Roboflow (format YOLO) jadi satu dataset dengan kelas Porsiwise.

    - Split asli tiap sumber dipertahankan (train->train, valid->val, test->test) supaya
      gambar hasil augmentasi Roboflow tidak bocor ke val/test.
    - dedupe_rf_aug: dari salinan augmentasi Roboflow ('..rf.<hash>') cuma satu yang disimpan,
      supaya jumlah data jujur dan epoch tidak 3x lebih lama. Augmentasi dilakukan Ultralytics.
    - Nama kelas dipetakan lewat remap_dari; kalau rows (kelas_v1) diberikan, nama yang sama dengan
      slug/nama kelas juga dikenali (project milik sendiri).
    - Label kelas yang tidak ada di kelas_v1 dibuang; gambar yang jadi kosong dibuang.
    - Baris polygon asli (buatan manusia) disalin juga ke human_polygons/<split>/ untuk uji kualitas SAM.
    """
    raw_root, out_root = Path(raw_root), Path(out_root)
    report = {}
    for tag in source_tags:
        src = raw_root / tag
        names = read_source_names(src)
        idx_map = match_class_names(names, rows, tag, remap)
        rep = {
            "kelas_sumber": names,
            "kelas_tidak_dipetakan": [n for i, n in enumerate(names) if idx_map[i] is None],
            "gambar_masuk": 0, "gambar_keluar": 0, "gambar_dibuang_kosong": 0, "salinan_augmentasi_dibuang": 0,
            "instance_masuk": 0, "instance_keluar": 0, "instance_polygon_asli": 0, "instance_dibuang": Counter(),
            "per_split": Counter(),
        }
        for split_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            split = SPLIT_ALIASES.get(split_dir.name)
            if not split or not (split_dir / "images").exists():
                continue
            out_img, out_lbl = out_root / "images" / split, out_root / "labels" / split
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)
            seen = set()
            for img in sorted((split_dir / "images").iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                rep["gambar_masuk"] += 1
                key = source_key(img.stem)
                if dedupe_rf_aug and key in seen:
                    rep["salinan_augmentasi_dibuang"] += 1
                    continue
                src_lbl = split_dir / "labels" / f"{img.stem}.txt"
                lines = src_lbl.read_text(encoding="utf-8").splitlines() if src_lbl.exists() else []
                kept, kept_poly = [], []
                for raw in lines:
                    parsed = parse_label_line(raw)
                    if not parsed:
                        continue
                    rep["instance_masuk"] += 1
                    cls, kind, vals = parsed
                    new_cls = idx_map.get(cls)
                    if new_cls is None:
                        rep["instance_dibuang"][names[cls] if cls < len(names) else str(cls)] += 1
                        continue
                    if kind == "poly" and is_rectangle(vals):
                        # kotak yang disimpan sebagai polygon 4 titik -> perlakukan sebagai box supaya diproses SAM
                        x1, y1, x2, y2 = label_bounds(kind, vals)
                        kind, vals = "box", [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                        rep["polygon_kotak_jadi_box"] = rep.get("polygon_kotak_jadi_box", 0) + 1
                    kept.append(format_line(new_cls, vals))
                    if kind == "poly":
                        kept_poly.append(format_line(new_cls, vals))
                if not kept:
                    rep["gambar_dibuang_kosong"] += 1
                    continue
                seen.add(key)
                new_stem = f"{tag}__{img.stem}"
                shutil.copy2(img, out_img / f"{new_stem}{img.suffix.lower()}")
                (out_lbl / f"{new_stem}.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
                if kept_poly:
                    hp = out_root / "human_polygons" / split
                    hp.mkdir(parents=True, exist_ok=True)
                    (hp / f"{new_stem}.txt").write_text("\n".join(kept_poly) + "\n", encoding="utf-8")
                    rep["instance_polygon_asli"] += len(kept_poly)
                rep["gambar_keluar"] += 1
                rep["instance_keluar"] += len(kept)
                rep["per_split"][split] += 1
        rep["instance_dibuang"] = dict(rep["instance_dibuang"])
        rep["per_split"] = dict(rep["per_split"])
        report[tag] = rep
    return report


def count_instances(out_root, names, unique=True):
    """Hitung instance per kelas per split -> ({split: Counter(class_id)}, {split: jumlah gambar}).

    unique=True: salinan augmentasi Roboflow dari foto yang sama dihitung sekali.
    """
    out_root = Path(out_root)
    counts, n_images = {s: Counter() for s in SPLITS}, Counter()
    for s in SPLITS:
        d = out_root / "labels" / s
        if not d.exists():
            continue
        seen = set()
        for lbl in sorted(d.glob("*.txt")):
            key = source_key(lbl.stem)
            if unique and key in seen:
                continue
            seen.add(key)
            n_images[s] += 1
            for cls, _kind, _vals in read_labels(lbl):
                counts[s][cls] += 1
    return counts, n_images


def class_table(rows, counts, min_train=150):
    """Tabel kelas x split (instance unik), plus status apakah data train sudah cukup."""
    table = []
    for r in rows:
        cid = int(r["class_id"])
        tr, va, te = counts["train"][cid], counts["val"][cid], counts["test"][cid]
        if tr == 0:
            status = "belum ada data"
        elif tr < min_train:
            status = f"kurang (< {min_train})"
        else:
            status = "cukup"
        table.append({"id": cid, "kelas": r["slug"], "prioritas": r["prioritas"],
                      "train": tr, "val": va, "test": te, "status": status})
    return table


def instance_profile(out_root, names, class_ids=None, splits=("train",)):
    """Profil cara pelabelan per kelas per sumber.

    Kelas yang dilabel per buah punya banyak instance kecil per gambar; kelas yang dilabel per kelompok
    punya satu kotak besar. Kalau dua sumber beda pola untuk kelas yang sama, labelnya tidak konsisten.
    """
    stats = defaultdict(lambda: {"gambar": 0, "instance": 0, "luas": [], "multi": 0, "maks": 0})
    for s in splits:
        d = Path(out_root) / "labels" / s
        if not d.exists():
            continue
        seen = set()
        for lbl in sorted(d.glob("*.txt")):
            key = source_key(lbl.stem)
            if key in seen:
                continue
            seen.add(key)
            per_cls = defaultdict(list)
            for cls, kind, vals in read_labels(lbl):
                if class_ids is None or cls in class_ids:
                    x1, y1, x2, y2 = label_bounds(kind, vals)
                    per_cls[cls].append(max(x2 - x1, 0) * max(y2 - y1, 0))
            for cls, areas in per_cls.items():
                st = stats[(cls, source_tag(lbl.stem))]
                st["gambar"] += 1
                st["instance"] += len(areas)
                st["luas"] += areas
                st["multi"] += len(areas) >= 2
                st["maks"] = max(st["maks"], len(areas))
    out = []
    for (cls, tag), st in sorted(stats.items()):
        out.append({"kelas": names.get(cls, str(cls)), "sumber": tag, "gambar": st["gambar"],
                    "instance": st["instance"], "instance_per_gambar": round(st["instance"] / st["gambar"], 2),
                    "maks_per_gambar": st["maks"], "gambar_isi_>=2": f"{st['multi'] / st['gambar']:.0%}",
                    "median_luas_kotak": f"{np.median(st['luas']):.1%}"})
    return out


def write_data_yaml(out_root, names):
    out_root = Path(out_root)
    cfg = {"path": str(out_root)}
    for s in SPLITS:
        d = out_root / "images" / s
        if d.exists() and any(d.iterdir()):
            cfg[s] = f"images/{s}"
    assert "train" in cfg and "val" in cfg, "Dataset gabungan butuh minimal split train dan val"
    cfg["names"] = {int(k): v for k, v in names.items()}
    path = out_root / "data.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return path


def write_subset_yaml(out_root, names, split, tag):
    """data.yaml yang val/test-nya cuma gambar dari satu sumber (untuk evaluasi per dataset).

    Mengembalikan (path_yaml, jumlah_gambar) atau (None, 0) kalau sumber itu tidak punya gambar di split tsb.
    """
    out_root = Path(out_root)
    d = out_root / "images" / split
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS and p.name.startswith(f"{tag}__")) \
        if d.exists() else []
    if not imgs:
        return None, 0
    sub = out_root / "subsets"
    sub.mkdir(exist_ok=True)
    lst = sub / f"{split}_{tag}.txt"
    lst.write_text("\n".join(str(p.resolve()) for p in imgs) + "\n", encoding="utf-8")
    cfg = {"path": str(out_root), "train": "images/train", "val": str(lst.resolve()), "test": str(lst.resolve()),
           "names": {int(k): v for k, v in names.items()}}
    yp = sub / f"{split}_{tag}.yaml"
    with open(yp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return yp, len(imgs)


# ---------------------------------------------------------------- duplikat lintas split
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path, size=8):
    """Difference hash 64-bit: tahan resize/kompresi, cocok untuk cari foto yang sama di dua split."""
    from PIL import Image

    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = np.packbits((a[:, 1:] > a[:, :-1]).flatten())
    return np.frombuffer(bits.tobytes(), dtype=">u8")[0].astype(np.uint64)


def _hamming(a, b):
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POPCOUNT[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


def find_cross_split_duplicates(out_root, max_dist=4, chunk=256):
    """Cari foto val/test yang (hampir) sama dengan foto di split lain -> kebocoran data."""
    out_root = Path(out_root)
    files, hashes = {}, {}
    for s in SPLITS:
        d = out_root / "images" / s
        fs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.exists() else []
        files[s] = fs
        hashes[s] = np.array([dhash(p) for p in fs], dtype=np.uint64)
    dups = []
    for eval_split, ref_split in (("val", "train"), ("test", "train"), ("test", "val")):
        a, b = hashes[eval_split], hashes[ref_split]
        if not len(a) or not len(b):
            continue
        for i0 in range(0, len(a), chunk):
            dist = _hamming(a[i0:i0 + chunk], b)
            j = dist.argmin(1)
            dmin = dist[np.arange(len(j)), j]
            for k in np.where(dmin <= max_dist)[0]:
                dups.append({"split": eval_split, "file": files[eval_split][i0 + k].name,
                             "mirip_dengan_split": ref_split, "mirip_dengan": files[ref_split][j[k]].name,
                             "jarak": int(dmin[k])})
    return dups


def remove_duplicates(out_root, dups):
    """Hapus salinan val/test dari pasangan duplikat (versi train tetap dipakai)."""
    out_root = Path(out_root)
    removed = Counter()
    for d in {(x["split"], x["file"]) for x in dups}:
        split, fname = d
        img = out_root / "images" / split / fname
        if not img.exists():
            continue
        img.unlink()
        for extra in (out_root / "labels" / split / f"{img.stem}.txt",
                      out_root / "human_polygons" / split / f"{img.stem}.txt"):
            extra.unlink(missing_ok=True)
        removed[(split, source_tag(img.stem))] += 1
    return {f"{s}/{t}": n for (s, t), n in sorted(removed.items())}


# ---------------------------------------------------------------- box -> mask (SAM)
def box_to_corners(cx, cy, w, h):
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return np.clip(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float), 0, 1)


def ellipse_polygon(cx, cy, w, h, n=32):
    """Elips di dalam kotak (ternormalisasi). Untuk makanan, tebakan bentuk yang lebih baik dari persegi."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + w / 2 * np.cos(t), cy + h / 2 * np.sin(t)], 1)
    return np.clip(pts, 0, 1)


def polygon_area(pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def image_size(path):
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (w, h)


def clean_mask(mask, box_xyxy, clip_to_box=True, min_frag_ratio=0.1, pad=2):
    """Rapikan mask SAM: potong bagian di luar kotak + buang serpihan kecil.

    Kotak anotasi = batas objek, jadi mask yang bocor keluar kotak (ikut piring/daun) pasti salah.
    Serpihan yang luasnya < min_frag_ratio x serpihan terbesar dianggap noise.
    """
    import cv2

    m = np.asarray(mask).astype(np.uint8)
    h, w = m.shape
    if clip_to_box:
        x1, y1, x2, y2 = box_xyxy
        x1, y1 = max(int(np.floor(x1)) - pad, 0), max(int(np.floor(y1)) - pad, 0)
        x2, y2 = min(int(np.ceil(x2)) + pad, w), min(int(np.ceil(y2)) + pad, h)
        clipped = np.zeros_like(m)
        clipped[y1:y2, x1:x2] = m[y1:y2, x1:x2]
        m = clipped
    if min_frag_ratio > 0 and m.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 2:
            areas = st[1:, cv2.CC_STAT_AREA]
            keep = np.where(areas >= min_frag_ratio * areas.max())[0] + 1
            m = np.isin(lab, keep).astype(np.uint8)
    return m


def mask_to_polygon(m):
    """Mask biner (H, W) -> polygon ternormalisasi (K, 2), pakai cara yang sama dengan Ultralytics."""
    from ultralytics.utils import ops

    h, w = m.shape
    seg = ops.masks2segments(np.asarray(m, dtype=np.uint8)[None])[0]
    return seg / np.array([w, h], dtype=np.float32) if len(seg) else seg


def make_sam_predict_fn(sam_weights="sam2.1_b.pt", device=None, clip_to_box=True, min_frag_ratio=0.1,
                        min_fill=0.08):
    """predict_fn berbasis Ultralytics SAM dengan prompt bounding box.

    Mengembalikan list dict per kotak (urutan = urutan kotak):
      {"poly": polygon ternormalisasi atau None, "fill": luas mask / luas kotak, "method": ...}
    Kotak yang mask-nya kosong / terlalu kecil (fill < min_fill) diberi poly=None -> di-fallback ke elips.
    """
    from ultralytics import SAM

    sam = SAM(sam_weights)

    def raw_masks(img_path, boxes_xyxy):
        """Mask mentah SAM per kotak (resolusi asli), urutan dijamin sama dengan urutan kotak."""
        res = sam(str(img_path), bboxes=boxes_xyxy, verbose=False, save=False, device=device, conf=0.0)[0]
        out = [None] * len(boxes_xyxy)
        if res.masks is None:
            return out
        data = res.masks.data.cpu().numpy().astype(bool)
        # SAM membuang mask berskor rendah; kolom cls menyimpan indeks kotak aslinya
        idx = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.arange(len(data))
        for j, i in enumerate(idx):
            if 0 <= i < len(out):
                out[i] = data[j]
        return out

    def predict(img_path, boxes_xyxy):
        results = []
        for m, box in zip(raw_masks(img_path, boxes_xyxy), boxes_xyxy):
            if m is None or not m.any():
                results.append({"poly": None, "fill": 0.0, "method": "kosong"})
                continue
            m = clean_mask(m, box, clip_to_box, min_frag_ratio)
            box_area = max(box[2] - box[0], 1) * max(box[3] - box[1], 1)
            fill = float(m.sum()) / box_area
            if fill < min_fill:
                results.append({"poly": None, "fill": fill, "method": "terlalu_kecil"})
                continue
            poly = mask_to_polygon(m)
            results.append({"poly": poly if len(poly) >= 3 else None, "fill": fill,
                            "method": "sam" if len(poly) >= 3 else "kosong"})
        return results

    predict.raw_masks = raw_masks
    predict.config = {"sam_weights": sam_weights, "clip_to_box": clip_to_box, "min_frag_ratio": min_frag_ratio,
                      "min_fill": min_fill}
    return predict


def convert_boxes_to_masks(out_root, predict_fn, splits=SPLITS, log_every=200, qa_low=0.15, qa_high=0.97):
    """Ubah label box jadi polygon pakai predict_fn(img_path, boxes_xyxy_pixel).

    - Bisa dilanjutkan (resume): file label yang semua barisnya sudah polygon dilewati.
    - Kalau mask kosong / terlalu kecil, dipakai elips di dalam kotak (instance tidak hilang).
    - Mengembalikan statistik + daftar instance mencurigakan (rasio luas mask/kotak ekstrem) untuk dicek manual.
    """
    out_root = Path(out_root)
    stats = Counter()
    suspicious = []
    for s in splits:
        lbl_dir, img_dir = out_root / "labels" / s, out_root / "images" / s
        if not lbl_dir.exists():
            continue
        files = sorted(lbl_dir.glob("*.txt"))
        for n, lbl in enumerate(files, 1):
            parsed = read_labels(lbl)
            boxes = [(cls, vals) for cls, kind, vals in parsed if kind == "box"]
            if not boxes:
                stats["dilewati_sudah_polygon"] += 1
                continue
            img = find_image(img_dir, lbl.stem)
            if img is None:
                stats["gambar_tidak_ketemu"] += 1
                continue
            w, h = image_size(img)
            xyxy = [[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                    for _, (cx, cy, bw, bh) in boxes]
            try:
                results = predict_fn(img, xyxy) or []
            except Exception as exc:  # satu gambar gagal tidak menghentikan semuanya
                stats["error_sam"] += 1
                print(f"[SAM error] {img.name}: {exc}")
                results = []
            out_lines = [format_line(cls, vals) for cls, kind, vals in parsed if kind == "poly"]
            for i, (cls, (cx, cy, bw, bh)) in enumerate(boxes):
                r = results[i] if i < len(results) else None
                if isinstance(r, dict):
                    poly, fill, method = r.get("poly"), r.get("fill"), r.get("method", "sam")
                else:  # predict_fn lama: langsung polygon
                    poly, fill, method = r, None, "kosong"
                poly = np.asarray(poly, dtype=float).reshape(-1, 2) if poly is not None else np.zeros((0, 2))
                if len(poly) < 3:
                    poly = ellipse_polygon(cx, cy, bw, bh)
                    stats[f"fallback_elips_{method}"] += 1
                else:
                    stats["mask_sam"] += 1
                    if fill is None:
                        fill = polygon_area(poly) / max(bw * bh, 1e-9)
                    if fill < qa_low or fill > qa_high:
                        suspicious.append({"split": s, "label": lbl.name, "class_id": cls, "rasio_luas": round(fill, 3)})
                out_lines.append(format_line(cls, np.clip(poly, 0, 1).reshape(-1).tolist()))
            tmp = lbl.with_suffix(".tmp")
            tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            tmp.replace(lbl)
            stats["gambar_dikonversi"] += 1
            if log_every and n % log_every == 0:
                print(f"  {s}: {n}/{len(files)} file label")
    return dict(stats), suspicious


def _raster(poly_norm, size_wh):
    """Polygon ternormalisasi -> mask biner (h, w)."""
    from PIL import Image, ImageDraw

    w, h = size_wh
    im = Image.new("L", (w, h), 0)
    pts = [(float(x) * w, float(y) * h) for x, y in np.asarray(poly_norm, dtype=float).reshape(-1, 2)]
    if len(pts) >= 3:
        ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, dtype=bool)


def _iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def sam_quality_check(out_root, predict_fn, max_images=200, work_side=512, seed=0):
    """Uji kualitas mask: bandingkan tebakan dari kotak vs polygon buatan manusia (human_polygons/).

    Untuk tiap polygon manusia, kotaknya dijadikan prompt lalu IoU dihitung untuk 4 cara:
      kotak (persegi), elips, SAM mentah, SAM dirapikan (yang dipakai untuk training).
    """
    out_root = Path(out_root)
    files = sorted((out_root / "human_polygons").glob("*/*.txt"))
    random.Random(seed).shuffle(files)
    cfg = getattr(predict_fn, "config", {})
    rows = []
    for f in files[:max_images]:
        split = f.parent.name
        img = find_image(out_root / "images" / split, f.stem)
        if img is None:
            continue
        W, H = image_size(img)
        s = work_side / max(W, H)
        ws = (max(int(round(W * s)), 1), max(int(round(H * s)), 1))
        polys = [(cls, np.asarray(vals, dtype=float).reshape(-1, 2)) for cls, kind, vals in read_labels(f) if kind == "poly"]
        if not polys:
            continue
        boxes = []
        for _cls, p in polys:
            x1, y1 = p.min(0)
            x2, y2 = p.max(0)
            boxes.append([x1 * W, y1 * H, x2 * W, y2 * H])
        raw = predict_fn.raw_masks(img, boxes)
        for (cls, p), box, m in zip(polys, boxes, raw):
            gt = _raster(p, ws)
            x1, y1, x2, y2 = (np.array(box) / [W, H, W, H]).tolist()
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            row = {"file": f.name, "class_id": cls,
                   "kotak": _iou(gt, _raster(box_to_corners(cx, cy, bw, bh), ws)),
                   "elips": _iou(gt, _raster(ellipse_polygon(cx, cy, bw, bh), ws))}
            if m is None or not m.any():
                row["sam_mentah"] = 0.0
                clean_poly = None
            else:
                from PIL import Image

                row["sam_mentah"] = _iou(gt, np.asarray(Image.fromarray(m.astype(np.uint8)).resize(ws, Image.NEAREST), dtype=bool))
                mc = clean_mask(m, box, cfg.get("clip_to_box", True), cfg.get("min_frag_ratio", 0.1))
                fill = mc.sum() / max((box[2] - box[0]) * (box[3] - box[1]), 1)
                clean_poly = mask_to_polygon(mc) if fill >= cfg.get("min_fill", 0.08) else None
            if clean_poly is None or len(clean_poly) < 3:
                clean_poly = ellipse_polygon(cx, cy, bw, bh)
            row["sam_rapi"] = _iou(gt, _raster(clean_poly, ws))
            rows.append(row)
    return rows


def summarize_quality(rows, names, min_instances=3):
    """Ringkas hasil sam_quality_check: rata-rata IoU per cara, keseluruhan dan per kelas."""
    methods = ("kotak", "elips", "sam_mentah", "sam_rapi")
    overall = {m: round(float(np.mean([r[m] for r in rows])), 3) for m in methods} if rows else {}
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class_id"]].append(r)
    per_class = []
    for cid, rs in sorted(by_cls.items()):
        if len(rs) >= min_instances:
            per_class.append({"kelas": names.get(cid, str(cid)), "instance": len(rs),
                              **{m: round(float(np.mean([r[m] for r in rs])), 3) for m in methods}})
    return overall, per_class


# ---------------------------------------------------------------- review visual
def draw_labels(img_path, lbl_path, names, max_side=640, extra_boxes=None):
    """Gambar polygon/box di atas gambar (untuk review). Mengembalikan array RGB.

    extra_boxes: [(nama, (x1, y1, x2, y2) ternormalisasi)] digambar abu-abu, mis. objek kelas yang belum ada.
    """
    from PIL import Image, ImageDraw

    im = Image.open(img_path).convert("RGB")
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    draw = ImageDraw.Draw(im, "RGBA")
    rng = random.Random(0)
    palette = {cid: (rng.randrange(40, 255), rng.randrange(40, 255), rng.randrange(40, 255)) for cid in names}
    for cls, kind, vals in read_labels(lbl_path):
        pts = box_to_corners(*vals) if kind == "box" else np.asarray(vals).reshape(-1, 2)
        xy = [(float(x) * im.width, float(y) * im.height) for x, y in pts]
        color = palette.get(cls, (255, 0, 0))
        draw.polygon(xy, fill=color + (70,), outline=color + (255,))
        draw.text(xy[0], names.get(cls, str(cls)), fill=(255, 255, 255, 255))
    for name, (x1, y1, x2, y2) in extra_boxes or []:
        box = [x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height]
        draw.rectangle(box, outline=(160, 160, 160, 255), width=2)
        draw.text((box[0] + 2, box[1] + 2), f"{name} (belum ada)", fill=(255, 255, 255, 255))
    return np.asarray(im)


def sample_label_files(out_root, split="train", k=12, class_id=None, seed=0):
    files = sorted((Path(out_root) / "labels" / split).glob("*.txt"))
    if class_id is not None:
        files = [f for f in files if any(cls == class_id for cls, _k, _v in read_labels(f))]
    random.Random(seed).shuffle(files)
    return files[:k]


# ---------------------------------------------------------------- foto sendiri (uji dunia nyata)
REAL_SPLIT = "real"


def export_splits(src_root):
    """Daftar (nama_split, folder, jumlah_gambar) dari export YOLO.

    Gaya Roboflow: train/ valid/ test/ (masing-masing berisi images/ + labels/).
    Gaya datar: images/ + labels/ langsung di folder utama -> nama split 'datar'.
    """
    src_root = Path(src_root)

    def n_img(folder):
        return sum(1 for p in (folder / "images").iterdir() if p.suffix.lower() in IMG_EXTS)

    if (src_root / "images").is_dir():
        return [("datar", src_root, n_img(src_root))]
    return [(p.name, p, n_img(p)) for p in sorted(src_root.iterdir()) if p.is_dir() and (p / "images").is_dir()]


def build_real_testset(src_root, out_root, rows, splits=("test",), tag="FotoSendiri", remap=None):
    """Siapkan set uji foto sendiri dari export YOLO (Roboflow atau folder sendiri) di out_root/*/real.

    - Hanya split di `splits` yang dipakai (default: test). Export datar dipakai semua.
    - Nama kelas dicocokkan ke kelas_v1 (slug/nama). Objek kelas yang belum ada disimpan di unknown/real/
      sebagai 'nama x1 y1 x2 y2', supaya kelihatan kalau model menebak sesuatu di objek itu.
    - Foto yang semua objeknya kelas baru tetap dipakai (foto nyata tetap foto nyata).
    - Polygon buatanmu juga disalin ke human_polygons/real/ (kunci jawaban asli untuk kualitas mask).
    """
    src_root, out_root = Path(src_root), Path(out_root)
    names = read_source_names(src_root)
    idx_map = match_class_names(names, rows, tag, remap)
    dirs = {d: out_root / d / REAL_SPLIT for d in ("images", "labels", "unknown", "human_polygons")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    want = {SPLIT_ALIASES.get(s, s) for s in splits}
    available = export_splits(src_root)
    rep = {"split_tersedia": {s: n for s, _f, n in available}, "split_dipakai": [], "gambar": 0, "instance": 0,
           "polygon": 0, "box": 0, "objek_kelas_belum_ada": 0, "kelas_belum_ada": Counter(),
           "kelas_dikenal": sorted({names[i] for i, c in idx_map.items() if c is not None})}
    for split_name, folder, _n in available:
        if split_name != "datar" and SPLIT_ALIASES.get(split_name, split_name) not in want:
            continue
        rep["split_dipakai"].append(split_name)
        for img in sorted((folder / "images").iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            kept, polys, unknown = [], [], []
            for cls, kind, vals in read_labels(folder / "labels" / f"{img.stem}.txt"):
                cid = idx_map.get(cls)
                if cid is None:
                    nm = re.sub(r"\s+", "_", names[cls].strip()) if cls < len(names) else str(cls)
                    x1, y1, x2, y2 = label_bounds(kind, vals)
                    unknown.append(f"{nm} {x1:.5f} {y1:.5f} {x2:.5f} {y2:.5f}")
                    rep["kelas_belum_ada"][nm] += 1
                    continue
                if kind == "poly" and is_rectangle(vals):
                    x1, y1, x2, y2 = label_bounds(kind, vals)
                    kind, vals = "box", [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                kept.append(format_line(cid, vals))
                rep["polygon" if kind == "poly" else "box"] += 1
                if kind == "poly":
                    polys.append(format_line(cid, vals))
            stem = f"{tag}__{img.stem}"
            shutil.copy2(img, dirs["images"] / f"{stem}{img.suffix.lower()}")
            (dirs["labels"] / f"{stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            if unknown:
                (dirs["unknown"] / f"{stem}.txt").write_text("\n".join(unknown) + "\n", encoding="utf-8")
            if polys:
                (dirs["human_polygons"] / f"{stem}.txt").write_text("\n".join(polys) + "\n", encoding="utf-8")
            rep["gambar"] += 1
            rep["instance"] += len(kept)
            rep["objek_kelas_belum_ada"] += len(unknown)
    rep["kelas_belum_ada"] = dict(rep["kelas_belum_ada"].most_common())
    return rep


def write_real_yaml(out_root, names):
    """data.yaml untuk set uji foto sendiri (train/val/test menunjuk ke folder yang sama)."""
    out_root = Path(out_root)
    cfg = {"path": str(out_root), "train": f"images/{REAL_SPLIT}", "val": f"images/{REAL_SPLIT}",
           "test": f"images/{REAL_SPLIT}", "names": {int(k): v for k, v in names.items()}}
    path = out_root / "real.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return path


def read_unknown(out_root, stem):
    """Objek kelas belum ada untuk satu foto -> [(nama, (x1, y1, x2, y2))]."""
    f = Path(out_root) / "unknown" / REAL_SPLIT / f"{stem}.txt"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5:
            out.append((p[0], tuple(float(v) for v in p[1:])))
    return out


def _iou_matrix(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def item_level_report(model, out_root, names, conf=0.25, iou_thr=0.5, imgsz=640, device=None, batch=16):
    """Nilai per makanan, seperti yang dilihat user di app.

    Tiap objek berlabel dicocokkan ke tebakan model yang kotaknya overlap >= iou_thr:
      benar (kelas sama) / salah_kelas (ketemu tapi kelas lain) / kelewat (tidak ada tebakan).
    Tebakan yang tidak terpakai: di objek kelas belum ada, atau deteksi palsu.
    Mengembalikan (rows per objek berlabel, rows per tebakan ekstra).
    """
    out_root = Path(out_root)
    imgs = sorted(p for p in (out_root / "images" / REAL_SPLIT).iterdir() if p.suffix.lower() in IMG_EXTS)
    rows, extra = [], []
    for k in range(0, len(imgs), batch):
        chunk = imgs[k:k + batch]
        results = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=conf, verbose=False, device=device)
        for img, r in zip(chunk, results):
            gts = [(cls, label_bounds(kind, vals))
                   for cls, kind, vals in read_labels(out_root / "labels" / REAL_SPLIT / f"{img.stem}.txt")]
            unk = read_unknown(out_root, img.stem)
            if r.boxes is not None and len(r.boxes):
                pb = r.boxes.xyxyn.cpu().numpy()
                pc = r.boxes.cls.cpu().numpy().astype(int)
                ps = r.boxes.conf.cpu().numpy()
            else:
                pb, pc, ps = np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
            iou = _iou_matrix([b for _, b in gts], pb)
            used = set()
            order = sorted(range(len(gts)), key=lambda g: -(gts[g][1][2] - gts[g][1][0]) * (gts[g][1][3] - gts[g][1][1]))
            for g in order:
                gcls = gts[g][0]
                cand = [j for j in range(len(pb)) if j not in used and iou[g, j] >= iou_thr]
                same = [j for j in cand if pc[j] == gcls]
                pool = same or cand
                j = max(pool, key=lambda j: iou[g, j]) if pool else None
                hasil = "benar" if same else ("salah_kelas" if cand else "kelewat")
                if j is not None:
                    used.add(j)
                rows.append({"foto": img.name, "kelas": names.get(gcls, str(gcls)), "hasil": hasil,
                             "tebakan": names.get(int(pc[j]), str(pc[j])) if j is not None else "",
                             "conf": round(float(ps[j]), 3) if j is not None else None,
                             "iou": round(float(iou[g, j]), 3) if j is not None else None})
            uiou = _iou_matrix(pb, [b for _, b in unk])
            for j in range(len(pb)):
                if j in used:
                    continue
                on_unknown = len(unk) and uiou[j].max() >= iou_thr
                extra.append({"foto": img.name, "jenis": "objek_kelas_belum_ada" if on_unknown else "deteksi_palsu",
                              "objek": unk[int(uiou[j].argmax())][0] if on_unknown else "",
                              "tebakan": names.get(int(pc[j]), str(pc[j])), "conf": round(float(ps[j]), 3)})
    return rows, extra


def summarize_items(rows, extra):
    c = Counter(r["hasil"] for r in rows)
    n = len(rows)
    return {"objek": n, "benar": c.get("benar", 0), "salah_kelas": c.get("salah_kelas", 0),
            "kelewat": c.get("kelewat", 0), "persen_benar": round(c.get("benar", 0) / n, 3) if n else None,
            "deteksi_palsu": sum(e["jenis"] == "deteksi_palsu" for e in extra),
            "tebakan_di_kelas_belum_ada": sum(e["jenis"] == "objek_kelas_belum_ada" for e in extra)}


def per_class_items(rows):
    """Rekap per kelas: berapa objek, berapa yang benar / salah kelas / kelewat."""
    by = defaultdict(Counter)
    for r in rows:
        by[r["kelas"]][r["hasil"]] += 1
    out = []
    for kelas, c in by.items():
        n = sum(c.values())
        out.append({"kelas": kelas, "objek": n, "benar": c["benar"], "salah_kelas": c["salah_kelas"],
                    "kelewat": c["kelewat"], "persen_benar": round(c["benar"] / n, 3)})
    return sorted(out, key=lambda r: (r["persen_benar"], -r["objek"]))


def top_confusions(rows, extra, k=10):
    """Kesalahan paling sering: (sebenarnya, ditebak), termasuk tebakan di objek kelas yang belum ada."""
    pairs = Counter((r["kelas"], r["tebakan"]) for r in rows if r["hasil"] == "salah_kelas")
    pairs.update((f"{e['objek']} (belum ada)", e["tebakan"]) for e in extra if e["jenis"] == "objek_kelas_belum_ada")
    return [{"sebenarnya": a, "ditebak": b, "jumlah": n} for (a, b), n in pairs.most_common(k)]


# ---------------------------------------------------------------- dokumentasi
def best_epoch_from_results(results_csv):
    """Epoch terbaik dari results.csv dengan fitness yang sama seperti Ultralytics 8.4:
    mAP50-95(B) + mAP50-95(M)."""
    with open(results_csv, encoding="utf-8") as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows:
        return None
    cols = [c for c in ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)") if c in rows[0]]
    if not cols:
        return None
    best = max(rows, key=lambda r: sum(float(r[c]) for c in cols))
    keep = [c for c in best if c.startswith("metrics/") or c == "epoch"]
    return {k: (int(float(best[k])) if k == "epoch" else round(float(best[k]), 4)) for k in keep}


def best_epoch_info(run_dir):
    """Epoch yang benar-benar tersimpan sebagai best.pt (dicocokkan dari metrik di checkpoint).

    Lebih akurat daripada menghitung ulang dari results.csv karena rumus fitness bisa beda antar versi.
    """
    run_dir = Path(run_dir)
    try:
        import torch

        ck = torch.load(run_dir / "weights" / "best.pt", map_location="cpu", weights_only=False)
        tm, tr = ck.get("train_metrics") or {}, ck.get("train_results") or {}
        keys = [k for k in tm if k.startswith("metrics/") and k in tr]
        if keys and "epoch" in tr:
            dist = [sum(abs(float(tr[k][i]) - float(tm[k])) for k in keys) for i in range(len(tr["epoch"]))]
            i = int(np.argmin(dist))
            return {"epoch": int(tr["epoch"][i]), **{k: round(float(tm[k]), 4) for k in keys}, "sumber": "best.pt"}
    except Exception as exc:
        print("Info epoch dari best.pt tidak terbaca, pakai results.csv:", exc)
    res = best_epoch_from_results(run_dir / "results.csv")
    if res:
        res["sumber"] = "results.csv"
    return res


def write_model_card(path, info):
    """Tulis MODEL_CARD.md supaya dokumentasi training tidak hilang lagi."""
    L = [f"# Model card: {info.get('run_name', '-')}", "",
         f"- Dibuat: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
         f"- Ultralytics: {info.get('ultralytics_version', '-')}",
         f"- Model dasar: {info.get('base_model', '-')}, imgsz {info.get('imgsz', '-')}",
         f"- Daftar kelas: {info.get('class_list', 'kelas_v1.csv')} ({info.get('n_classes', '-')} kelas)",
         f"- Mask: {info.get('mask_method', '-')}", "", "## Sumber data", ""]
    for s in info.get("sources", []):
        L.append(f"- {s.get('tag')}: {s.get('workspace')}/{s.get('project')} v{s.get('version')} ({s.get('license', '-')})")
    L += ["", "## Jumlah data (foto unik)", "", "| split | gambar |", "|---|---|"]
    for k, v in (info.get("n_images") or {}).items():
        L.append(f"| {k} | {v} |")
    for title, key in (("Pembersihan data", "cleaning"), ("Statistik SAM", "sam_stats"),
                       ("Uji kualitas mask (IoU vs polygon manusia)", "sam_quality")):
        if info.get(key):
            L += ["", f"## {title}", "", "```", json.dumps(info[key], indent=2, ensure_ascii=False), "```"]
    if info.get("best_epoch"):
        L += ["", "## Epoch terbaik (validasi)", "", "```", json.dumps(info["best_epoch"], indent=2), "```"]
    if info.get("final_metrics"):
        L += ["", f"## Evaluasi akhir ({info.get('final_split', '-')})", "", "```",
              json.dumps(info["final_metrics"], indent=2), "```"]
    if info.get("per_source"):
        L += ["", "## Evaluasi per sumber", "", "| sumber | gambar | box mAP50 | box mAP50-95 | mask mAP50 | mask mAP50-95 |",
              "|---|---|---|---|---|---|"]
        for r in info["per_source"]:
            L.append(f"| {r['sumber']} | {r['gambar']} | {r['box_mAP50']} | {r['box_mAP50-95']} | "
                     f"{r['mask_mAP50']} | {r['mask_mAP50-95']} |")
    if info.get("notes"):
        L += ["", "## Catatan", "", info["notes"]]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")
    return Path(path)

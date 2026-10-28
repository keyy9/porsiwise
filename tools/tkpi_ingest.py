#!/usr/bin/env python3
"""
Bangun database gizi Porsiwise dari data TKPI (Tabel Komposisi Pangan Indonesia).

Input
  --tkpi   CSV TKPI. Mendukung format Kaggle "Indonesian Food and Drink Nutrition
           Dataset" (kolom: id, calories, proteins, fat, carbohydrate, name, image)
           maupun export lain selama nama kolomnya dikenali (lihat COLUMN_ALIASES).
  --kelas  data/kelas_v1.csv (daftar kelas deteksi + kata kunci TKPI).
  --mapping (opsional) CSV mapping yang sudah dikonfirmasi: class_id,food_id,porsi_fraksi

Output (di --out)
  gizi.sqlite             database untuk dibundel di app Flutter (offline) & seed backend
  foods_clean.csv         data pangan yang sudah dibersihkan
  mapping_suggestions.csv kandidat item TKPI untuk tiap kelas (top 3) untuk direview
  quality_report.md       ringkasan kualitas data

Hanya pakai standard library Python 3.9+.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

COLUMN_ALIASES = {
    "source_id": ["id", "kode", "code", "kode_bahan", "no"],
    "name": ["name", "nama", "nama_bahan", "nama_pangan", "food", "bahan"],
    "energy_kcal": ["calories", "calorie", "energi", "energy", "kalori", "energi_kal", "energi (kal)"],
    "protein_g": ["proteins", "protein", "protein (g)"],
    "fat_g": ["fat", "lemak", "lemak (g)"],
    "carbs_g": ["carbohydrate", "carbohydrates", "karbohidrat", "karbo", "kh", "karbohidrat (g)"],
    "fiber_g": ["fiber", "fibre", "serat", "serat (g)"],
    "image_url": ["image", "image_url", "gambar"],
}

# Ejaan yang sering berbeda antara nama kelas dan nama TKPI.
SYNONYMS = {
    "mie": "mi",
    "telor": "telur",
    "bergedel": "perkedel",
    "empek empek": "pempek",
    "cap cay": "capcay",
    "timun": "ketimun",
}


def norm(text: str) -> str:
    """Huruf kecil, tanpa aksen/tanda baca, spasi tunggal, sinonim diseragamkan."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    text = re.sub(r"\s+", " ", text)
    for src, dst in SYNONYMS.items():
        text = re.sub(rf"\b{src}\b", dst, text)
    return text


def parse_num(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"", "-", "na", "n/a", "nan", "none", "null"}:
        return None
    if s in {"tr", "trace", "sedikit"}:
        return 0.0
    s = s.replace(" ", "")
    # "1.234,5" -> 1234.5 ; "1,5" -> 1.5
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _hnorm(h: str) -> str:
    """Seragamkan nama kolom: huruf kecil, spasi/underscore tunggal."""
    return re.sub(r"[\s_]+", " ", h.strip().lower())


def detect_columns(header: list[str]) -> dict[str, str]:
    lookup = {}
    for h in header:
        lookup.setdefault(_hnorm(h), h)
        # juga versi tanpa satuan, mis. "Energi (Kal)" -> "energi"
        lookup.setdefault(_hnorm(re.sub(r"\(.*?\)", "", h)), h)
    found = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if _hnorm(alias) in lookup:
                found[field] = lookup[_hnorm(alias)]
                break
    missing = [f for f in ("name", "energy_kcal", "protein_g", "fat_g", "carbs_g") if f not in found]
    if missing:
        raise SystemExit(f"Kolom wajib tidak ditemukan: {missing}. Header CSV: {header}")
    return found


def load_tkpi(path: Path) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    cols = detect_columns(reader.fieldnames or [])

    foods, seen = [], {}
    stats = {"rows_in": 0, "skipped_no_name": 0, "skipped_no_energy": 0, "duplicates": 0, "negative": 0}
    for row in reader:
        stats["rows_in"] += 1
        name = re.sub(r"\s+", " ", (row.get(cols["name"]) or "").strip())
        if not name:
            stats["skipped_no_name"] += 1
            continue
        rec = {
            "source_id": (row.get(cols["source_id"]) or "").strip() if "source_id" in cols else "",
            "name": name,
            "name_norm": norm(name),
            "energy_kcal": parse_num(row.get(cols["energy_kcal"])),
            "protein_g": parse_num(row.get(cols["protein_g"])),
            "fat_g": parse_num(row.get(cols["fat_g"])),
            "carbs_g": parse_num(row.get(cols["carbs_g"])),
            "fiber_g": parse_num(row.get(cols["fiber_g"])) if "fiber_g" in cols else None,
            "image_url": (row.get(cols["image_url"]) or "").strip() if "image_url" in cols else "",
        }
        if rec["energy_kcal"] is None:
            stats["skipped_no_energy"] += 1
            continue
        if any(v is not None and v < 0 for k, v in rec.items() if k.endswith(("_kcal", "_g"))):
            stats["negative"] += 1
            continue
        if rec["name_norm"] in seen:
            stats["duplicates"] += 1
            continue
        seen[rec["name_norm"]] = True
        rec["flags"] = quality_flags(rec)
        foods.append(rec)

    for i, rec in enumerate(foods, start=1):
        rec["food_id"] = i
    stats["rows_out"] = len(foods)
    stats["columns"] = cols
    return foods, stats


def quality_flags(rec: dict) -> str:
    flags = []
    p, f, c, e = rec["protein_g"], rec["fat_g"], rec["carbs_g"], rec["energy_kcal"]
    if None in (p, f, c):
        flags.append("makro_tidak_lengkap")
    else:
        if p + f + c > 100.5:
            flags.append("makro_lebih_100g")
        atwater = 4 * p + 4 * c + 9 * f
        if e and e > 20 and abs(e - atwater) / e > 0.25:
            flags.append(f"energi_vs_atwater_{atwater:.0f}")
    return ";".join(flags)


def load_classes(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["class_id"] = int(r["class_id"])
        r["porsi_default_g"] = parse_num(r.get("porsi_default_g"))
        r["keywords"] = [k.strip() for k in (r.get("tkpi_kata_kunci") or "").split("|") if k.strip()]
    return rows


def score(keyword: str, food_norm: str) -> float:
    k = norm(keyword)
    if not k:
        return 0.0
    if food_norm == k:
        return 1.0
    if food_norm.startswith(k + " ") or food_norm.endswith(" " + k):
        return 0.9
    k_tokens, f_tokens = set(k.split()), set(food_norm.split())
    if k_tokens <= f_tokens:
        # semua kata kunci ada; makin sedikit kata tambahan makin bagus
        return 0.75 + 0.1 * len(k_tokens) / len(f_tokens)
    return 0.6 * difflib.SequenceMatcher(None, k, food_norm).ratio()


def suggest_mapping(classes: list[dict], foods: list[dict], top_n: int = 3) -> list[dict]:
    out = []
    for cls in classes:
        best = {}
        for kw in cls["keywords"]:
            for food in foods:
                s = score(kw, food["name_norm"])
                if s > best.get(food["food_id"], (0, ""))[0]:
                    best[food["food_id"]] = (s, kw)
        ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
        for rank, (food_id, (s, kw)) in enumerate(ranked, start=1):
            food = foods[food_id - 1]
            out.append({
                "class_id": cls["class_id"], "slug": cls["slug"], "rank": rank,
                "food_id": food_id, "tkpi_name": food["name"], "energy_kcal_100g": food["energy_kcal"],
                "match_score": round(s, 3), "keyword": kw,
                "pilih": "1" if rank == 1 and s >= 0.75 else "",
            })
        if not ranked:
            out.append({"class_id": cls["class_id"], "slug": cls["slug"], "rank": "", "food_id": "",
                        "tkpi_name": "(tidak ada kandidat)", "energy_kcal_100g": "", "match_score": 0,
                        "keyword": "", "pilih": ""})
    return out


def load_confirmed(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    result = []
    for r in rows:
        if "pilih" in r and r["pilih"].strip() not in {"1", "y", "ya", "yes", "x"}:
            continue
        frac = parse_num(r.get("porsi_fraksi")) or 1.0
        result.append({"class_id": int(r["class_id"]), "food_id": int(r["food_id"]), "fraction": frac})
    return result


SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE foods (
    food_id      INTEGER PRIMARY KEY,
    source_id    TEXT,
    name         TEXT NOT NULL,
    name_norm    TEXT NOT NULL,
    energy_kcal  REAL NOT NULL,   -- per 100 g bagian yang dapat dimakan
    protein_g    REAL,
    fat_g        REAL,
    carbs_g      REAL,
    fiber_g      REAL,
    image_url    TEXT,
    flags        TEXT
);
CREATE INDEX idx_foods_name_norm ON foods(name_norm);
CREATE TABLE classes (
    class_id         INTEGER PRIMARY KEY,   -- = indeks kelas YOLO
    slug             TEXT NOT NULL UNIQUE,
    nama             TEXT NOT NULL,
    kelompok         TEXT,
    tipe_label       TEXT,
    prioritas        TEXT,
    porsi_default_g  REAL,
    porsi_urt        TEXT
);
CREATE TABLE class_food_map (
    class_id  INTEGER NOT NULL REFERENCES classes(class_id),
    food_id   INTEGER NOT NULL REFERENCES foods(food_id),
    fraction  REAL NOT NULL DEFAULT 1.0,    -- porsi komposit, total per kelas = 1
    status    TEXT NOT NULL,                -- 'confirmed' atau 'auto'
    PRIMARY KEY (class_id, food_id)
);
CREATE VIEW class_nutrition_100g AS
SELECT c.class_id, c.slug, c.nama, c.porsi_default_g,
       SUM(m.fraction * f.energy_kcal) AS energy_kcal,
       SUM(m.fraction * f.protein_g)   AS protein_g,
       SUM(m.fraction * f.fat_g)       AS fat_g,
       SUM(m.fraction * f.carbs_g)     AS carbs_g,
       MIN(m.status)                   AS status
FROM classes c JOIN class_food_map m USING (class_id) JOIN foods f USING (food_id)
GROUP BY c.class_id;
"""


def build_sqlite(out: Path, foods, classes, mapping, source_label: str, keep_images: bool) -> None:
    db_path = out / "gizi.sqlite"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO foods VALUES (:food_id,:source_id,:name,:name_norm,:energy_kcal,:protein_g,"
        ":fat_g,:carbs_g,:fiber_g,:image_url,:flags)",
        [dict(f, image_url=f["image_url"] if keep_images else None) for f in foods],
    )
    con.executemany(
        "INSERT INTO classes VALUES (?,?,?,?,?,?,?,?)",
        [(c["class_id"], c["slug"], c["nama"], c["kelompok"], c["tipe_label"], c["prioritas"],
          c["porsi_default_g"], c["porsi_urt"]) for c in classes],
    )
    con.executemany("INSERT INTO class_food_map VALUES (?,?,?,?)",
                    [(m["class_id"], m["food_id"], m["fraction"], m["status"]) for m in mapping])
    meta = {
        "schema_version": "1",
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": source_label,
        "unit": "per 100 g bagian yang dapat dimakan",
        "foods": str(len(foods)),
        "classes": str(len(classes)),
    }
    con.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
    con.commit()
    con.close()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_report(path: Path, stats: dict, foods: list[dict], classes: list[dict], mapping: list[dict],
                 suggestions: list[dict]) -> None:
    flagged = [f for f in foods if f["flags"]]
    mapped = {m["class_id"] for m in mapping}
    unmapped = [c for c in classes if c["class_id"] not in mapped]
    weak = [s for s in suggestions if s["rank"] == 1 and float(s["match_score"] or 0) < 0.75]
    lines = [
        "# Laporan kualitas data gizi",
        "",
        f"- Baris masuk: {stats['rows_in']}",
        f"- Baris dipakai: {stats['rows_out']}",
        f"- Dibuang tanpa nama: {stats['skipped_no_name']}, tanpa energi: {stats['skipped_no_energy']}, "
        f"nilai negatif: {stats['negative']}, duplikat nama: {stats['duplicates']}",
        f"- Kolom terdeteksi: `{json.dumps(stats['columns'], ensure_ascii=False)}`",
        f"- Baris dengan catatan kualitas: {len(flagged)}",
        "",
        f"## Mapping kelas → TKPI",
        "",
        f"- Kelas total: {len(classes)}; punya mapping: {len(mapped)}; belum: {len(unmapped)}",
        f"- Kandidat teratas dengan skor < 0.75 (perlu dicek manual): {len(weak)}",
        "",
    ]
    if unmapped:
        lines.append("Kelas tanpa mapping: " + ", ".join(c["slug"] for c in unmapped))
        lines.append("")
    if flagged:
        lines += ["## Contoh baris dengan catatan", "", "| nama | energi | P | L | KH | catatan |", "|---|---|---|---|---|---|"]
        for f in flagged[:40]:
            lines.append(f"| {f['name']} | {f['energy_kcal']} | {f['protein_g']} | {f['fat_g']} | {f['carbs_g']} | {f['flags']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tkpi", required=True, type=Path)
    ap.add_argument("--kelas", required=True, type=Path)
    ap.add_argument("--mapping", type=Path, help="CSV mapping terkonfirmasi (class_id,food_id[,porsi_fraksi][,pilih])")
    ap.add_argument("--out", type=Path, default=Path("build"))
    ap.add_argument("--source-label", default="TKPI (Kemenkes RI) via Kaggle: Indonesian Food and Drink Nutrition Dataset")
    ap.add_argument("--keep-images", action="store_true", help="Simpan URL gambar di SQLite (default: tidak)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    foods, stats = load_tkpi(args.tkpi)
    classes = load_classes(args.kelas)
    suggestions = suggest_mapping(classes, foods)

    if args.mapping:
        mapping = [dict(m, status="confirmed") for m in load_confirmed(args.mapping)]
    else:
        mapping = [{"class_id": s["class_id"], "food_id": s["food_id"], "fraction": 1.0, "status": "auto"}
                   for s in suggestions if s["pilih"] == "1"]

    build_sqlite(args.out, foods, classes, mapping, args.source_label, args.keep_images)
    write_csv(args.out / "foods_clean.csv", foods,
              ["food_id", "source_id", "name", "energy_kcal", "protein_g", "fat_g", "carbs_g", "fiber_g", "flags"])
    write_csv(args.out / "mapping_suggestions.csv", suggestions,
              ["class_id", "slug", "rank", "food_id", "tkpi_name", "energy_kcal_100g", "match_score", "keyword", "pilih"])
    write_report(args.out / "quality_report.md", stats, foods, classes, mapping, suggestions)

    print(f"OK: {stats['rows_out']} pangan, {len(classes)} kelas, {len(mapping)} mapping -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

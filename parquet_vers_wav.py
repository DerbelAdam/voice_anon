#!/usr/bin/env python3
"""Convertit des fichiers .parquet (format des datasets HuggingFace) en fichiers .wav
utilisables par l'application : un sous-dossier par locuteur, 16 kHz mono.

    python parquet_vers_wav.py <fichier.parquet | dossier> --info          # voir le contenu d'abord
    python parquet_vers_wav.py <fichier.parquet | dossier> --out DOSSIER   # convertir

Résultat :
    DOSSIER/<locuteur>/<nom>.wav     un sous-dossier par locuteur (si une colonne locuteur existe)
    DOSSIER/metadata.csv             une ligne par fichier, avec les autres colonnes (texte, label…)

Formats audio reconnus dans une colonne :
  - {"bytes": ..., "path": ...}            (Audio HuggingFace, non décodé : le cas le plus courant)
  - {"array": [...], "sampling_rate": ...} (Audio HuggingFace décodé)
  - octets bruts d'un fichier wav / flac / mp3 / ogg
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
SPEAKER_CANDIDATES = ["speaker_id", "speaker", "spk_id", "spk", "client_id", "speakerid", "speaker_name",
                      "participant_id", "participant", "subject_id", "subject", "user_id", "patient_id"]


def lister_parquets(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    files = sorted(src.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"Aucun fichier .parquet dans {src}")
    return files


def est_audio(t: pa.DataType) -> bool:
    if pa.types.is_struct(t):
        noms = {t.field(i).name for i in range(t.num_fields)}
        return "bytes" in noms or "array" in noms
    return pa.types.is_binary(t) or pa.types.is_large_binary(t)


def detecter_colonnes(schema: pa.Schema, audio_col: str | None, speaker_col: str | None):
    if audio_col is None:
        audios = [f.name for f in schema if est_audio(f.type)]
        if not audios:
            raise SystemExit("Aucune colonne audio reconnue : précisez-la avec --audio-col (voir --info).")
        audio_col = audios[0]
    if speaker_col is None:
        noms = {f.name.lower(): f.name for f in schema}
        speaker_col = next((noms[c] for c in SPEAKER_CANDIDATES if c in noms), None)
    elif speaker_col.lower() == "aucune":
        speaker_col = None
    return audio_col, speaker_col


def decoder(valeur) -> tuple[np.ndarray, str | None]:
    """Valeur d'une cellule audio -> (signal 16 kHz mono float32, nom de fichier d'origine éventuel)."""
    nom = None
    if isinstance(valeur, dict):
        nom = valeur.get("path")
        if valeur.get("bytes"):
            data = valeur["bytes"]
        elif valeur.get("array") is not None:
            y = np.asarray(valeur["array"], dtype=np.float32)
            sr = int(valeur.get("sampling_rate") or SR)
            if y.ndim > 1:
                y = y.mean(axis=0 if y.shape[0] < y.shape[-1] else -1)
            return (librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y), nom
        else:
            raise ValueError("cellule audio vide")
    else:
        data = valeur
    try:  # wav, flac, ogg (et mp3 avec une libsndfile récente)
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:  # noqa: BLE001  — autres formats : décodage par librosa
        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32), nom


def propre(s: str) -> str:
    """Nom utilisable comme nom de fichier / dossier sous Windows."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(s)).strip(" .")
    return s[:80] or "sans_nom"


def info(files: list[Path], audio_col, speaker_col) -> None:
    pf = pq.ParquetFile(files[0])
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"{len(files)} fichier(s) .parquet, {total} lignes au total\n")
    print("Colonnes :")
    for f in pf.schema_arrow:
        tag = "  <- audio" if est_audio(f.type) else ""
        print(f"   {f.name:25s} {str(f.type)[:60]}{tag}")
    a, s = detecter_colonnes(pf.schema_arrow, audio_col, speaker_col)
    print(f"\nColonne audio détectée    : {a}")
    print(f"Colonne locuteur détectée : {s or 'aucune (précisez --speaker-col si elle existe)'}")
    autres = [f.name for f in pf.schema_arrow if f.name != a]
    lot = next(pf.iter_batches(batch_size=3, columns=autres)) if autres else None
    if lot is not None:
        print("\nExemple (3 premières lignes, colonnes non audio) :")
        for row in lot.to_pylist():
            print("   ", {k: (str(v)[:60] if v is not None else None) for k, v in row.items()})
    lot = next(pf.iter_batches(batch_size=1, columns=[a]))
    y, _ = decoder(lot.column(0)[0].as_py())
    print(f"\nPremier audio décodé : {len(y) / SR:.1f} s à 16 kHz")


def convertir(files: list[Path], out: Path, audio_col, speaker_col, max_total, per_speaker, min_s) -> None:
    out.mkdir(parents=True, exist_ok=True)
    a, s = detecter_colonnes(pq.ParquetFile(files[0]).schema_arrow, audio_col, speaker_col)
    if s is None:
        print("Pas de colonne locuteur : tous les fichiers vont dans « sans_locuteur ». "
              "Pour l'application, traitez-les alors un par un (pas le dossier entier).")
    meta_cols = [f.name for f in pq.ParquetFile(files[0]).schema_arrow
                 if f.name != a and not est_audio(f.type)]
    n_ok = n_err = n_court = 0
    par_loc: dict[str, int] = {}
    with open(out / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fichier", "locuteur", "duree_s", *meta_cols])
        for pfile in files:
            pf = pq.ParquetFile(pfile)
            ligne = 0
            for lot in pf.iter_batches(batch_size=32):  # par paquets : un .parquet peut peser des Go
                for row in lot.to_pylist():
                    ligne += 1
                    if max_total and n_ok >= max_total:
                        print(f"\n--max {max_total} atteint.")
                        break
                    loc = propre(row[s]) if s and row.get(s) is not None else "sans_locuteur"
                    if per_speaker and par_loc.get(loc, 0) >= per_speaker:
                        continue
                    try:
                        y, nom = decoder(row[a])
                    except Exception as e:  # noqa: BLE001
                        n_err += 1
                        print(f"   ligne {ligne} de {pfile.name} ignorée : {type(e).__name__}")
                        continue
                    if len(y) < min_s * SR:
                        n_court += 1
                        continue
                    base = propre(Path(nom).stem) if nom else f"{propre(pfile.stem)}_{ligne:06d}"
                    dst = out / loc / f"{base}.wav"
                    if dst.exists():
                        dst = out / loc / f"{base}_{propre(pfile.stem)}_{ligne:06d}.wav"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(dst, y, SR, subtype="PCM_16")
                    w.writerow([dst.relative_to(out).as_posix(), loc, round(len(y) / SR, 2),
                                *[row.get(c) for c in meta_cols]])
                    n_ok += 1
                    par_loc[loc] = par_loc.get(loc, 0) + 1
                    if n_ok % 50 == 0:
                        print(f"   {n_ok} fichiers écrits...")
                else:
                    continue
                break
            if max_total and n_ok >= max_total:
                break
    print(f"\nTerminé : {n_ok} fichiers .wav, {len(par_loc)} locuteur(s) -> {out}")
    if n_court:
        print(f"   {n_court} ignoré(s) car plus courts que {min_s} s")
    if n_err:
        print(f"   {n_err} ignoré(s) car illisibles")
    print(f"   Les autres colonnes sont dans {out / 'metadata.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="un fichier .parquet, ou un dossier qui en contient")
    ap.add_argument("--info", action="store_true", help="afficher les colonnes et un exemple, sans convertir")
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut : ../dataset_<nom>)")
    ap.add_argument("--audio-col", default=None, help="nom de la colonne audio (détectée sinon)")
    ap.add_argument("--speaker-col", default=None, help="nom de la colonne locuteur (détectée sinon ; 'aucune')")
    ap.add_argument("--max", type=int, default=0, help="nombre maximal de fichiers à écrire (0 = tous)")
    ap.add_argument("--per-speaker", type=int, default=0, help="nombre maximal de fichiers par locuteur")
    ap.add_argument("--min-seconds", type=float, default=1.0, help="ignorer les audios plus courts")
    a = ap.parse_args()

    src = Path(a.source)
    if not src.exists():
        raise SystemExit(f"Introuvable : {src}")
    files = lister_parquets(src)
    if a.info:
        info(files, a.audio_col, a.speaker_col)
        return
    root = Path(__file__).resolve().parent
    out = Path(a.out) if a.out else root.parent / f"dataset_{propre(src.stem)}"
    convertir(files, out, a.audio_col, a.speaker_col, a.max, a.per_speaker, a.min_seconds)


if __name__ == "__main__":
    sys.exit(main())
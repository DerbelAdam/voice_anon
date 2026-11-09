#!/usr/bin/env python3
"""Anonymise la voix d'un patient et produit un rapport explicatif par fichier.

    python anonymiser.py <voix> [--out-dir DOSSIER] [--whisper base]

<voix> : un fichier audio, ou un dossier contenant les enregistrements d'UN SEUL patient
         (même pseudo-locuteur et même décalage de F0 pour tous ses fichiers ; ses autres
         enregistrements servent aussi au test de ré-identification).

Résultat (par défaut dans results/<nom>) :
    audio/<fichier>_anon.wav        la voix anonymisée
    audio/<fichier>_anon.json       ce qui a été fait (pseudo-locuteur, décalage de F0…)
    rapports/<fichier>_rapport.html le rapport à ouvrir dans le navigateur
    index.html                      la synthèse de tous les fichiers

Au premier lancement, le pool de voix (LibriSpeech) est pré-calculé automatiquement.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # avertissements internes de PyTorch / kNN-VC sans conséquence

import numpy as np
import torch

from anon.audio import load_audio, save_audio
from anon.compare import compare, load_whisper
from anon.convert import Anonymizer
from anon.knnvc import load_knnvc
from anon.prosody import f0_contour
from anon.pseudo_speaker import AUDIO_EXT, build_pool, pool_is_complete
from anon.report import write_index, write_report
from anon.speaker import SpeakerEncoder, l2norm

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voix", help="fichier audio, ou dossier d'UN patient")
    ap.add_argument("--out-dir", default=None, help="dossier de sortie (défaut : results/<nom>)")
    ap.add_argument("--patient-id", default=None, help="identifiant du patient (défaut : nom du fichier/dossier)")
    ap.add_argument("--whisper", default=None, help="modèle Whisper pour l'intelligibilité : tiny, base, small…")
    ap.add_argument("--cache", default=str(ROOT / "cache" / "pool"), help="pool pré-calculé")
    ap.add_argument("--pool-dir", default=str(ROOT.parent / "librispeech" / "LibriSpeech" / "dev-clean"),
                    help="voix du pool (un sous-dossier par locuteur)")
    ap.add_argument("--f0-mode", choices=["transfer", "knn"], default="transfer",
                    help="transfer = intonation du patient réimposée (R2) ; knn = sans R2")
    ap.add_argument("--no-rules", action="store_true", help="désactive R3 et R4 (pour comparer)")
    ap.add_argument("--gender", choices=["same", "any"], default="same")
    ap.add_argument("--n-farthest", type=int, default=10)
    ap.add_argument("--n-select", type=int, default=5)
    ap.add_argument("--topk", type=int, default=4)
    a = ap.parse_args()

    src = Path(a.voix)
    if not src.exists():
        raise SystemExit(f"Introuvable : {src}")
    files = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"Aucun fichier audio dans {src}")
    pid = a.patient_id or (src.stem if src.is_file() else src.name)
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "results" / pid
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    # 1) modèles et pool
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement des modèles (sur {device})...")
    enc = SpeakerEncoder(device=device)
    kv = load_knnvc(device)
    if not pool_is_complete(a.pool_dir, a.cache):
        print(f"Pré-calcul du pool de voix depuis {a.pool_dir} (une seule fois)...")
        build_pool(a.pool_dir, a.cache, enc, kv)
    an = Anonymizer(a.cache, device=device, encoder=enc, knnvc=kv)
    wm = load_whisper(a.whisper)

    # 2) le patient : empreinte et F0 moyenne sur TOUS ses fichiers
    wavs = [load_audio(str(f)) for f in files]
    embs = [enc.embed(w) for w in wavs]
    emb = l2norm(np.mean(embs, axis=0))
    f0_patient = float(np.nanmean(np.concatenate([f0_contour(w)[1] for w in wavs])))
    print(f"Patient « {pid} » : {len(files)} fichier(s), F0 moyenne {f0_patient:.1f} Hz\n")

    # 3) anonymisation + rapport, fichier par fichier
    rows = []
    for i, (f, w) in enumerate(zip(files, wavs)):
        out, info, ps = an.anonymize(w, patient_id=pid, patient_emb=emb, patient_f0=f0_patient,
                                     f0_mode=a.f0_mode, topk=a.topk, n_farthest=a.n_farthest,
                                     n_select=a.n_select, gender=a.gender, apply_rules=not a.no_rules)
        dst = out_dir / "audio" / f"{f.stem}_anon.wav"
        save_audio(str(dst), out)
        info["input"], info["output"] = str(f), str(dst)
        dst.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")

        autres = [x for j, x in enumerate(wavs) if j != i]  # autres enregistrements du même patient
        res = compare(w, out, enc, pseudo_emb=ps.embedding,
                      f0_shift=info["f0_shift_hz"] if a.f0_mode == "transfer" else None,
                      same_speaker_wavs=autres or None, whisper_model=wm)
        page = write_report(res, info, str(out_dir / "rapports"), f.stem, w, out, str(f), str(dst))
        ok = sum(c["ok"] for c in res["checks"])
        print(f"{f.name} -> {dst.name} | pseudo-locuteur {info['pseudo_speakers']} | "
              f"ΔF0 {info['f0_shift_hz']:+.1f} Hz | {ok}/{len(res['checks'])} vérifications OK")
        for c in res["checks"]:
            if not c["ok"]:
                print(f"      ✗ {c['check']} : {c['value']} ({c['rule']})")
        rows.append({"file": f.name, "cos_in_out": res["identity"]["cos_in_out"],
                     "cos_out_same": res["identity"].get("cos_out_same_speaker_mean"),
                     "ok": ok, "n": len(res["checks"]), "link": f"rapports/{page.name}"})

    index = write_index(rows, str(out_dir), f"Anonymisation — {pid}")
    print(f"\nTerminé. Ouvrez la synthèse : {index}")


if __name__ == "__main__":
    main()
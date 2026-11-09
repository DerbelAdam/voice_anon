#!/usr/bin/env python3
"""Compare une voix originale, sa version anonymisée et d'autres extraits de la même personne.

    python comparer.py <originale> <anonymisée> [<autre extrait> ...] [--whisper base]

- <originale> et <anonymisée> : le MÊME enregistrement avant / après anonymisation
  (signaux alignés dans le temps : on vérifie durée, pauses, intonation, volume, silences, texte).
- <autre extrait> (un ou plusieurs, recommandé) : d'AUTRES enregistrements de la même personne,
  non anonymisés. Ils servent au test de ré-identification : la voix anonymisée
  ressemble-t-elle encore à cette personne ?

Si le fichier .json écrit par anonymiser.py se trouve à côté de la voix anonymisée, il est lu
automatiquement (pseudo-locuteur visé, décalage de F0 visé).
Résultat : <nom>_rapport.html (à ouvrir dans le navigateur), <nom>_rapport.json, <nom>_courbes.png
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

from anon.audio import load_audio
from anon.compare import compare, load_whisper
from anon.report import write_report
from anon.speaker import SpeakerEncoder

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("originale", help="voix originale")
    ap.add_argument("anonymisee", help="la même voix après anonymisation")
    ap.add_argument("autres", nargs="*", help="autres extraits NON anonymisés de la même personne")
    ap.add_argument("--whisper", default=None, help="modèle Whisper pour l'intelligibilité : tiny, base, small…")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "comparaisons"))
    a = ap.parse_args()

    for p in [a.originale, a.anonymisee, *a.autres]:
        if not Path(p).is_file():
            raise SystemExit(f"Introuvable : {p}")
    if not a.autres:
        print("Remarque : sans autre extrait de la même personne, le test de ré-identification est omis.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SpeakerEncoder(device=device)
    wm = load_whisper(a.whisper)
    w_in, w_out = load_audio(a.originale), load_audio(a.anonymisee)
    autres = [load_audio(p) for p in a.autres if Path(p).resolve() != Path(a.originale).resolve()]

    js = Path(a.anonymisee).with_suffix(".json")
    info = json.loads(js.read_text(encoding="utf-8")) if js.exists() else None
    if info is None:
        print("Remarque : pas de .json à côté de la voix anonymisée (pseudo-locuteur et décalage visé inconnus).")

    res = compare(w_in, w_out, enc,
                  pseudo_emb=np.array(info["pseudo_embedding"]) if info and "pseudo_embedding" in info else None,
                  f0_shift=info["f0_shift_hz"] if info and info.get("f0_mode") == "transfer" else None,
                  same_speaker_wavs=autres or None, whisper_model=wm)
    name = Path(a.originale).stem
    page = write_report(res, info, a.out_dir, name, w_in, w_out, a.originale, a.anonymisee)

    idt = res["identity"]
    print(f"\nIdentité : cos(originale, anonymisée) = {idt['cos_in_out']:.3f}")
    if "cos_out_same_speaker_mean" in idt:
        print(f"Ré-identification : cos(anonymisée, autres extraits) = {idt['cos_out_same_speaker_mean']:.3f} "
              f"(référence même personne : {idt['cos_in_same_speaker_mean']:.3f})")
    ok = sum(c["ok"] for c in res["checks"])
    print(f"\n{ok}/{len(res['checks'])} vérifications OK")
    for c in res["checks"]:
        print(f"   {'✓' if c['ok'] else '✗'} {c['check']:42s} {str(c['value']):28s} ({c['rule']})")
    print(f"\nRapport : {page}")


if __name__ == "__main__":
    main()
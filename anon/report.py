"""Étape 7 — Rapport HTML autonome et explicatif (un fichier à ouvrir dans le navigateur).

Le rapport contient : le verdict global, les deux audios à écouter, les paramètres de
l'anonymisation, l'identité (ECAPA), les vérifications des règles R1–R5, la figure
(intégrée dans le fichier) avec l'explication de chaque courbe, et les mesures Praat.
Chaque métrique est accompagnée d'une phrase qui dit ce qu'elle mesure et ce qu'on attend.
"""
from __future__ import annotations

import base64
import html
import json
import os
from datetime import datetime
from pathlib import Path

from .compare import ROWS, THRESHOLDS, plot

# ---------------------------------------------------------------- textes explicatifs

EXPLAIN_CHECK = {
    "Identité changée": "Similarité entre l'empreinte vocale de l'entrée et celle de la sortie. "
                        "Deux enregistrements d'une même personne donnent en général 0,6 à 0,9 ; "
                        "deux personnes différentes, moins de 0,3.",
    "Non reliable aux autres enregistrements": "Test de ré-identification : la sortie ressemble-t-elle "
                        "encore aux AUTRES enregistrements du patient ? C'est ce qu'un attaquant "
                        "possédant une autre voix du patient pourrait tenter.",
    "R1 Même durée": "La conversion se fait trame par trame (20 ms) : la sortie doit avoir exactement "
                     "la même durée. Le débit de parole est donc conservé.",
    "R1 Pauses retrouvées": "Part des pauses de l'entrée que l'on retrouve au même endroit dans la sortie "
                            "(recouvrement d'au moins 50 %). Les pauses sont un marqueur clé d'Alzheimer.",
    "R1 Durée totale des pauses": "Écart entre la durée cumulée des pauses avant et après.",
    "R2 Forme du contour F0": "Corrélation entre les courbes de hauteur de voix (intonation) de l'entrée et "
                              "de la sortie. Proche de 1 : l'intonation du patient est conservée.",
    "R2 Pitch range (P95−P5)": "Étendue de la hauteur de voix (entre les percentiles 5 et 95, robuste aux "
                               "erreurs de mesure). Seule la moyenne doit changer, pas l'étendue.",
    "R2 Décalage de F0 obtenu": "Décalage de hauteur mesuré trame par trame, comparé au décalage visé "
                                "(F0 moyenne du pseudo-locuteur − F0 moyenne du patient).",
    "R2 Voisement conservé": "Part des trames où l'entrée et la sortie sont d'accord sur « voix présente / "
                             "absente » (voyelles contre consonnes sourdes et silences).",
    "R3 Énergie RMS (parole)": "Écart de volume moyen sur les zones de parole. La sortie est ramenée au "
                               "niveau de l'original.",
    "R4 Aucun bruit ajouté dans les silences": "Écart d'énergie dans les silences de l'original. Positif = "
                                               "du bruit a été ajouté, ce qui masquerait les pauses.",
    "R5 Intelligibilité (WER sortie vs entrée)": "Taux de mots différents entre la transcription automatique "
                                                 "(Whisper) de l'entrée et celle de la sortie. Bas = mêmes mots.",
}

EXPLAIN_IDENTITY = {
    "cos_in_out": ("entrée ↔ sortie", "doit être bas : la voix a changé"),
    "cos_out_pseudo": ("sortie ↔ pseudo-locuteur visé", "doit être élevé : la nouvelle voix a bien été prise"),
    "cos_in_pseudo": ("entrée ↔ pseudo-locuteur visé", "bas par construction (voix choisies les plus éloignées)"),
    "cos_in_same_speaker_mean": ("entrée ↔ autres enregistrements du patient",
                                 "référence « même personne », avant anonymisation"),
    "cos_out_same_speaker_mean": ("sortie ↔ autres enregistrements du patient",
                                  "doit être bas : le patient n'est plus reconnaissable"),
}

EXPLAIN_ROW = {
    "duration_s": "durée totale de l'enregistrement",
    "f0_mean_hz": "hauteur moyenne de la voix — c'est la seule valeur prosodique censée changer",
    "f0_std_hz": "variabilité de la hauteur (sensible aux erreurs d'octave de Praat)",
    "f0_range_hz": "étendue robuste de la hauteur (P95 − P5)",
    "voiced_ratio": "part du temps où la voix vibre (voyelles, consonnes voisées)",
    "rms_speech_db": "volume moyen pendant la parole",
    "rms_silence_db": "niveau du bruit de fond pendant les silences",
    "n_pauses": "nombre de silences internes de plus de 150 ms",
    "pause_total_s": "durée cumulée de ces pauses",
    "speech_ratio": "part du temps passée à parler",
    "jitter_local_pct": "irrégularité de durée d'une vibration à l'autre (informatif)",
    "shimmer_local_pct": "irrégularité d'amplitude d'une vibration à l'autre (informatif)",
    "hnr_db": "rapport harmoniques / bruit : bas = voix soufflée ou rauque (informatif)",
}

EXPLAIN_FIGURE = [
    ("Forme d'onde", "le signal sonore de l'entrée (bleu) et de la sortie (rose) superposés. Les zones "
                     "jaunes sont les pauses détectées dans l'entrée : la sortie doit y rester silencieuse."),
    ("Spectrogrammes", "l'énergie de chaque fréquence au cours du temps (clair = fort). Les bandes "
                       "horizontales sont les harmoniques de la voix. Le timbre change (répartition de "
                       "l'énergie), mais les syllabes et les silences restent aux mêmes endroits."),
    ("Contour de F0", "la hauteur de la voix toutes les 10 ms. Bleu = patient, gris = cible (patient décalé "
                      "de Δ Hz), rose = sortie. Le rose doit suivre le gris : même intonation, autre hauteur "
                      "moyenne. Des points bleus isolés très hauts sont des erreurs de mesure de Praat."),
    ("Énergie RMS", "le volume toutes les 20 ms. Les deux courbes doivent avoir la même forme et le même "
                    "niveau, et la courbe rose ne doit pas remonter dans les zones jaunes."),
]

CSS = """
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1050px;margin:2em auto;padding:0 1em;
     color:#222;line-height:1.5}
h1{font-size:1.6em;margin-bottom:.2em} h2{font-size:1.25em;margin-top:1.8em;border-bottom:1px solid #ddd;
     padding-bottom:.2em} .muted{color:#666;font-size:.92em}
table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:.94em}
th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top} th{background:#f5f5f5}
td.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
.ok{color:#1b7f3b;font-weight:600} .ko{color:#b3261e;font-weight:600}
.verdict{padding:.8em 1em;border-radius:8px;margin:1em 0;font-size:1.05em}
.v-ok{background:#e8f5ec;border:1px solid #9fd3ae} .v-ko{background:#fdecea;border:1px solid #f2b8b5}
.audio{display:flex;gap:2em;flex-wrap:wrap} .audio div{flex:1;min-width:280px}
img{max-width:100%;border:1px solid #eee} code{background:#f3f3f3;padding:0 .3em;border-radius:3px}
"""


def _src(path: str, report_dir: Path) -> str:
    """Chemin d'un audio relatif au rapport (fonctionne si on déplace le dossier entier)."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), report_dir.resolve())).as_posix()
    except ValueError:  # autre lecteur sous Windows : chemin absolu
        return Path(path).resolve().as_uri()


def _fmt(v, d: int) -> str:
    try:
        return f"{v:.{d}f}"
    except (TypeError, ValueError):
        return "—"


def build_html(res: dict, info: dict | None, name: str, png_path: Path, report_dir: Path,
               input_path: str | None, output_path: str | None) -> str:
    e = html.escape
    checks = res["checks"]
    n_ok = sum(c["ok"] for c in checks)
    all_ok = n_ok == len(checks)
    L = [f"<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'><title>Rapport — {e(name)}</title>"
         f"<style>{CSS}</style></head><body>",
         f"<h1>Rapport d'anonymisation — {e(name)}</h1>",
         f"<div class='muted'>Généré le {datetime.now():%d/%m/%Y à %H:%M}</div>",
         f"<div class='verdict {'v-ok' if all_ok else 'v-ko'}'><b>{n_ok} / {len(checks)} vérifications "
         f"réussies.</b> " + ("Identité changée et marqueurs prosodiques préservés selon les critères ci-dessous."
                               if all_ok else "Voir les lignes marquées ✗ ci-dessous.") + "</div>"]

    # audios
    if input_path or output_path:
        L.append("<h2>Écouter</h2><div class='audio'>")
        for lab, p in [("Voix originale", input_path), ("Voix anonymisée", output_path)]:
            if p:
                L.append(f"<div><b>{lab}</b><br><audio controls preload='none' src='{e(_src(p, report_dir))}'>"
                         f"</audio><div class='muted'>{e(Path(p).name)}</div></div>")
        L.append("</div><p class='muted'>À écouter : la voix doit être différente, mais les mots, le rythme, "
                 "les pauses et l'intonation identiques.</p>")

    # paramètres
    if info:
        L += ["<h2>Paramètres de l'anonymisation</h2><table>",
              f"<tr><th>Pseudo-locuteur</th><td>moyenne des voix {e(', '.join(info['pseudo_speakers']))} "
              f"du pool, choisies parmi les plus éloignées du patient ({info['candidates_considered']} candidat(s) "
              f"du même genre)</td></tr>",
              f"<tr><th>F0 moyenne</th><td>patient {info['patient_f0_mean_hz']} Hz → pseudo-locuteur "
              f"{info['pseudo_f0_mean_hz']} Hz, soit un décalage Δ = {info['f0_shift_hz']:+} Hz</td></tr>",
              f"<tr><th>Règles</th><td>R2 (intonation du patient réimposée) : "
              f"{'oui' if info.get('f0_mode') == 'transfer' else 'non'} — R3/R4 (volume, silences) : "
              f"{'oui' if info.get('rules_applied', True) else 'non'}</td></tr>",
              f"<tr><th>Conversion</th><td>kNN-VC : chaque trame de 20 ms remplacée par la moyenne de ses "
              f"{info.get('topk', 4)} plus proches voisines parmi {info['matching_set_frames']} trames du "
              f"pseudo-locuteur ; {info.get('wavlm_segments', 1)} segment(s) ; {info['seconds']} s de calcul"
              f"</td></tr></table>"]

    # identité
    L += ["<h2>1. Identité (empreinte vocale ECAPA-TDNN)</h2>",
          "<p>Similarité cosinus entre empreintes vocales de 192 nombres : 1 = même voix, 0 = sans rapport.</p>",
          "<table><tr><th>Comparaison</th><th>Similarité</th><th>Ce qu'on attend</th></tr>"]
    for k, (lab, attendu) in EXPLAIN_IDENTITY.items():
        if k in res["identity"]:
            L.append(f"<tr><td>{lab}</td><td class='num'>{res['identity'][k]:.3f}</td><td>{attendu}</td></tr>")
    L.append("</table>")

    # vérifications
    L += ["<h2>2. Vérification des règles de protection</h2>",
          "<p>R1 débit et pauses · R2 intonation (seule la moyenne de F0 change) · R3 volume · "
          "R4 aucun bruit dans les silences · R5 intelligibilité.</p>",
          "<table><tr><th></th><th>Vérification</th><th>Résultat</th><th>Critère</th><th>Ce que ça mesure</th></tr>"]
    for c in checks:
        mark = "<span class='ok'>✓</span>" if c["ok"] else "<span class='ko'>✗</span>"
        L.append(f"<tr><td>{mark}</td><td>{e(c['check'])}</td><td class='num'>{e(str(c['value']))}</td>"
                 f"<td>{e(c['rule'])}</td><td>{e(EXPLAIN_CHECK.get(c['check'], ''))}</td></tr>")
    L.append("</table>")

    # figure
    b64 = base64.b64encode(png_path.read_bytes()).decode()
    L += ["<h2>3. Courbes</h2>", f"<img alt='Figure de comparaison' src='data:image/png;base64,{b64}'>",
          "<table><tr><th>Panneau</th><th>Comment le lire</th></tr>"]
    for t, d in EXPLAIN_FIGURE:
        L.append(f"<tr><td><b>{t}</b></td><td>{d}</td></tr>")
    L.append("</table>")

    # mesures Praat
    L += ["<h2>4. Mesures acoustiques (Praat)</h2>",
          "<table><tr><th>Mesure</th><th>Entrée</th><th>Sortie</th><th>Δ</th><th>Signification</th></tr>"]
    for k, lab, d in ROWS:
        a, b = res["summary_in"][k], res["summary_out"][k]
        L.append(f"<tr><td>{lab}</td><td class='num'>{_fmt(a, d)}</td><td class='num'>{_fmt(b, d)}</td>"
                 f"<td class='num'>{_fmt(b - a, d)}</td><td>{EXPLAIN_ROW.get(k, '')}</td></tr>")
    c = res["contour"]
    L.append(f"</table><p class='muted'>Contour F0 : corrélation {_fmt(c.get('f0_corr'), 3)}, écart de forme "
             f"{_fmt(c.get('f0_shape_rmse_hz'), 1)} Hz. Enveloppe d'énergie : corrélation "
             f"{_fmt(res['energy']['envelope_corr_speech'], 3)}. Jitter, shimmer et HNR sont informatifs : le "
             f"vocodeur régénère la forme d'onde et ne garantit pas ces micro-perturbations.</p>")

    # intelligibilité
    it = res["intelligibility"]
    L.append("<h2>5. Intelligibilité</h2>")
    if it:
        L += [f"<p>Transcription automatique Whisper (langue détectée : <code>{e(str(it['language']))}</code>).</p>",
              f"<table><tr><th>Entrée</th><td>{e(it['transcript_in'])}</td></tr>",
              f"<tr><th>Sortie</th><td>{e(it['transcript_out'])}</td></tr>",
              f"<tr><th>WER</th><td>{it['wer_out_vs_in']:.1%} de mots différents</td></tr></table>"]
    else:
        L.append("<p class='muted'>Non mesurée : relancez avec <code>--whisper base</code> "
                 "(nécessite <code>pip install openai-whisper</code>).</p>")

    L += ["<h2>Notes</h2><p class='muted'>Seuils indicatifs (première version) : "
          + ", ".join(f"{k} = {v}" for k, v in THRESHOLDS.items())
          + ". La comparaison prosodique se fait avec le même enregistrement (signaux alignés dans le temps) ; "
            "la ré-identification avec d'autres enregistrements du patient.</p></body></html>"]
    return "\n".join(L)


def write_report(res: dict, info: dict | None, report_dir: str, name: str, wav_in, wav_out,
                 input_path: str | None = None, output_path: str | None = None) -> Path:
    """Écrit <name>_rapport.html (autonome), <name>_rapport.json et <name>_courbes.png."""
    d = Path(report_dir)
    d.mkdir(parents=True, exist_ok=True)
    shift = info["f0_shift_hz"] if info and info.get("f0_mode") == "transfer" else None
    png = d / f"{name}_courbes.png"
    plot(wav_in, wav_out, png, shift, title=f"{name} — entrée vs sortie anonymisée")
    page = d / f"{name}_rapport.html"
    page.write_text(build_html(res, info, name, png, d, input_path, output_path), encoding="utf-8")
    info_light = {k: v for k, v in (info or {}).items() if k != "pseudo_embedding"} or None
    (d / f"{name}_rapport.json").write_text(json.dumps({"info": info_light, **res}, indent=1,
                                                       ensure_ascii=False, default=float), encoding="utf-8")
    return page


def write_index(rows: list[dict], out_dir: str, title: str) -> Path:
    """Page de synthèse quand un dossier (plusieurs fichiers) a été traité."""
    e = html.escape
    L = [f"<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'><title>{e(title)}</title>"
         f"<style>{CSS}</style></head><body><h1>{e(title)}</h1>",
         "<table><tr><th>Fichier</th><th>cos(entrée, sortie)</th><th>cos(sortie, autres enreg.)</th>"
         "<th>Vérifications</th><th>Rapport</th></tr>"]
    for r in rows:
        other = _fmt(r.get("cos_out_same"), 3)
        cls = "ok" if r["ok"] == r["n"] else "ko"
        L.append(f"<tr><td>{e(r['file'])}</td><td class='num'>{r['cos_in_out']:.3f}</td><td class='num'>{other}</td>"
                 f"<td class='{cls}'>{r['ok']} / {r['n']}</td><td><a href='{e(r['link'])}'>ouvrir</a></td></tr>")
    L.append("</table></body></html>")
    p = Path(out_dir) / "index.html"
    p.write_text("\n".join(L), encoding="utf-8")
    return p
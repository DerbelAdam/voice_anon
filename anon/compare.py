"""Étape 6 — Comparaison voix d'entrée / voix anonymisée (sans évaluation pathologique).

Deux questions :
  1. L'identité a-t-elle changé ?        -> similarité ECAPA (entrée, sortie, pseudo-locuteur)
  2. Les règles R1–R5 sont-elles tenues ? -> durée, pauses, contour de F0, énergie, silences,
     intelligibilité (WER si Whisper est installé) ; jitter/shimmer/HNR à titre informatif.

Les seuils (THRESHOLDS) sont indicatifs pour une première version : à recalibrer.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on écrit seulement des fichiers .png
import matplotlib.pyplot as plt
import numpy as np

from .audio import HOP, SR, detect_speech, frame_rms_db
from .convert import clean_f0
from .prosody import f0_contour, rms_db, summarize
from .speaker import SpeakerEncoder, cosine

THRESHOLDS = {
    "identity_cos_max": 0.30,     # cos(entrée, sortie) en dessous = « autre locuteur »
    "duration_tol_s": 0.01,       # R1
    "pause_match_min": 0.90,      # R1
    "f0_corr_min": 0.80,          # R2
    "f0_range_rel_tol": 0.15,     # R2
    "f0_shift_tol_hz": 10.0,      # R2
    "voicing_agreement_min": 0.85,  # R2
    "rms_tol_db": 1.0,            # R3
    "silence_margin_db": 1.0,     # R4
    "wer_max": 0.20,              # R5
}


# ---------------------------------------------------------------- outils

def _overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def pause_match_ratio(p_in, p_out) -> float:
    """Part des pauses de l'entrée retrouvées dans la sortie (recouvrement ≥ 50 %)."""
    if not p_in:
        return 1.0 if not p_out else 0.0
    hit = sum(any(_overlap(p, q) >= 0.5 * (p[1] - p[0]) for q in p_out) for p in p_in)
    return hit / len(p_in)


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate = (substitutions + suppressions + insertions) / nb de mots de ref."""
    norm = lambda s: "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in s).split()
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(h)] / max(1, len(r)))


def load_whisper(name: str | None):
    """Charge un modèle Whisper (tiny, base, small…) ; None si indisponible."""
    if not name:
        return None
    try:
        import whisper
        return whisper.load_model(name)
    except Exception as e:  # noqa: BLE001
        print(f"[compare] Whisper indisponible ({type(e).__name__}: {e}) : WER ignoré.")
        return None


# ---------------------------------------------------------------- comparaison

def compare(wav_in: np.ndarray, wav_out: np.ndarray, encoder: SpeakerEncoder,
            pseudo_emb: np.ndarray | None = None, f0_shift: float | None = None,
            same_speaker_wavs: list[np.ndarray] | None = None, whisper_model=None,
            thresholds: dict | None = None) -> dict:
    th = {**THRESHOLDS, **(thresholds or {})}
    seg_in, seg_out = detect_speech(wav_in), detect_speech(wav_out)
    s_in, s_out = summarize(wav_in, seg=seg_in), summarize(wav_out, seg=seg_out)

    # 1. identité
    e_in, e_out = encoder.embed(wav_in), encoder.embed(wav_out)
    ident = {"cos_in_out": cosine(e_in, e_out)}
    if pseudo_emb is not None:
        ident["cos_out_pseudo"] = cosine(e_out, pseudo_emb)
        ident["cos_in_pseudo"] = cosine(e_in, pseudo_emb)
    if same_speaker_wavs:
        autres = [encoder.embed(w) for w in same_speaker_wavs]
        ident["cos_in_same_speaker_mean"] = float(np.mean([cosine(e_in, e) for e in autres]))
        ident["cos_out_same_speaker_mean"] = float(np.mean([cosine(e_out, e) for e in autres]))

    # 2. contour de F0 (erreurs d'octave retirées des deux côtés)
    _, f_in = f0_contour(wav_in)
    _, f_out = f0_contour(wav_out)
    n = min(len(f_in), len(f_out))
    f_in, f_out = clean_f0(f_in[:n]), clean_f0(f_out[:n])
    both = np.isfinite(f_in) & np.isfinite(f_out)
    contour = {"voicing_agreement": float(np.mean(np.isfinite(f_in) == np.isfinite(f_out)))}
    if both.sum() > 3:
        a, b = f_in[both], f_out[both]
        contour["f0_corr"] = float(np.corrcoef(a, b)[0, 1])
        contour["f0_shape_rmse_hz"] = float(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))
        contour["f0_shift_measured_hz"] = float(np.median(b - a))  # décalage trame à trame
    else:
        contour["f0_shift_measured_hz"] = s_out["f0_mean_hz"] - s_in["f0_mean_hz"]

    # 3. énergie et silences (masque de parole de l'ENTRÉE pour les deux signaux)
    mask = seg_in.speech_mask(len(wav_in))
    r_in, r_out = frame_rms_db(wav_in), frame_rms_db(wav_out)
    m = min(len(r_in), len(r_out))
    sp = mask[np.minimum(np.arange(m) * HOP, len(wav_in) - 1)]
    energy = {
        "envelope_corr_speech": float(np.corrcoef(r_in[:m][sp], r_out[:m][sp])[0, 1]) if sp.sum() > 3 else float("nan"),
        "silence_rms_in_db": rms_db(wav_in[~mask]) if (~mask).any() else float("nan"),
        "silence_rms_out_db": rms_db(wav_out[:len(mask)][~mask]) if (~mask).any() else float("nan"),
    }

    # 4. pauses
    pauses = {"pause_match_ratio": pause_match_ratio(seg_in.pauses, seg_out.pauses),
              "pauses_in": seg_in.pauses, "pauses_out": seg_out.pauses}

    # 5. intelligibilité (optionnel) : la transcription de l'entrée sert de référence
    intel = {}
    if whisper_model is not None:
        r_in_ = whisper_model.transcribe(wav_in.astype(np.float32), fp16=False)
        lang = r_in_.get("language")
        r_out_ = whisper_model.transcribe(wav_out.astype(np.float32), language=lang, fp16=False)
        intel = {"language": lang, "transcript_in": r_in_["text"].strip(),
                 "transcript_out": r_out_["text"].strip()}
        intel["wer_out_vs_in"] = wer(intel["transcript_in"], intel["transcript_out"])

    # ---- vérifications ✅ / ❌
    checks = []

    def chk(name, value, rule, ok):
        checks.append({"check": name, "value": value, "rule": rule, "ok": bool(ok)})

    chk("Identité changée", round(ident["cos_in_out"], 3),
        f"cos(entrée, sortie) < {th['identity_cos_max']}", ident["cos_in_out"] < th["identity_cos_max"])
    if "cos_out_same_speaker_mean" in ident:
        chk("Non reliable aux autres enregistrements", round(ident["cos_out_same_speaker_mean"], 3),
            f"cos moyen < {th['identity_cos_max']}", ident["cos_out_same_speaker_mean"] < th["identity_cos_max"])
    dd = s_out["duration_s"] - s_in["duration_s"]
    chk("R1 Même durée", f"{dd:+.4f} s", f"abs(Δ) ≤ {th['duration_tol_s']} s", abs(dd) <= th["duration_tol_s"])
    chk("R1 Pauses retrouvées", f"{pauses['pause_match_ratio']:.0%} ({s_in['n_pauses']} → {s_out['n_pauses']})",
        f"≥ {th['pause_match_min']:.0%}", pauses["pause_match_ratio"] >= th["pause_match_min"])
    dp = s_out["pause_total_s"] - s_in["pause_total_s"]
    chk("R1 Durée totale des pauses", f"{dp:+.3f} s", "abs(Δ) ≤ 10 % ou ≤ 0,1 s",
        abs(dp) <= max(0.1, 0.1 * s_in["pause_total_s"]))
    if "f0_corr" in contour:
        chk("R2 Forme du contour F0", round(contour["f0_corr"], 3), f"corrélation ≥ {th['f0_corr_min']}",
            contour["f0_corr"] >= th["f0_corr_min"])
    rr = (s_out["f0_range_hz"] - s_in["f0_range_hz"]) / max(1e-6, s_in["f0_range_hz"])
    chk("R2 Pitch range (P95−P5)", f"{s_in['f0_range_hz']:.1f} → {s_out['f0_range_hz']:.1f} Hz ({rr:+.0%})",
        f"abs(Δ) ≤ {th['f0_range_rel_tol']:.0%}", abs(rr) <= th["f0_range_rel_tol"])
    if f0_shift is not None:
        err = contour["f0_shift_measured_hz"] - f0_shift
        chk("R2 Décalage de F0 obtenu", f"{contour['f0_shift_measured_hz']:+.1f} Hz (visé {f0_shift:+.1f})",
            f"erreur ≤ {th['f0_shift_tol_hz']} Hz", abs(err) <= th["f0_shift_tol_hz"])
    chk("R2 Voisement conservé", f"{contour['voicing_agreement']:.0%}", f"≥ {th['voicing_agreement_min']:.0%}",
        contour["voicing_agreement"] >= th["voicing_agreement_min"])
    de = s_out["rms_speech_db"] - s_in["rms_speech_db"]
    chk("R3 Énergie RMS (parole)", f"{de:+.2f} dB", f"abs(Δ) ≤ {th['rms_tol_db']} dB", abs(de) <= th["rms_tol_db"])
    ds = energy["silence_rms_out_db"] - energy["silence_rms_in_db"]
    if np.isfinite(ds):
        chk("R4 Aucun bruit ajouté dans les silences", f"{ds:+.2f} dB", f"≤ +{th['silence_margin_db']} dB",
            ds <= th["silence_margin_db"])
    else:
        chk("R4 Aucun bruit ajouté dans les silences", "pas de silence dans l'entrée", "non applicable", True)
    if intel:
        chk("R5 Intelligibilité (WER sortie vs entrée)", f"{intel['wer_out_vs_in']:.1%}",
            f"≤ {th['wer_max']:.0%}", intel["wer_out_vs_in"] <= th["wer_max"])

    return {"identity": ident, "summary_in": s_in, "summary_out": s_out, "contour": contour,
            "energy": energy, "pauses": pauses, "intelligibility": intel, "checks": checks}


# ---------------------------------------------------------------- figure et rapport

def plot(wav_in, wav_out, path, f0_shift: float | None = None, title: str = "") -> None:
    seg = detect_speech(wav_in)
    fig, ax = plt.subplots(5, 1, figsize=(13, 13), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.3, 1.3, 1.3, 1]})
    for a, b in seg.pauses:
        for k in (0, 3, 4):
            ax[k].axvspan(a, b, color="#f2c94c", alpha=0.25, lw=0)
    ax[0].plot(np.arange(len(wav_in)) / SR, wav_in, lw=0.5, color="#2f4b7c", label="entrée")
    ax[0].plot(np.arange(len(wav_out)) / SR, wav_out, lw=0.5, color="#d45087", alpha=0.7, label="sortie")
    ax[0].set_title("Forme d'onde (zones jaunes = pauses détectées dans l'entrée)")
    ax[0].legend(loc="upper right")
    for k, (w, nom) in enumerate([(wav_in, "entrée"), (wav_out, "sortie anonymisée")], start=1):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(w, n_fft=1024, hop_length=160)), ref=np.max)
        librosa.display.specshow(S, sr=SR, hop_length=160, x_axis="time", y_axis="hz", ax=ax[k],
                                 cmap="magma", vmin=-80)
        ax[k].set_ylim(0, 5000)
        ax[k].set_title(f"Spectrogramme — {nom}")
        ax[k].set_xlabel("")
    ti, fi = f0_contour(wav_in)
    to, fo = f0_contour(wav_out)
    ax[3].plot(ti, fi, ".", ms=3, color="#2f4b7c", label="F0 entrée")
    if f0_shift is not None:
        ax[3].plot(ti, clean_f0(fi) + f0_shift, "-", lw=1, color="#888",
                   label=f"cible = entrée {f0_shift:+.0f} Hz")
    ax[3].plot(to, fo, ".", ms=3, color="#d45087", label="F0 sortie")
    ax[3].set_ylabel("Hz")
    ax[3].set_title("Contour de F0")
    ax[3].legend(loc="upper right")
    ri, ro = frame_rms_db(wav_in), frame_rms_db(wav_out)
    ax[4].plot(np.arange(len(ri)) * HOP / SR, ri, color="#2f4b7c", lw=1, label="entrée")
    ax[4].plot(np.arange(len(ro)) * HOP / SR, ro, color="#d45087", lw=1, label="sortie")
    ax[4].set_ylim(max(-100, min(ri.min(), ro.min())), max(ri.max(), ro.max()) + 3)
    ax[4].set_ylabel("dB")
    ax[4].set_xlabel("temps (s)")
    ax[4].set_title("Énergie RMS par trame (40 ms)")
    ax[4].legend(loc="upper right")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


ROWS = [("duration_s", "Durée (s)", 3), ("f0_mean_hz", "F0 moyenne (Hz)", 1),
        ("f0_std_hz", "F0 écart-type (Hz)", 1), ("f0_range_hz", "Pitch range P95−P5 (Hz)", 1),
        ("voiced_ratio", "Taux de voisement", 3), ("rms_speech_db", "RMS parole (dB)", 2),
        ("rms_silence_db", "RMS silences (dB)", 2), ("n_pauses", "Nombre de pauses", 0),
        ("pause_total_s", "Durée totale des pauses (s)", 3), ("speech_ratio", "Proportion de parole", 3),
        ("jitter_local_pct", "Jitter local (%)", 2), ("shimmer_local_pct", "Shimmer local (%)", 2),
        ("hnr_db", "HNR (dB)", 1)]


def to_markdown(res: dict, info: dict | None, name: str) -> str:
    L = [f"# Comparaison entrée / sortie — {name}", ""]
    if info:
        L += [f"- Pseudo-locuteur : {', '.join(info['pseudo_speakers'])} (F0 moyenne {info['pseudo_f0_mean_hz']} Hz, "
              f"décalage {info['f0_shift_hz']:+} Hz, mode F0 : {info['f0_mode']}, règles R3/R4 : "
              f"{'oui' if info.get('rules_applied', True) else 'non'})", ""]
    idt = res["identity"]
    L += ["## Identité (ECAPA-TDNN, similarité cosinus)", "", "| Paire | cos |", "|---|---|",
          f"| entrée ↔ sortie | {idt['cos_in_out']:.3f} |"]
    for k, lab in [("cos_out_pseudo", "sortie ↔ pseudo-locuteur visé"),
                   ("cos_in_pseudo", "entrée ↔ pseudo-locuteur visé"),
                   ("cos_in_same_speaker_mean", "entrée ↔ autres enreg. du patient (référence « même personne »)"),
                   ("cos_out_same_speaker_mean", "sortie ↔ autres enreg. du patient")]:
        if k in idt:
            L.append(f"| {lab} | {idt[k]:.3f} |")
    L += ["", "## Vérifications", "", "| | Vérification | Valeur | Critère |", "|---|---|---|---|"]
    for c in res["checks"]:
        L.append(f"| {'✅' if c['ok'] else '❌'} | {c['check']} | {c['value']} | {c['rule']} |")
    L += ["", "## Mesures acoustiques (Praat)", "", "| Mesure | Entrée | Sortie | Δ |", "|---|---|---|---|"]
    for k, lab, d in ROWS:
        a, b = res["summary_in"][k], res["summary_out"][k]
        L.append(f"| {lab} | {a:.{d}f} | {b:.{d}f} | {b - a:+.{d}f} |")
    c = res["contour"]
    L += ["", f"Contour F0 : corrélation {c.get('f0_corr', float('nan')):.3f}, RMSE de forme "
              f"{c.get('f0_shape_rmse_hz', float('nan')):.1f} Hz, accord de voisement {c['voicing_agreement']:.0%}. "
              f"Enveloppe d'énergie (parole) : corrélation {res['energy']['envelope_corr_speech']:.3f}."]
    if res["intelligibility"]:
        it = res["intelligibility"]
        L += ["", f"## Intelligibilité (Whisper, langue détectée : {it['language']})", "",
              f"- Entrée : « {it['transcript_in']} »", f"- Sortie : « {it['transcript_out']} »",
              f"- WER sortie vs entrée : {it['wer_out_vs_in']:.1%}"]
    L += ["", "*Jitter, shimmer et HNR sont rapportés à titre informatif : le vocodeur régénère la forme "
              "d'onde, ces micro-perturbations ne sont donc pas garanties.*", ""]
    return "\n".join(L)


def save_report(res: dict, info: dict | None, report_dir: str, name: str, wav_in, wav_out) -> None:
    d = Path(report_dir)
    d.mkdir(parents=True, exist_ok=True)
    shift = info["f0_shift_hz"] if info and info.get("f0_mode") == "transfer" else None
    plot(wav_in, wav_out, d / f"{name}_comparaison.png", shift, title=f"{name} — entrée vs sortie anonymisée")
    (d / f"{name}_rapport.md").write_text(to_markdown(res, info, name), encoding="utf-8")
    (d / f"{name}_rapport.json").write_text(json.dumps({"info": info, **res}, indent=1, ensure_ascii=False,
                                                       default=float), encoding="utf-8")
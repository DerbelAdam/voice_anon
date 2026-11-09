"""Étape 2 — Prosodie (F0, énergie, durées) et marqueurs de qualité vocale (Praat).

Toutes les mesures sont faites avec Praat via parselmouth, pour rester comparables
avec ce qu'on mesurera plus tard dans l'évaluation (« mesures acoustiques Praat »).
"""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from .audio import SR, Segmentation, detect_speech

F0_MIN, F0_MAX = 60.0, 500.0
F0_STEP = 0.01  # 10 ms


def f0_contour(wav: np.ndarray, sr: int = SR, fmin: float = F0_MIN, fmax: float = F0_MAX,
               step: float = F0_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Contour de F0 (Hz) ; NaN sur les trames non voisées."""
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=step, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 <= 0] = np.nan
    return pitch.xs(), f0


def mean_f0(wav: np.ndarray, sr: int = SR) -> float:
    _, f0 = f0_contour(wav, sr)
    return float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else float("nan")


def voice_quality(wav: np.ndarray, sr: int = SR) -> dict:
    """Jitter local, shimmer local (%) et HNR (dB) — paramètres Praat standard."""
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
    pp = call(snd, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
    jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harm = call(snd, "To Harmonicity (cc)", 0.01, F0_MIN, 0.1, 1.0)
    hnr = call(harm, "Get mean", 0, 0)
    return {"jitter_local_pct": 100 * jitter, "shimmer_local_pct": 100 * shimmer, "hnr_db": hnr}


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def summarize(wav: np.ndarray, sr: int = SR, seg: Segmentation | None = None) -> dict:
    """Résumé prosodique d'un signal (ce que l'anonymisation doit préserver)."""
    seg = seg or detect_speech(wav, sr)
    _, f0 = f0_contour(wav, sr)
    v = f0[~np.isnan(f0)]
    mask = seg.speech_mask(len(wav), sr)
    pause_d = [b - a for a, b in seg.pauses]
    speech_time = sum(b - a for a, b in seg.speech)
    out = {
        "duration_s": seg.duration,
        "f0_mean_hz": float(np.mean(v)) if len(v) else float("nan"),
        "f0_std_hz": float(np.std(v)) if len(v) else float("nan"),
        "f0_p5_hz": float(np.percentile(v, 5)) if len(v) else float("nan"),
        "f0_p95_hz": float(np.percentile(v, 95)) if len(v) else float("nan"),
        "voiced_ratio": float(len(v) / max(1, len(f0))),
        "rms_db": rms_db(wav),
        "rms_speech_db": rms_db(wav[mask]) if mask.any() else float("nan"),
        "rms_silence_db": rms_db(wav[~mask]) if (~mask).any() else float("nan"),
        "n_pauses": len(seg.pauses),
        "pause_total_s": float(sum(pause_d)),
        "pause_mean_s": float(np.mean(pause_d)) if pause_d else 0.0,
        "speech_time_s": float(speech_time),
        "speech_ratio": float(speech_time / seg.duration) if seg.duration else 0.0,
    }
    out["f0_range_hz"] = out["f0_p95_hz"] - out["f0_p5_hz"]  # range robuste (P95 − P5)
    out.update(voice_quality(wav, sr))
    return out
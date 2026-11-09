"""Étape 5 — Conversion : contenu (WavLM) → trames du pseudo-locuteur (kNN) → audio
(HiFi-GAN), puis règles de protection prosodique.

Correspondance avec l'architecture du projet :
  contenu linguistique  : WavLM-Large, couche 6 (une trame / 20 ms)
  identité / timbre      : chaque trame du patient est remplacée par la moyenne de ses
                           k plus proches voisines dans le réservoir du pseudo-locuteur
  prosodie               : F0 d'origine réimposée, décalée seulement en moyenne (Praat PSOLA)
  reconstruction         : vocodeur HiFi-GAN « prematched » de kNN-VC
Règles de protection :
  R1 débit/pauses  : conversion trame à trame → même nombre de trames, même durée
  R2 pitch range   : contour F0 cible = F0 patient + Δ (addition, en Hz)
  R3 énergie RMS   : volume de sortie ramené au RMS (parole) de l'original
  R4 pauses        : dans les silences de l'original, la sortie ne peut pas être plus forte
  R5 texte         : vérifié a posteriori (WER), étape 6
"""
from __future__ import annotations

import time

import librosa
import numpy as np
import parselmouth
import torch
from parselmouth.praat import call

from .audio import FRAME, HOP, SR, detect_speech
from .knnvc import load_knnvc
from .prosody import f0_contour
from .pseudo_speaker import PseudoSpeaker, SpeakerPool
from .speaker import SpeakerEncoder

WAVLM_WIN = 400  # fenêtre du premier étage convolutif de WavLM (en échantillons)


# ---------------------------------------------------------------- règles prosodiques

def clean_f0(f0: np.ndarray, win: int = 15, max_octave: float = 0.6) -> np.ndarray:
    """Retire les erreurs d'octave de Praat (points isolés à ×2 ou ÷2) : un point qui
    s'écarte de plus de `max_octave` octave de la médiane locale (`win` trames voisées)
    est mis à NaN. Ils ne sont alors pas recopiés dans la voix de sortie."""
    out = f0.copy()
    idx = np.where(np.isfinite(f0))[0]
    if len(idx) < 3:
        return out
    v = np.log2(f0[idx])
    h = win // 2
    med = np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(len(v))])
    out[idx[np.abs(v - med) > max_octave]] = np.nan
    return out


def impose_f0(wav: np.ndarray, times: np.ndarray, f0_target: np.ndarray, sr: int = SR) -> np.ndarray:
    """R2 : remplace le contour de F0 de `wav` par `f0_target` (PSOLA de Praat).
    La durée est inchangée (overlap-add sans modification temporelle)."""
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
    manip = call(snd, "To Manipulation", 0.01, 50.0, 600.0)
    tier = call("Create PitchTier", "target", 0.0, snd.duration)
    for t, f in zip(times, f0_target):
        if np.isfinite(f):
            call(tier, "Add point", float(t), float(np.clip(f, 50.0, 600.0)))
    call([tier, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)").values[0]
    out = out[:len(wav)] if len(out) >= len(wav) else np.pad(out, (0, len(wav) - len(out)))
    return out.astype(np.float32)


def match_rms(out: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """R3 : même RMS que l'original, mesuré sur les zones de parole."""
    m = mask if mask.any() else np.ones_like(mask)
    g = np.sqrt(np.mean(ref[m] ** 2)) / (np.sqrt(np.mean(out[m] ** 2)) + 1e-9)
    return (out * g).astype(np.float32)


def cap_silences(out: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """R4 : dans les silences de l'original, l'énergie de la sortie (par trame de
    40 ms) est plafonnée à celle de l'original → aucun bruit ajouté, les pauses
    restent détectables de la même façon."""
    r_in = librosa.feature.rms(y=ref, frame_length=FRAME, hop_length=HOP)[0]
    r_out = librosa.feature.rms(y=out, frame_length=FRAME, hop_length=HOP)[0]
    n = min(len(r_in), len(r_out))
    centers = np.minimum(np.arange(n) * HOP, len(mask) - 1)
    silent = ~mask[centers]
    gain = np.ones(n)
    gain[silent] = np.minimum(1.0, r_in[:n][silent] / (r_out[:n][silent] + 1e-9))
    # lissage (min glissant sur 3 trames) pour éviter les clics aux frontières
    gain = np.minimum.reduce([gain, np.roll(gain, 1), np.roll(gain, -1)])
    g = np.interp(np.arange(len(out)), np.arange(n) * HOP, gain)
    return (out * g).astype(np.float32)


# ---------------------------------------------------------------- pipeline

class Anonymizer:
    def __init__(self, pool_cache: str, device: str | None = None,
                 encoder: SpeakerEncoder | None = None, knnvc=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder or SpeakerEncoder(device=self.device)
        self.knnvc = knnvc or load_knnvc(self.device)
        self.pool = SpeakerPool(pool_cache)

    @staticmethod
    def segment_bounds(wav: np.ndarray, max_s: float = 15.0) -> list[tuple[int, int]]:
        """Découpe en segments de `max_s` secondes au plus, en coupant AU MILIEU DES
        PAUSES (jamais dans un mot). Bornes en numéros de trames de 20 ms.
        S'il n'y a pas de pause assez tôt, on coupe à `max_s` (cas rare)."""
        n_frames = -(-len(wav) // HOP)  # division arrondie au supérieur
        max_f = int(max_s * SR / HOP)
        cuts = sorted(int((a + b) / 2 * SR / HOP) for a, b in detect_speech(wav).pauses)
        bounds, start = [], 0
        while n_frames - start > max_f:
            possibles = [c for c in cuts if start < c <= start + max_f]
            end = possibles[-1] if possibles else start + max_f  # la pause la plus tardive possible
            bounds.append((start, end))
            start = end
        bounds.append((start, n_frames))
        return bounds

    @torch.inference_mode()
    def wavlm_features(self, wav: np.ndarray, max_s: float = 15.0) -> tuple[torch.Tensor, int]:
        """Trames WavLM d'un signal de N échantillons -> ceil(N / 320) trames.
        Chaque segment (coupé dans une pause) est traité séparément, comme une
        phrase : c'est la durée sur laquelle WavLM a été entraîné, et cela évite de
        saturer la mémoire du GPU sur les longs enregistrements."""
        bounds = self.segment_bounds(wav, max_s)
        n_frames = bounds[-1][1]
        padded = np.pad(wav, (0, n_frames * HOP + (WAVLM_WIN - HOP) - len(wav)))
        out = []
        for a, b in bounds:
            seg = torch.from_numpy(padded[a * HOP: b * HOP + (WAVLM_WIN - HOP)])[None]
            out.append(self.knnvc.get_features(seg, vad_trigger_level=0))  # (b - a, 1024)
        return torch.cat(out, dim=0), len(bounds)

    @torch.inference_mode()
    def knn_regress(self, query: torch.Tensor, matching: torch.Tensor, topk: int = 4,
                    chunk: int = 1000) -> torch.Tensor:
        """Pour chaque trame du patient : moyenne de ses `topk` plus proches voisines
        (similarité cosinus) dans le réservoir du pseudo-locuteur. Par paquets de
        `chunk` trames pour limiter la mémoire."""
        m = matching.to(self.device)
        m_unit = torch.nn.functional.normalize(m, dim=-1)
        out = []
        for i in range(0, len(query), chunk):
            q = torch.nn.functional.normalize(query[i:i + chunk].to(self.device), dim=-1)
            idx = (q @ m_unit.T).topk(topk, dim=-1).indices  # (chunk, topk)
            out.append(m[idx].mean(dim=1))
        return torch.cat(out, dim=0)

    @torch.inference_mode()
    def convert_frames(self, wav: np.ndarray, matching: torch.Tensor, topk: int = 4) -> np.ndarray:
        """WavLM -> kNN -> HiFi-GAN ; sortie exactement de la même longueur (R1)."""
        query, self.last_n_segments = self.wavlm_features(wav)
        feats = self.knn_regress(query, matching, topk)
        y = self.knnvc.vocode(feats[None]).cpu().squeeze().numpy()  # 320 échantillons / trame
        n = len(wav)
        return (y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))).astype(np.float32)

    def anonymize(self, wav: np.ndarray, patient_id: str = "", patient_emb: np.ndarray | None = None,
                  patient_f0: float | None = None, f0_mode: str = "transfer", topk: int = 4,
                  n_farthest: int = 10, n_select: int = 5, gender: str = "same",
                  apply_rules: bool = True) -> tuple[np.ndarray, dict, PseudoSpeaker]:
        """`patient_emb` / `patient_f0` : valeurs au niveau du PATIENT (tous ses fichiers),
        pour que tous ses enregistrements reçoivent le même pseudo-locuteur et le même Δ."""
        t0 = time.time()
        seg = detect_speech(wav)
        mask = seg.speech_mask(len(wav))
        times, f0 = f0_contour(wav)
        f0_patient = patient_f0 if patient_f0 is not None else float(np.nanmean(f0))
        emb = patient_emb if patient_emb is not None else self.encoder.embed(wav)

        ps = self.pool.select(emb, f0_patient, patient_id, n_farthest, n_select, gender)
        matching = torch.cat([self.pool.feats(s) for s in ps.speakers], dim=0)

        out = self.convert_frames(wav, matching, topk)
        delta = ps.f0_mean - f0_patient
        if f0_mode == "transfer":
            out = impose_f0(out, times, clean_f0(f0) + delta)
        if apply_rules:
            out = match_rms(out, wav, mask)
            out = cap_silences(out, wav, mask)

        info = {
            "patient_id": patient_id,
            "patient_f0_mean_hz": round(f0_patient, 2),
            "patient_gender_est": ps.gender_patient,
            "pseudo_speakers": ps.speakers,
            "cos_patient_vs_selected": ps.similarities,
            "pseudo_f0_mean_hz": round(ps.f0_mean, 2),
            "f0_shift_hz": round(delta, 2),
            "f0_mode": f0_mode,
            "topk": topk,
            "matching_set_frames": int(matching.shape[0]),
            "wavlm_segments": self.last_n_segments,
            "candidates_considered": ps.candidates_considered,
            "rules_applied": apply_rules,
            "pseudo_embedding": ps.embedding.tolist(),
            "seconds": round(time.time() - t0, 2),
        }
        return out, info, ps
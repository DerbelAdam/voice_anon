"""Étape 1 — Prétraitement : chargement 16 kHz mono + détection de parole (VAD).

Le niveau du signal n'est PAS normalisé ici : l'entrée sert aussi de référence
pour la règle « énergie RMS » (le volume de sortie est ramené à celui de l'entrée).
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

SR = 16000
HOP = 320    # 20 ms : même pas que WavLM / HiFi-GAN
FRAME = 640  # 40 ms


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Charge un fichier audio en mono float32 à `sr` Hz, sans offset DC."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    wav = wav - np.mean(wav)
    return wav.astype(np.float32)


def save_audio(path: str, wav: np.ndarray, sr: int = SR) -> None:
    peak = np.max(np.abs(wav)) if len(wav) else 0.0
    if peak > 0.999:  # évite l'écrêtage à l'écriture
        wav = wav * (0.999 / peak)
    sf.write(path, wav.astype(np.float32), sr, subtype="PCM_16")


def frame_rms_db(wav: np.ndarray, hop: int = HOP, frame: int = FRAME) -> np.ndarray:
    rms = librosa.feature.rms(y=wav, frame_length=frame, hop_length=hop, center=True)[0]
    return 20 * np.log10(rms + 1e-8)


@dataclass
class Segmentation:
    """Résultat de la VAD, en secondes."""
    speech: list[tuple[float, float]]   # intervalles de parole
    pauses: list[tuple[float, float]]   # silences internes (hors début/fin)
    duration: float

    def speech_mask(self, n_samples: int, sr: int = SR) -> np.ndarray:
        m = np.zeros(n_samples, dtype=bool)
        for a, b in self.speech:
            m[int(a * sr):int(b * sr)] = True
        return m


def detect_speech(wav: np.ndarray, sr: int = SR, top_db: float = 35.0,
                  min_pause: float = 0.15, min_speech: float = 0.05) -> Segmentation:
    """VAD par énergie, relative au pic du fichier.

    - top_db    : une trame est « silence » si elle est `top_db` dB sous le maximum
    - min_pause : un silence interne plus court n'est pas compté comme pause
      (150 ms est un seuil courant dans la littérature sur les pauses)
    """
    intervals = librosa.effects.split(wav, top_db=top_db, frame_length=FRAME, hop_length=HOP)
    segs = [[a / sr, b / sr] for a, b in intervals if (b - a) / sr >= min_speech]
    # fusionne les segments séparés par un silence < min_pause
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_pause:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    speech = [(float(a), float(b)) for a, b in merged]
    pauses = [(speech[i][1], speech[i + 1][0]) for i in range(len(speech) - 1)]
    return Segmentation(speech=speech, pauses=pauses, duration=len(wav) / sr)
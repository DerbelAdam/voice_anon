"""Étape 4 — Pool de voix candidates et construction du pseudo-locuteur.

Stratégie (baseline du VoicePrivacy Challenge, Fang et al. 2019 ; Srivastava et al. 2020) :
  1. embedding ECAPA de chaque locuteur du pool et du patient ;
  2. on garde les candidats du même genre que le patient (option `gender`) ;
  3. on trie par similarité cosinus et on garde les `n_farthest` plus éloignés ;
  4. on en tire `n_select` au hasard (graine dérivée de l'ID patient → le même
     patient reçoit toujours le même pseudo-locuteur) ;
  5. le pseudo-locuteur = la « moyenne » de ces voix.

Avec kNN-VC, « moyenner » les voix revient à réunir leurs trames WavLM dans un
seul ensemble de correspondance (matching set). On calcule aussi l'embedding
ECAPA moyen de ces voix (identité cible théorique), utile pour l'évaluation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import SR, detect_speech, frame_rms_db, load_audio
from .prosody import mean_f0
from .speaker import SpeakerEncoder, cosine, l2norm

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
GENDER_F0_THRESHOLD = 165.0  # Hz : estimation grossière homme/femme par la F0 moyenne


def estimate_gender(f0_mean: float) -> str:
    return "F" if f0_mean >= GENDER_F0_THRESHOLD else "M"


def list_speakers(pool_dir: str) -> dict[str, list[Path]]:
    """Un sous-dossier de premier niveau = un locuteur (compatible LibriSpeech :
    pool/<speaker>/<chapter>/*.flac, mais aussi pool/<speaker>/*.wav)."""
    if not Path(pool_dir).is_dir():
        raise FileNotFoundError(f"Dossier du pool introuvable : {Path(pool_dir).resolve()}")
    out = {}
    for d in sorted(Path(pool_dir).iterdir()):
        if d.is_dir():
            files = sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT)
            if files:
                out[d.name] = files
    return out


MIN_LEVEL_DB = -45.0  # en dessous (trame la plus forte), le fichier est considéré vide


def trim_silences(wav: np.ndarray) -> np.ndarray | None:
    """Retire les silences de début et de fin avec NOTRE VAD (étape 1).
    Renvoie None si le fichier ne contient pas de parole exploitable : trop faible
    (silence, bruit seul) ou moins de 0,5 s de parole."""
    if len(wav) < SR // 2 or frame_rms_db(wav).max() < MIN_LEVEL_DB:
        return None
    seg = detect_speech(wav)
    if not seg.speech:
        return None
    w = wav[int(seg.speech[0][0] * SR):int(seg.speech[-1][1] * SR)]
    return w if len(w) >= SR // 2 else None


def pool_is_complete(pool_dir: str, cache_dir: str, max_speakers: int | None = None) -> bool:
    """Vrai si toutes les voix du dossier sont déjà dans le cache."""
    idx = Path(cache_dir) / "index.json"
    if not idx.exists():
        return False
    voulus = list(list_speakers(pool_dir))[:max_speakers] if max_speakers else list(list_speakers(pool_dir))
    return set(voulus) <= set(json.loads(idx.read_text()))


def build_pool(pool_dir: str, cache_dir: str, encoder: SpeakerEncoder, knnvc,
               max_seconds: float = 300.0, max_speakers: int | None = None, log=print) -> None:
    """Pré-calcule, pour chaque locuteur du pool : embedding ECAPA, F0 moyenne,
    genre estimé et trames WavLM (matching set kNN-VC).

    Reprenable : l'index est enregistré après CHAQUE voix ; une voix déjà dans le
    cache n'est pas recalculée. Si le calcul s'interrompt, relancez simplement."""
    cache = Path(cache_dir)
    (cache / "feats").mkdir(parents=True, exist_ok=True)
    speakers = list_speakers(pool_dir)
    if not speakers:
        raise ValueError(f"Aucun sous-dossier de locuteur avec de l'audio dans {pool_dir}")
    if max_speakers:
        speakers = dict(list(speakers.items())[:max_speakers])
    idx_path = cache / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    for i, (spk, files) in enumerate(speakers.items(), 1):
        if spk in index and ("ignored" in index[spk] or (cache / "feats" / f"{spk}.pt").exists()):
            log(f"  [{i:3d}/{len(speakers)}] {spk:>12} : déjà traité")
            continue
        wavs, total, ignores = [], 0.0, 0
        for f in files:
            w = trim_silences(load_audio(str(f)))
            if w is None:  # fichier sans parole détectable : on l'ignore
                ignores += 1
                continue
            wavs.append(w)
            total += len(w) / SR
            if total >= max_seconds:
                break
        if not wavs:
            index[spk] = {"ignored": "aucune parole exploitable"}  # noté, pour ne pas réessayer
            idx_path.write_text(json.dumps(index, indent=1))
            log(f"  [{i:3d}/{len(speakers)}] {spk:>12} : aucune parole exploitable, voix ignorée")
            continue
        emb = encoder.embed_many(wavs)
        f0 = float(np.nanmean([mean_f0(w) for w in wavs]))
        # silences déjà retirés par trim_silences -> on désactive la VAD interne de kNN-VC
        feats = knnvc.get_matching_set([torch.from_numpy(w)[None] for w in wavs], vad_trigger_level=0)
        torch.save(feats.half(), cache / "feats" / f"{spk}.pt")
        index[spk] = {"emb": emb.tolist(), "f0_mean": f0, "gender": estimate_gender(f0),
                      "seconds": round(total, 1), "n_files": len(wavs)}
        idx_path.write_text(json.dumps(index, indent=1))  # sauvegarde après chaque voix
        note = f", {ignores} fichier(s) ignoré(s)" if ignores else ""
        log(f"  [{i:3d}/{len(speakers)}] {spk:>12} : {total:6.1f} s, F0 moyenne {f0:5.1f} Hz "
            f"({index[spk]['gender']}), {feats.shape[0]} trames WavLM{note}")
    n_ok = sum("emb" in v for v in index.values())
    log(f"Pool : {n_ok} locuteurs utilisables ({len(index) - n_ok} ignoré(s)) -> {cache}")


@dataclass
class PseudoSpeaker:
    speakers: list[str]
    similarities: dict[str, float]  # cos(patient, candidat) des locuteurs retenus
    embedding: np.ndarray           # embedding ECAPA moyen (identité cible)
    f0_mean: float                  # F0 moyenne du pseudo-locuteur
    gender_patient: str
    candidates_considered: int
    ranking: list[tuple[str, float]]  # tous les candidats, du plus éloigné au plus proche


class SpeakerPool:
    def __init__(self, cache_dir: str):
        self.cache = Path(cache_dir)
        index = json.loads((self.cache / "index.json").read_text())
        self.index = {s: v for s, v in index.items() if "emb" in v}  # sans les voix ignorées
        if not self.index:
            raise ValueError(f"Pool vide : {cache_dir}")

    def __len__(self):
        return len(self.index)

    def feats(self, spk: str) -> torch.Tensor:
        return torch.load(self.cache / "feats" / f"{spk}.pt").float()

    def select(self, patient_emb: np.ndarray, patient_f0: float, patient_id: str = "",
               n_farthest: int = 10, n_select: int = 5, gender: str = "same",
               exclude: set[str] | None = None) -> PseudoSpeaker:
        g = estimate_gender(patient_f0)
        cands = {s: v for s, v in self.index.items() if s not in (exclude or set())}
        if gender == "same":
            same = {s: v for s, v in cands.items() if v["gender"] == g}
            cands = same or cands  # repli si aucun candidat du même genre
        sims = {s: cosine(patient_emb, np.array(v["emb"])) for s, v in cands.items()}
        ranking = sorted(sims.items(), key=lambda kv: kv[1])  # du plus éloigné au plus proche
        farthest = [s for s, _ in ranking[:max(1, min(n_farthest, len(ranking)))]]
        seed = int(hashlib.sha256(patient_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        k = max(1, min(n_select, len(farthest)))
        chosen = sorted(rng.choice(farthest, size=k, replace=False).tolist())
        emb = l2norm(np.mean([np.array(self.index[s]["emb"]) for s in chosen], axis=0))
        f0 = float(np.mean([self.index[s]["f0_mean"] for s in chosen]))
        return PseudoSpeaker(chosen, {s: round(sims[s], 4) for s in chosen}, emb, f0, g,
                             len(cands), ranking)
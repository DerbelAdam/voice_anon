"""Étape 3 — Identité / timbre : embedding ECAPA-TDNN (192 dimensions).

Modèle pré-entraîné sur VoxCeleb2 (TaoRuijie/ECAPA-TDNN, EER ≈ 0,96 % sur Vox1-O).
L'embedding est normalisé L2, donc la similarité cosinus = produit scalaire.
"""
from __future__ import annotations

import numpy as np
import torch

from . import _paths
from .audio import SR


class SpeakerEncoder:
    def __init__(self, ckpt: str | None = None, device: str = "cpu"):
        _paths.require("ecapa")
        from model import ECAPA_TDNN  # third_party/ecapa/model.py

        ckpt = ckpt or str(_paths.MODELS / "ecapa_pretrain.model")
        self.device = torch.device(device)
        self.model = ECAPA_TDNN(C=1024)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        # le checkpoint contient aussi la tête de classification (speaker_loss.*)
        enc = {k[len("speaker_encoder."):]: v for k, v in state.items()
               if k.startswith("speaker_encoder.")}
        self.model.load_state_dict(enc)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Embedding d'un signal 16 kHz -> vecteur (192,) normalisé."""
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32))[None].to(self.device)
        if x.shape[1] < SR // 2:  # < 0,5 s : on répète pour avoir assez de contexte
            x = x.repeat(1, int(np.ceil(SR / 2 / x.shape[1])))
        e = self.model(x, aug=False)[0].cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def embed_many(self, wavs: list[np.ndarray]) -> np.ndarray:
        """Embedding moyen de plusieurs fichiers d'un même locuteur."""
        return l2norm(np.mean([self.embed(w) for w in wavs], axis=0))


def l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2norm(a), l2norm(b)))
"""Chargement de kNN-VC (Baas et al., 2023) : WavLM-Large (encodeur) + HiFi-GAN (vocodeur).

- WavLM transforme l'audio en une suite de vecteurs de 1024 nombres (un toutes les 20 ms).
- HiFi-GAN fait l'inverse : il reconstruit un signal audio à partir de ces vecteurs.
- KNeighborsVC relie les deux et contient la recherche des k plus proches voisins.

Séparé de convert.py car on en a besoin dès l'étape 4 (pré-calcul du pool).
"""
from __future__ import annotations

import json

import torch

from . import _paths


def load_knnvc(device: str = "cpu"):
    _paths.require("knn-vc")
    from hifigan.models import Generator as HiFiGAN
    from hifigan.utils import AttrDict
    from matcher import KNeighborsVC
    from wavlm.WavLM import WavLM, WavLMConfig

    # vocodeur HiFi-GAN (version « prematched », la plus robuste d'après les auteurs)
    kd = _paths.THIRD_PARTY / "knn-vc"
    h = AttrDict(json.loads((kd / "hifigan" / "config_v1_wavlm.json").read_text()))
    gen = HiFiGAN(h)
    sd = torch.load(_paths.MODELS / "prematch_g_02500000.pt", map_location="cpu", weights_only=False)
    gen.load_state_dict(sd["generator"])
    gen.eval().remove_weight_norm()

    # encodeur WavLM-Large (316 M paramètres)
    ck = torch.load(_paths.MODELS / "WavLM-Large.pt", map_location="cpu", weights_only=False)
    wavlm = WavLM(WavLMConfig(ck["cfg"]))
    wavlm.load_state_dict(ck["model"])
    wavlm.eval()
    return KNeighborsVC(wavlm.to(device), gen.to(device), h, device)
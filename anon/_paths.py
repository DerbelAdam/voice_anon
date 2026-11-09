"""Chemins du projet et accès au code tiers (ECAPA-TDNN, puis kNN-VC à l'étape 4)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # le dossier voice_anon/
MODELS = ROOT / "models"
THIRD_PARTY = ROOT / "third_party"


def require(name: str) -> Path:
    """Rend importable le code tiers `third_party/<name>` ; message clair s'il manque."""
    p = THIRD_PARTY / name
    if not p.exists():
        cmd = {"ecapa": "ecapa", "knn-vc": "knnvc"}.get(name, "all")
        raise RuntimeError(f"{p} introuvable : lancez d'abord  python setup_models.py {cmd}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p
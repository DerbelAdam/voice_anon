"""Télécharge le code tiers et les poids pré-entraînés (Windows, Linux, Mac).

    python setup_models.py ecapa     # étape 3 : ECAPA-TDNN (≈ 66 Mo)
    python setup_models.py knnvc     # étape 4 : kNN-VC = WavLM-Large + HiFi-GAN (≈ 1,3 Go)
    python setup_models.py all
"""
import http.client
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
THIRD_PARTY = ROOT / "third_party"

ECAPA_RAW = "https://raw.githubusercontent.com/TaoRuijie/ECAPA-TDNN/main"
KNNVC_RAW = "https://raw.githubusercontent.com/bshall/knn-vc/master"
KNNVC_FILES = ["LICENSE", "matcher.py", "knnvc_utils.py",
               "hifigan/models.py", "hifigan/utils.py", "hifigan/config_v1_wavlm.json",
               "wavlm/WavLM.py", "wavlm/modules.py"]
KNNVC_REL = "https://github.com/bshall/knn-vc/releases/download/v0.1"


def _taille(n: int) -> str:
    return f"{n / 1e6:.1f} Mo" if n >= 1e6 else f"{n / 1e3:.0f} Ko"


def download(url: str, dst: Path, essais: int = 20) -> None:
    """Téléchargement avec REPRISE : si la connexion coupe, on redemande seulement
    la partie manquante (en-tête HTTP « Range »), jusqu'à `essais` fois."""
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  déjà présent : {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  téléchargement : {url}")
    for essai in range(1, essais + 1):
        deja = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={deja}-"} if deja else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if deja and r.status != 206:  # le serveur ignore la reprise : on recommence
                    deja = 0
                total = deja + int(r.headers.get("Content-Length", 0))
                done = deja
                with open(tmp, "ab" if deja else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        print(f"\r    {_taille(done)} / {_taille(total)}", end="", flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 416 and deja:  # « plage non satisfiable » : le fichier est déjà complet
                break
            raise
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            print(f"\n    coupure ({type(e).__name__}) : reprise {essai}/{essais}")
            continue
        print()
        if done >= total:
            break
        print(f"    incomplet ({_taille(done)} / {_taille(total)}) : reprise {essai}/{essais}")
    else:
        raise RuntimeError(f"Échec après {essais} essais : relancez la commande (la reprise continuera).")
    tmp.replace(dst)  # on ne renomme qu'une fois complet


def setup_ecapa() -> None:
    print("ECAPA-TDNN (TaoRuijie/ECAPA-TDNN, pré-entraîné VoxCeleb2)")
    download(f"{ECAPA_RAW}/model.py", THIRD_PARTY / "ecapa" / "model.py")
    download(f"{ECAPA_RAW}/exps/pretrain.model", MODELS / "ecapa_pretrain.model")


def setup_knnvc() -> None:
    print("kNN-VC (bshall/knn-vc, licence MIT)")
    # seuls les fichiers nécessaires à l'inférence (pas l'entraînement)
    for f in KNNVC_FILES:
        download(f"{KNNVC_RAW}/{f}", THIRD_PARTY / "knn-vc" / f)
    download(f"{KNNVC_REL}/prematch_g_02500000.pt", MODELS / "prematch_g_02500000.pt")
    download(f"{KNNVC_REL}/WavLM-Large.pt", MODELS / "WavLM-Large.pt")


if __name__ == "__main__":
    quoi = sys.argv[1] if len(sys.argv) > 1 else "all"
    if quoi in ("ecapa", "all"):
        setup_ecapa()
    if quoi in ("knnvc", "all"):
        setup_knnvc()
        
    print("Terminé.")
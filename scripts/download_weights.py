"""Download BOTHROS model weights from Hugging Face Hub into release/weights/.

Weights are licensed CC BY-NC-SA 4.0 (see WEIGHTS_LICENSE), separate from the
MIT code. ~233 MB total (one detector + two classifiers).

    python3 scripts/download_weights.py
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "JMacD263/linear-a-linear-b-bothros"   # weights repo on Hugging Face Hub
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

FILES = {
    "detector":     "yolo_aegean_unified.pt",      # unified Aegean detector (~19 MB)
    "la_classifier": "la_classifier.pth",          # Linear A ConvNeXt-Tiny (~107 MB)
    "lb_classifier": "lb_classifier.pth",          # Linear B ConvNeXt-Tiny (~107 MB)
    "lb_class_map":  "lb_class_to_reading.json",   # B-code -> reading map
}


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, fname in FILES.items():
        print(f"Downloading {name}: {fname} ...")
        path = hf_hub_download(repo_id=HF_REPO, filename=fname,
                               local_dir=str(WEIGHTS_DIR))
        print(f"  -> {path}")
    print(f"\nAll weights in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()

"""Download BOTHROS model weights from Hugging Face Hub into release/weights/.

Weights are licensed CC BY-NC-SA 4.0 (see WEIGHTS_LICENSE), separate from the
MIT code. ~244 MB total (one detector + two classifiers). Public repo, no token.

    python3 scripts/download_weights.py            # benchmark weights (the published numbers)
    python3 scripts/download_weights.py --release  # full-data 'release' variants (NOT benchmarkable)
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "JMacD263/linear-a-linear-b-bothros"   # weights repo on Hugging Face Hub
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

BENCHMARK = {
    "detector":      "yolo_aegean_unified.pt",     # unified Aegean detector (~19 MB)
    "la_classifier": "la_classifier.pth",          # Linear A ConvNeXt-Tiny (~113 MB)
    "lb_classifier": "lb_classifier.pth",          # Linear B ConvNeXt-Tiny (~112 MB)
}
RELEASE = {  # full-data variants — max-capability, NOT benchmarkable (trained incl. held-out)
    "detector":      "yolo_aegean_release.pt",
    "la_classifier": "la_classifier_release.pth",
    "lb_classifier": "lb_classifier_release.pth",
}
SHARED = {"lb_class_map": "lb_class_to_reading.json"}   # B-code -> reading map (both)


def main():
    ap = argparse.ArgumentParser(description="Download BOTHROS weights from Hugging Face.")
    ap.add_argument("--release", action="store_true",
                    help="fetch the full-data 'release' variants (NOT benchmarkable) "
                         "instead of the benchmark weights")
    args = ap.parse_args()
    files = {**(RELEASE if args.release else BENCHMARK), **SHARED}
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(("RELEASE" if args.release else "BENCHMARK") + " weights -> " + str(WEIGHTS_DIR))
    for name, fname in files.items():
        print(f"Downloading {name}: {fname} ...")
        path = hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=str(WEIGHTS_DIR))
        print(f"  -> {path}")
    print("\nDone." + ("  Run:  python3 -m bothros read <img> --script la --release"
                        if args.release else ""))


if __name__ == "__main__":
    main()

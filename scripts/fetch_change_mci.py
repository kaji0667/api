"""Fetch and seal the author's fixed pretrained MCI checkpoint, without pip."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from tasks.change.inference import ARCHITECTURE_ID, CHECKPOINT_SHA256, MEAN, STD, file_sha256

REVISION = "1e679060173434cd22fd22a52effbf3c785c97a0"
URL = f"https://huggingface.co/lcybuaa/Change-Agent/resolve/{REVISION}/MCI_model.pth?download=true"
SIZE = 913762855


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=CODE_DIR / "runs/change/mci")
    parser.add_argument("--local-checkpoint", type=Path, help="Use an already downloaded exact author file")
    args = parser.parse_args()
    root = args.bundle_dir.expanduser().resolve()
    (root / "weights").mkdir(parents=True, exist_ok=True)
    checkpoint = root / "weights/MCI_model.pth"
    if args.local_checkpoint and args.local_checkpoint.resolve() != checkpoint:
        if checkpoint.exists():
            raise ValueError("Refusing to overwrite existing model weights")
        if file_sha256(args.local_checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Local file is not the released MCI checkpoint")
        shutil.copyfile(args.local_checkpoint, checkpoint)
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix(".pth.partial")
        if temporary.exists():
            raise ValueError("A previous partial download exists; inspect it before retrying")
        print(f"Downloading public MCI: {SIZE} bytes from {URL}", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as source, temporary.open("xb") as target:
            downloaded = 0
            while block := source.read(1024 * 1024):
                downloaded += len(block)
                if downloaded > SIZE:
                    raise ValueError("Unexpected MCI download size")
                target.write(block)
        if temporary.stat().st_size != SIZE or file_sha256(temporary) != CHECKPOINT_SHA256:
            raise ValueError("MCI download checksum or size mismatch")
        temporary.rename(checkpoint)
    if checkpoint.stat().st_size != SIZE or file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("MCI checkpoint checksum or size mismatch")
    vocabulary = root / "vocab.json"
    source_vocab = CODE_DIR / "third_party/change_agent/vocab.json"
    vocab_bytes = source_vocab.read_bytes()
    if vocabulary.exists() and vocabulary.read_bytes() != vocab_bytes:
        raise ValueError("Refusing to overwrite a different MCI vocabulary")
    vocabulary.write_bytes(vocab_bytes)
    inactive_keys = root / "checkpoint_inactive_keys.json"
    inactive_bytes = (CODE_DIR / "third_party/change_agent/checkpoint_inactive_keys.json").read_bytes()
    if inactive_keys.exists() and inactive_keys.read_bytes() != inactive_bytes:
        raise ValueError("Refusing to overwrite a different checkpoint compatibility audit")
    inactive_keys.write_bytes(inactive_bytes)
    manifest = {
        "architecture_id": ARCHITECTURE_ID,
        "source": {"code": "https://github.com/Chen-Yang-Liu/Change-Agent",
                   "code_revision": "68cbaa7f388b36e4fc10872f7a2911482d26ae5b",
                   "model": "https://huggingface.co/lcybuaa/Change-Agent", "model_revision": REVISION},
        "checkpoint": {"path": "weights/MCI_model.pth", "sha256": CHECKPOINT_SHA256, "size": SIZE},
        "vocabulary": {"path": "vocab.json", "sha256": hashlib.sha256(vocab_bytes).hexdigest()},
        "checkpoint_inactive_keys": {"path": "checkpoint_inactive_keys.json",
                                     "sha256": hashlib.sha256(inactive_bytes).hexdigest()},
        "preprocessing": {"size": 256, "interpolation": "opencv_linear",
                          "mean": MEAN.tolist(), "std": STD.tolist()},
        "configuration": {"network": "segformer-mit_b1", "n_layers": 3, "heads": 8,
                          "encoder_dim": 512, "decoder_n_layers": 1, "max_length": 41,
                          "feature_size": [16, 16, 512], "decoding": "greedy_k1"},
        "mask_classes": ["background", "road", "building"],
        "mask_temporal_direction": False,
        "license": {"code_file": "MIT", "code_readme_restriction": "academic purposes only",
                    "model_card": "apache-2.0"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(root), "checkpoint_sha256": CHECKPOINT_SHA256,
                      "checkpoint_bytes": SIZE, "vocabulary_sha256": manifest["vocabulary"]["sha256"]}))


if __name__ == "__main__":
    main()

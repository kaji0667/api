"""Download the explicitly authorized, revision-pinned independent trial asset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REPOSITORY = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
MODEL_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
MODEL_BYTES = 689359096
FILES = ("config.json", "preprocessor_config.json", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md",
         "model.safetensors")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    def fetch(name: str) -> tuple[str, str]:
        path = destination / name
        if path.exists():
            if name == "model.safetensors" and (path.stat().st_size != MODEL_BYTES
                                                or digest(path) != MODEL_SHA256):
                raise ValueError("Existing independent model differs; refusing overwrite")
            return name, digest(path)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        temporary = path.with_name(path.name + ".partial")
        if temporary.exists():
            raise ValueError("Incomplete download exists; choose a fresh asset directory")
        request = urllib.request.Request(url, headers={"User-Agent": "HFSA-public-grounding-trial/1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as output:
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                output.write(chunk)
        actual = digest(temporary)
        if name == "model.safetensors" and (temporary.stat().st_size != MODEL_BYTES or actual != MODEL_SHA256):
            raise ValueError("Official independent model size/SHA256 mismatch")
        temporary.rename(path)
        print(json.dumps({"downloaded": name, "bytes": path.stat().st_size}), flush=True)
        return name, actual
    with ThreadPoolExecutor(max_workers=3) as pool:
        hashes = dict(pool.map(fetch, FILES))
    provenance = {"repository": REPOSITORY, "revision": REVISION, "license": "Apache-2.0",
                  "model_bytes": MODEL_BYTES, "file_sha256": hashes,
                  "scope": "independent public offline trial; no production dispatch"}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(download(parser.parse_args().output), indent=2), flush=True)

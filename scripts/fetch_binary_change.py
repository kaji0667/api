"""Prepare only the fixed author's DSIFN ChangeFormer release and demo subset.

The original ZIP is retained. Only a reviewed best-checkpoint member is read;
weights_only=True is mandatory and only pure model tensors are saved.
"""

from __future__ import annotations

import argparse
import ast
from io import BytesIO
import hashlib
import json
from pathlib import Path
import pickletools
import urllib.request
import zipfile

CODE_DIR = Path(__file__).resolve().parents[1]
REVISION = "5ca45352d06e15a01e2d6525d891f9c5bbf54f52"
LICENSE_REVISION = "afd1b7ed640aa265a2c730de958416ae7356a2f9"
ARCHITECTURE_ID = "changeformer_v6_dsifn_binary_v1"
ASSET_NAME = ("CD_ChangeFormerV6_DSIFN_b16_lr0.00006_adamw_train_test_200_linear_ce_"
              "multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip")
ASSET_URL = "https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/" + ASSET_NAME
ASSET_SIZE = 985645254
ASSET_SHA256 = "97073ef7e57b23157b73b6d13b67ea78ec277f47dd2fa63da81c99868258241b"
AUTHOR_CHECKPOINT_SHA256 = "df2bc1e3e48980c6d6f5dfd226eaea38a0a950f62f117b0a0f903f2c9b491330"
SAMPLES = ("0_2.png", "1_1.png", "2_4.png", "3_4.png")
SOURCE_FILES = ("models/ChangeFormer.py", "models/ChangeFormerBaseNetworks.py",
                "models/evaluator.py", "models/basic_model.py", "models/help_funcs.py",
                "models/pixel_shuffel_up.py", "datasets/data_utils.py", "datasets/CD_dataset.py",
                "demo_DSIFN.py", "README.md")
NETWORK_NAMES = ("OverlapPatchEmbed", "resize", "Mlp", "Attention", "Block", "DWConv", "MLP",
                 "conv_diff", "make_prediction", "EncoderTransformer_v3",
                 "DecoderTransformer_v3", "ChangeFormerV6")
BASE_NAMES = ("ConvLayer", "UpsampleConvLayer", "ResidualBlock")
SOURCE_HASHES = {
    "models/ChangeFormer.py": "4b5f3f771f590dbec04cafeef8d3aca4b42d3cf50747973dd704700ce2282ff5",
    "models/ChangeFormerBaseNetworks.py": "2f8968d5601036a4d0721f91c7891af3a6f04f77e03f25e0045b5e44ca478bfe",
    "models/evaluator.py": "70028d528cfd3f686ee935e6d29c800cbd9ee0e59d59aef9163aebcb57b6ddff",
    "models/basic_model.py": "ce02192f1103db497670cb244c5d29a7715ad51802523ace74d417c1992ef470",
    "models/help_funcs.py": "7976d9134100b23848b909a9c5ba70ab4ca8f7974b53aec2124abecb225f055d",
    "models/pixel_shuffel_up.py": "73990f8e88028c2516e6bfcb03238bb70bcb38f17e9b350e343ab96fc9a124f6",
    "datasets/data_utils.py": "896b6dc7686b1591fcdc32b861192707157ef88e9ae22a32d3b510ea31fa7ec0",
    "datasets/CD_dataset.py": "a19eaa21a1dd69641bd718b9fbce42c83c27597b21bee34f020b6000b2393428",
    "demo_DSIFN.py": "278204bb044db5826421718461656f17bc121e44d5d54860960ff1b7e9208ec7",
    "README.md": "72afee1deac6c526e0e6fa3c149cfcdd292f481dc3a62e606d5256ac2344f870",
}
PUBLIC_HASHES = {
    "A/0_2.png": "4770c6a69420279de4c91d81699551a80b651ebfcc1c7282af16e52166d4ab97",
    "A/1_1.png": "ccb7509b0af6a49c2aa2baed04de0e26ee9ea25d32f1222a110db6678779a84c",
    "A/2_4.png": "e30565cf204e4de2cbda8cab2202b975bdac523247f2969db1228e104327f20b",
    "A/3_4.png": "33c0b39c1d161aa879ebc736e804c193122611c1c2d14ed34b83705dad7ba8e8",
    "B/0_2.png": "dcce5bda15be11459c37fe98e869816f46a304b2275913a07d03d7759c375b35",
    "B/1_1.png": "6b6d2f03d3f1bfaf5e24919dfd3b6b6787c588663c8b9be32293f2e42fcce3b7",
    "B/2_4.png": "97e2282ed1e51ad23ba67988774a02ca3ac80fe575511ebdada4c78f8da31cfc",
    "B/3_4.png": "c676689bdb464deb0c5cef8d5ce7ac61189a27be7b12d947cde71e742de97ce0",
    "label/0_2.png": "ef3666e623d6b3a5df223845eb1c29cab306693baf393594e89f085f79050d02",
    "label/1_1.png": "3b29106438ab6e1217e273dd902687bd5ce1b0038edd53e33bac40d32018e0df",
    "label/2_4.png": "767fed35581f7bdd0f7d4a3f185ad6754d555ddc1598d7ba62dba99d3b76a73b",
    "label/3_4.png": "9c8710a5419fc9b2090cd73d0c071238590e9978cef2e16171c5cac8da5a7fbd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response, path.open("wb") as output:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            output.write(chunk)


def selected_definitions(path: Path, names: tuple[str, ...]) -> list[str]:
    source = path.read_text(encoding="utf-8")
    definitions = {node.name: node for node in ast.parse(source).body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    if not set(names).issubset(definitions):
        raise ValueError("Pinned author definitions are missing")
    return [ast.get_source_segment(source, definitions[name]) for name in names]


def prepare(bundle: Path, offline: bool = False) -> dict:
    bundle.mkdir(parents=True, exist_ok=True)
    zip_path = bundle / "weights/release.zip"
    if not zip_path.is_file() and not offline:
        download(ASSET_URL, zip_path)
    if zip_path.stat().st_size != ASSET_SIZE:
        raise ValueError("Release ZIP size mismatch")
    zip_sha = sha256(zip_path)
    if zip_sha != ASSET_SHA256:
        raise ValueError("Release ZIP hash mismatch")
    source_root = bundle / "source"
    for relative in SOURCE_FILES:
        path = source_root / relative
        if not path.is_file() and not offline:
            download(f"https://raw.githubusercontent.com/wgcban/ChangeFormer/{REVISION}/{relative}", path)
        if not path.is_file():
            raise FileNotFoundError("Pinned author source file is missing")
        if sha256(path) != SOURCE_HASHES[relative]:
            raise ValueError("Pinned author source integrity mismatch")
    target = CODE_DIR / "tasks/change/_binary"
    target.mkdir(parents=True, exist_ok=True)
    license_path = target / "LICENSE"
    if not license_path.is_file() and not offline:
        download(f"https://raw.githubusercontent.com/wgcban/ChangeFormer/{LICENSE_REVISION}/LICENSE",
                 license_path)
    header = ('"""Necessary ChangeFormer V6 author definitions, unchanged below imports.\n\n'
              f'Source: wgcban/ChangeFormer {REVISION}; see LICENSE and the\n'
              'upstream README research/non-commercial restriction in the bundle.\n'
              'Only unused model variants/imports are excluded; computation is retained.\n"""\n\n'
              'import math\nimport warnings\nfrom functools import partial\n\n'
              'import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n'
              'from timm.models.layers import DropPath, to_2tuple, trunc_normal_\n\n')
    network = "\n\n".join([header,
        *selected_definitions(source_root / "models/ChangeFormerBaseNetworks.py", BASE_NAMES),
        *selected_definitions(source_root / "models/ChangeFormer.py", NETWORK_NAMES)]) + "\n"
    (target / "network.py").write_text(network, encoding="utf-8")
    public_files, public_samples = [], []
    for name in SAMPLES:
        sample = {"id": name.removesuffix(".png")}
        for part, key in (("A", "image_a"), ("B", "image_b"), ("label", "label")):
            relative = f"public/{part}/{name}"
            path = bundle / relative
            if not path.is_file() and not offline:
                download(f"https://raw.githubusercontent.com/wgcban/ChangeFormer/{REVISION}/samples_DSIFN/{part}/{name}", path)
            if sha256(path) != PUBLIC_HASHES[f"{part}/{name}"]:
                raise ValueError("Fixed public sample integrity mismatch")
            public_files.append({"path": relative, "sha256": sha256(path), "size": path.stat().st_size})
            sample[key] = relative
        public_samples.append(sample)
    public_manifest = {"public_only": True, "source_revision": REVISION,
                       "selection_rule": "First four author DSIFN demo filenames in lexical order; fixed before prediction",
                       "ground_truth_role": "Independent scoring only; no labels enter model prediction",
                       "samples": public_samples, "files": public_files}
    (bundle / "public_manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n", encoding="utf-8")
    import torch
    with zipfile.ZipFile(zip_path) as archive:
        members = [item for item in archive.infolist()
                   if Path(item.filename).name == "best_ckpt.pt" and not item.is_dir()]
        if len(members) != 1 or not 0 < members[0].file_size < 600 * 1024 * 1024:
            raise ValueError("Expected exactly one bounded author best checkpoint")
        member = members[0]
        # No extractall or filename-derived path is used; the ZIP member stays
        # in memory and unpickling retains PyTorch's restricted weights loader.
        payload = archive.read(member)
    if hashlib.sha256(payload).hexdigest() != AUTHOR_CHECKPOINT_SHA256:
        raise ValueError("Author checkpoint member hash mismatch")
    with zipfile.ZipFile(BytesIO(payload)) as checkpoint_zip:
        pickle_members = [name for name in checkpoint_zip.namelist() if name.endswith("/data.pkl")]
        if len(pickle_members) != 1:
            raise ValueError("Unexpected author checkpoint serialization")
        pickle_bytes = checkpoint_zip.read(pickle_members[0])
    opcodes = list(pickletools.genops(pickle_bytes))
    globals_found = sorted({argument for opcode, argument, _ in opcodes if opcode.name == "GLOBAL"})
    expected_globals = sorted({"_codecs encode", "collections OrderedDict", "numpy dtype",
                              "numpy.core.multiarray scalar", "torch FloatStorage",
                              "torch LongStorage", "torch._utils _rebuild_tensor_v2"})
    if (globals_found != expected_globals
            or any(opcode.name == "STACK_GLOBAL" for opcode, _, _ in opcodes)
            or not any(opcode.name == "BINUNICODE" and argument == "f8"
                       for opcode, argument, _ in opcodes)):
        raise ValueError("Author checkpoint pickle globals differ from audited numeric metadata")
    # The fixed checkpoint's sole NumPy scalar is best_val_acc (float64).
    # No ndarray reconstruction, external/custom class, or unsafe loader is allowed.
    import numpy as np
    from numpy.core.multiarray import scalar
    float64_dtype_class = type(np.dtype(np.float64))
    scalar_module = scalar.__module__
    try:
        # NumPy 2 renamed core to _core; torch 2.5 allowlists function names
        # without aliases. Scope this metadata-only compatibility alias to the
        # isolated preparation process and retain the same native function.
        scalar.__module__ = "numpy.core.multiarray"
        with torch.serialization.safe_globals([scalar, np.dtype, float64_dtype_class]):
            checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    finally:
        scalar.__module__ = scalar_module
    if type(checkpoint.get("best_val_acc")) is not np.float64:
        raise ValueError("Unexpected author training metadata type")
    state = checkpoint.get("model_G_state_dict") if isinstance(checkpoint, dict) else None
    if (not isinstance(state, dict) or not state
            or not all(isinstance(key, str) and isinstance(value, torch.Tensor)
                       for key, value in state.items())):
        raise ValueError("Author checkpoint does not contain a pure model state")
    weight_path = bundle / "weights/model_state.pt"
    torch.save(state, weight_path)
    reloaded = torch.load(weight_path, map_location="cpu", weights_only=True)
    if (set(reloaded) != set(state)
            or any(not torch.equal(reloaded[name], value) for name, value in state.items())):
        raise ValueError("Pure tensor state reload mismatch")
    source_receipt = [{"path": relative, "sha256": sha256(source_root / relative),
                       "size": (source_root / relative).stat().st_size} for relative in SOURCE_FILES]
    manifest = {"architecture_id": ARCHITECTURE_ID, "source_revision": REVISION,
                "model": {"input_nc": 3, "output_nc": 2, "embed_dim": 256, "decoder_softmax": False},
                "preprocessing": "author_eval_rgb_pil_bicubic_256_to_tensor_norm_0.5",
                "release": {"url": ASSET_URL, "asset_id": 57415938, "size": ASSET_SIZE, "sha256": zip_sha},
                "author_checkpoint_member": {"path": member.filename, "size": member.file_size,
                                             "sha256": hashlib.sha256(payload).hexdigest()},
                "checkpoint": {"path": "weights/model_state.pt", "size": weight_path.stat().st_size,
                               "sha256": sha256(weight_path), "tensor_count": len(state)},
                "source_files": source_receipt,
                "vendored_network_sha256": sha256(target / "network.py"),
                "license_revision": LICENSE_REVISION, "license_sha256": sha256(license_path),
                "weights_only": True,
                "serialization_audit": {"pickle_globals": globals_found, "stack_global_count": 0,
                    "explicit_safe_globals": ["numpy.core.multiarray.scalar", "numpy.dtype",
                                              float64_dtype_class.__module__ + "." + float64_dtype_class.__name__],
                    "numpy_scalar_role": "best_val_acc only, fixed author's float64 metadata",
                    "numpy2_compatibility": "Native scalar __module__ temporarily aliases legacy core; try/finally restores original",
                    "native_scalar_module_restored": scalar.__module__ == scalar_module,
                    "pure_state_reloaded_without_safe_globals": True,
                    "unsafe_pickle_fallback": False}}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=CODE_DIR / "runs/change/changeformer_dsifn")
    parser.add_argument("--offline", action="store_true", help="Only use already downloaded fixed files")
    args = parser.parse_args()
    manifest = prepare(args.bundle_dir.resolve(), args.offline)
    print(json.dumps({"stage": "binary_bundle_prepared", "release": manifest["release"],
                      "checkpoint": manifest["checkpoint"], "weights_only": True}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Portable orchestration of the existing HFSA VQA CLIs.

Use --materials-dir to select the supplied training materials containing
raw/Annotations_train.zip and raw/Annotations_val.zip. Moving this runner into
the code tree does not supply those annotations; it orchestrates task modules.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ASSETS = {
    "pretrain_model/yolov12m.pt": "1b5eb229105a69db0ead76b82fd6edb85673716ed898f1c5b16f77ceba23d5a1",
    "openclip/ViT-L-14.pt": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
}


def path(value):
    value = str(value)
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + value[2:].replace("\\", "/")
    return Path(value).expanduser().resolve()


def digest(file):
    h = hashlib.sha256()
    with Path(file).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(file):
    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(f"{file}: run prepare first")
    rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty JSONL: {file}")
    return rows


class Driver:
    def __init__(self, args):
        self.a = args
        self.code = path(args.code_dir)
        self.dataset = path(args.dataset_dir)
        self.materials = path(args.materials_dir)
        self.weights = path(args.weights_dir or self.materials / "weights")
        self.work = path(args.work_dir or self.materials / "work")
        self.data = self.work / "data"
        self.run = self.work / "runs" / args.run_name
        self.cache = self.work / "cache" / args.run_name / "full"
        self.work.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log = self.work / "logs" / f"{args.stage}_{stamp}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.commands = []

    def execute(self, argv):
        argv = [str(x) for x in argv]
        self.commands.append(argv)
        print("\nRUN " + json.dumps(argv, ensure_ascii=False), flush=True)
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        with self.log.open("ab") as log:
            log.write(("\nRUN " + json.dumps(argv, ensure_ascii=False) + "\n").encode("utf-8"))
            log.flush()
            proc = subprocess.Popen(argv, cwd=self.code, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line.decode("utf-8", errors="replace"), end="", flush=True)
            code = proc.wait()
        if code:
            raise RuntimeError(f"Command failed ({code}); see {self.log}")

    def module(self, name, *args):
        self.execute([sys.executable, "-B", "-m", name, *args])

    def required_code(self):
        if os.name == "nt" and self.dataset.drive.lower() != self.work.drive.lower():
            raise ValueError("Windows native preparation requires dataset and --work-dir on the same drive; "
                             "the existing data CLI writes relative image paths. Use WSL/Linux or select a same-drive --work-dir.")
        required = ["tasks/vqa/train.py", "tasks/vqa/data.py", "tasks/vqa/evaluate.py",
                    "tasks/vqa/inference.py", "tasks/vqa/qformer.py", "tasks/vqa/qformer_core.py",
                    "tasks/classification/model.py", "tasks/classification/data.py",
                    "ultralytics/nn/tasks.py", "ultralytics/cfg/models/v12/yolov12-classification.yaml",
                    "config/vqa/qformer.json", "scripts/finalize_vqa_bundle.py"]
        for rel in required:
            if not (self.code / rel).is_file():
                raise FileNotFoundError(f"Incomplete source: {self.code / rel}")
        if not (self.dataset / "JPEGImages").is_dir():
            raise FileNotFoundError(f"Expected flat image directory: {self.dataset / 'JPEGImages'}")

    def verify_assets(self):
        result = {}
        for rel, expected in ASSETS.items():
            file = self.weights / rel
            if not file.is_file():
                raise FileNotFoundError(f"Missing weight: {file}; extract the weights ZIP or use --weights-dir")
            actual = digest(file)
            if actual != expected:
                raise ValueError(f"Different/corrupted frozen weight: {file}")
            result[rel] = {"sha256": actual, "bytes": file.stat().st_size}
        return result

    def check(self):
        self.required_code()
        assets = self.verify_assets()
        image_index = Counter(p.stem for p in (self.dataset / "JPEGImages").iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        annotations = {}
        for split in ("train", "val"):
            file = self.materials / "raw" / f"Annotations_{split}.zip"
            if not file.is_file():
                raise FileNotFoundError(file)
            valid = qa_count = mismatches = 0
            with zipfile.ZipFile(file) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError(f"Annotation ZIP CRC error: {bad}")
                for name in archive.namelist():
                    if not name.endswith(".json") or name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                        continue
                    obj = json.loads(archive.read(name))
                    image_id = Path(obj.get("image", "")).stem
                    if Path(name).stem != image_id:
                        mismatches += 1
                        continue
                    if image_index[image_id] != 1:
                        raise ValueError(f"Missing or ambiguous image stem: {image_id}")
                    if "qa_pairs" not in obj:
                        raise ValueError(f"Not a QA annotation: {name}")
                    valid += 1
                    qa_count += len(obj["qa_pairs"])
            annotations[split] = {"images": valid, "raw_qa_pairs": qa_count, "skipped_filename_mismatches": mismatches,
                                  "sha256": digest(file)}
        source_changes = []
        reference = self.materials / "SOURCE_REFERENCE.json"
        if reference.is_file():
            for rel, expected in json.loads(reference.read_text(encoding="utf-8")).get("sha256", {}).items():
                file = self.code / rel
                if not file.is_file() or digest(file) != expected:
                    source_changes.append(rel)
        report = {"python": sys.version, "code_dir": str(self.code), "dataset_dir": str(self.dataset),
                  "assets": assets, "annotations": annotations, "source_changes_from_handoff": source_changes,
                  "note": "Image contents/dimensions/XML are checked by prepare, not by this filename inventory."}
        write_json(self.work / "check_files.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        probe = ("import importlib.metadata as md,json,pathlib,torch,torchvision,open_clip,ultralytics; "
                 "from tasks.vqa import train,evaluate,inference; "
                 "from ultralytics.nn.tasks import yaml_model_load; "
                 "c=yaml_model_load('ultralytics/cfg/models/v12/yolov12m-classification.yaml'); "
                 "assert c['scale']=='m'; "
                 "assert pathlib.Path(ultralytics.__file__).resolve().is_relative_to(pathlib.Path.cwd()/'ultralytics'); "
                 "d=" + repr(self.a.device) + "; "
                 "assert d=='cpu' or (d.startswith('cuda:') and torch.cuda.is_available() and "
                 "int(d.split(':')[1])<torch.cuda.device_count()), 'Requested CUDA device unavailable'; "
                 "print(json.dumps({'packages':{k:md.version(k) for k in "
                 "('torch','torchvision','open-clip-torch','numpy','Pillow','PyYAML','opencv-python','huggingface-hub')},"
                 "'local_ultralytics':ultralytics.__file__,'model_scale':c['scale'],'cuda':torch.cuda.is_available()}))")
        self.execute([sys.executable, "-B", "-c", probe])

    def prepare(self):
        self.required_code()
        self.data.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            args = ["--annotations-dir", self.materials / "raw" / f"Annotations_{split}.zip",
                    "--images-dir", self.dataset / "JPEGImages", "--split", split,
                    "--out", self.data / f"{split}_raw.jsonl"]
            if not self.a.no_xml_audit:
                if not (self.dataset / "Annotations").is_dir():
                    raise FileNotFoundError("XML directory absent; see guide before using --no-xml-audit")
                args += ["--xml-dir", self.dataset / "Annotations"]
            self.module("tasks.vqa.data", *args)
        self.module("tasks.vqa.data", "--clean-splits", self.data / "train_raw.jsonl",
                    self.data / "val_raw.jsonl", "--out-dir", self.data)
        print("Prepared train.jsonl / val.jsonl. Read data_summary.json before training.", flush=True)

    def config(self, output, cache, smoke=False):
        source = path(self.a.config) if self.a.config else self.code / "config/vqa/qformer.json"
        c = json.loads(source.read_text(encoding="utf-8-sig"))
        # Inference config/checkpoint bind this string: keep it portable across machines.
        # All child commands run at code CWD; the m alias resolves the common YAML.
        c.update(model_yaml="ultralytics/cfg/models/v12/yolov12m-classification.yaml",
                 visual_backbone=str(self.weights / "pretrain_model/yolov12m.pt"),
                 text_encoder=str(self.weights / "openclip/ViT-L-14.pt"), cache_dir=str(cache))
        if smoke:
            c.update(epochs=1, batch=8, min_answer_count=1, patience=1)
        else:
            if self.a.batch is not None:
                c["batch"] = self.a.batch
            if self.a.epochs is not None:
                c["epochs"] = self.a.epochs
        write_json(output, c)
        return output

    def subset(self, source, target, size):
        import sys
        sys.path.insert(0, str(self.code))
        from tasks.vqa.config import SUPPORTED
        rows = read_rows(source)
        chosen = []
        ids = set()
        for kind in SUPPORTED:
            matches = [r for r in rows if r["question_type"] == kind]
            if not matches:
                raise ValueError(f"Smoke requires every supported type; missing {kind}")
            chosen.append(matches[0]); ids.add(matches[0]["sample_id"])
        for row in rows:
            if len(chosen) >= size:
                break
            if row["sample_id"] not in ids and row["question_type"] in SUPPORTED:
                chosen.append(row); ids.add(row["sample_id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as out:
            for row in chosen:
                row = dict(row)
                image = Path(row["image_path"])
                if not image.is_absolute():
                    image = path(Path(source).parent / image)
                row["image_path"] = os.path.relpath(image, target.parent).replace("\\", "/")
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    def training(self, train, val, config, output, cache):
        self.module("tasks.vqa.train", "--train-jsonl", train, "--val-jsonl", val,
                    "--config", config, "--device", self.a.device, "--save-dir", output, "--cache-dir", cache)

    def evaluation(self, output, val, cache):
        self.module("tasks.vqa.evaluate", "--bundle-dir", output, "--jsonl", val,
                    "--visual-backbone", self.weights / "pretrain_model/yolov12m.pt",
                    "--text-encoder", self.weights / "openclip/ViT-L-14.pt", "--device", self.a.device,
                    "--cache-dir", cache, "--out", output / "evaluation", "--formats", "--ablations")

    def smoke(self):
        self.required_code(); self.verify_assets()
        root = self.work / "smoke" / self.a.run_name
        output = root / "bundle"
        if (output / "manifest.json").exists():
            raise FileExistsError("Existing smoke result; select a different --run-name")
        train, val, cache = root / "data/train.jsonl", root / "data/val.jsonl", root / "cache"
        self.subset(self.data / "train.jsonl", train, 32)
        self.subset(self.data / "val.jsonl", val, 16)
        config = self.config(root / "config.json", cache, smoke=True)
        self.training(train, val, config, output, cache)
        self.evaluation(output, val, cache)
        # Intentionally leave smoke unsealed: its tiny split is not a formal validation.
        self.demo(output, val)
        print("SMOKE PASSED. Tiny-split scores are plumbing checks, not model quality or delivery evidence.", flush=True)

    def train(self):
        self.required_code(); self.verify_assets()
        if (self.run / "manifest.json").exists():
            raise FileExistsError("Existing trained bundle; select a different --run-name. There is no resume mode.")
        config = self.config(self.work / "configs" / f"{self.a.run_name}.json", self.cache)
        self.training(self.data / "train.jsonl", self.data / "val.jsonl", config, self.run, self.cache)

    def eval(self):
        self.evaluation(self.run, self.data / "val.jsonl", self.cache)

    def seal(self):
        self.execute([sys.executable, "-B", self.code / "scripts/finalize_vqa_bundle.py", "--bundle-dir", self.run])

    def demo(self, output=None, val=None):
        output = output or self.run
        val = val or self.data / "val.jsonl"
        train = Path(val).with_name("train.jsonl")
        request = dict(bundle=str(output), val=str(val), train=str(train), device=self.a.device,
                       visual=str(self.weights / "pretrain_model/yolov12m.pt"),
                       text=str(self.weights / "openclip/ViT-L-14.pt"))
        # Use the actual scope guard and one reusable predictor. Selecting a public
        # training fallback is explicitly recorded and never affects evaluation metrics.
        program = '''import json,sys
from pathlib import Path
import torch
from tasks.vqa.data import read_rows
from tasks.vqa.config import SUPPORTED,write_json
from tasks.vqa.inference import VQAPredictor,scope_guard
torch.set_num_threads(4)
a=json.loads(sys.argv[1]);root=Path(a['bundle'])
labels=json.loads((root/'config/labels.json').read_text(encoding='utf-8'))
selected=None;origin=None
for file,source in ((a['val'],'public_validation'),(a['train'],'public_training_fallback')):
    if not Path(file).is_file():continue
    candidates=[r for r in read_rows(file) if r['question_type'] in SUPPORTED
                and r['normalized_answer'] in labels['answers'][r['question_type']]
                and not scope_guard(r['question'])]
    if candidates:
        selected=next((r for r in candidates if r['question_type']=='color'),candidates[0]);origin=source;break
if selected is None:raise ValueError('No public in-scope known-answer demo row')
r=selected;values=[r['normalized_answer']]+[x for x in labels['answers'][r['question_type']] if x!=r['normalized_answer']][:3]
choices={chr(65+i):x for i,x in enumerate(values)}
write_json(root/'demo/public_input.json',dict(image=r['image_path'],question=r['question'],choices=choices,source=origin,
           note='PUBLIC QA synthetic format example; gold-first options; NOT an accuracy benchmark.'))
p=VQAPredictor(root,a['device'],dict(visual_backbone=a['visual'],text_encoder=a['text']))
results={}
for mode,options in (('short_text',None),('single_choice',choices)):
    results[mode]=p.predict(r['image_path'],r['question'],choices=options)
    print(mode+': '+json.dumps(results[mode],ensure_ascii=False),flush=True)
write_json(root/'demo/results.json',dict(source=origin,predictions=results,note='Format/load check; predictions may be wrong or unsupported.'))
'''
        self.execute([sys.executable, "-B", "-c", program, json.dumps(request, ensure_ascii=False)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "prepare", "smoke", "train", "eval", "seal", "demo"))
    parser.add_argument("--code-dir", required=True, help="Already supplied HFSA-main directory")
    parser.add_argument("--dataset-dir", required=True, help="Existing VRSBench root, containing JPEGImages and Annotations")
    parser.add_argument("--materials-dir", default=str(Path(__file__).resolve().parent),
                        help="Supplied training materials directory containing raw/Annotations_train.zip and raw/Annotations_val.zip; pass explicitly when annotations are outside the code tree")
    parser.add_argument("--weights-dir", help="Root containing pretrain_model/ and openclip/")
    parser.add_argument("--work-dir", help="New local data/cache/runs/logs root; default materials/work")
    parser.add_argument("--run-name", default="qformer_team", help="Distinct experiment name; letters/digits/_/- only")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (recommended) or cpu; train and full eval must use the same device")
    parser.add_argument("--batch", type=int, help="Formal training batch override")
    parser.add_argument("--epochs", type=int, help="Formal training epoch override")
    parser.add_argument("--config", help="Alternative task configuration JSON; frozen assets/paths are supplied by this driver")
    parser.add_argument("--no-xml-audit", action="store_true", help="Skip optional XML comparison, retaining actual QA/image/hash audits")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name must contain only letters, digits, underscore or hyphen")
    if args.batch is not None and args.batch <= 0 or args.epochs is not None and args.epochs <= 0:
        parser.error("batch/epochs must be positive")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    driver = Driver(args)
    try:
        getattr(driver, args.stage)()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        write_json(driver.log.with_suffix(".commands.json"), {"cwd": str(driver.code), "python": sys.executable,
                   "stage": args.stage, "commands": driver.commands})
    print(f"DONE: {args.stage}; logs: {driver.log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

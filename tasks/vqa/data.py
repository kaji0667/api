"""Restore original VRSBench QA, audit image association, and build train-only labels."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image
from .config import SUPPORTED, TYPES, resolve_path, write_json, sha256

_WORD2NUM = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
    "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15",
    "sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19","twenty":"20",
}

def _singularize(word):
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        if word.endswith(("ous","is","us")):
            return word
        if word[-3:] in ("ses","xes","zes","ches","shes"):
            return word[:-2]
        return word[:-1]
    return word

def normalize_answer(answer: str) -> str:
    s = " ".join(str(answer).lower().strip().split()).strip(".?!")
    s = re.sub(r"\bgrey\b", "gray", s)
    s = re.sub(r"^the\s+", "", s)
    s = s.replace("-", " ")
    for word, num in _WORD2NUM.items():
        s = re.sub(rf"\b{word}\b", num, s)
    s = re.sub(r"\bsemi\s+", "semi", s)
    words = s.split()
    words = [_singularize(w) for w in words]
    return " ".join(words)

def comparison_intent(question: str) -> bool:
    """Distinguish a requested comparison from an object-size qualifier."""
    q = question.lower().strip()
    if re.match(r"(?:what|which)\b.{0,30}\b(?:colou?r|shape|position|direction)\b|where\b", q):
        return False
    return bool(re.search(r"\b(?:compared|compare|similar size|same size)\b", q)
                or re.search(r"\b(?:bigger|larger|smaller)\b.{0,50}\bthan\b", q)
                or re.match(r"which\b.*\b(?:bigger|larger|smaller)\s*[?.!]*$", q)
                or re.search(r"\b(?:is|are)\b.*\b(?:bigger|larger|smaller)\s*[?.!]*$", q))

def question_type(raw: str, question: str) -> str:
    raw = raw.lower().strip()
    mapping = {"object category":"category", "object color":"color", "object shape":"shape",
               "object existence":"existence", "scene type":"scene", "rural or urban":"scene",
               "object quantity":"quantity", "object position":"position", "object size":"size",
               "object direction":"direction"}
    q = question.lower()
    # Scope distinctions preserve questions; no target extraction or answer inference.
    if comparison_intent(question):
        return "comparison"
    if raw == "reasoning":
        return "relation"
    if raw == "object existence" and re.match(r"\s*(what|which)\b",q):
        if re.search(r"\bcolou?r\b",q):return "color"
        if re.search(r"\bshape\b",q):return "shape"
        return "category"
    if raw in mapping:
        return mapping[raw]
    if raw == "image":
        if re.search(r"\b(colou?r|grayscale)\b", q): return "color"
        if re.search(r"\b(urban|rural|scene|landscape|environment|terrain|land use)\b", q): return "scene"
    return "other"

def read_rows(path):
    path = resolve_path(path)
    rows = []
    resolved_images = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip(): continue
            row = json.loads(line)
            for key in ("sample_id", "image_id", "image_path", "question", "normalized_answer", "question_type"):
                if key not in row: raise ValueError(f"Missing {key} at line {n}")
            if not isinstance(row["question"], str) or not row["question"].strip(): raise ValueError("Empty question")
            if row["question_type"] not in TYPES: raise ValueError("Unknown type")
            row["normalized_answer"] = normalize_answer(row.get("answer", row["normalized_answer"]))
            raw_path = row["image_path"]
            if raw_path not in resolved_images:
                image_path=raw_path
                if os.name != "nt" and len(image_path)>2 and image_path[1]==":":
                    image_path="/mnt/"+image_path[0].lower()+image_path[2:].replace("\\","/")
                p=Path(image_path).expanduser()
                # Normalize against the already-resolved JSONL directory; actual decode/hash
                # checks happen in prepare/cache. Avoid 30k p9 symlink walks on /mnt/d.
                resolved_images[raw_path]=os.path.abspath(p if p.is_absolute() else path.parent/p)
            row["image_path"] = resolved_images[raw_path]
            rows.append(row)
    if not rows or len({r["sample_id"] for r in rows}) != len(rows): raise ValueError("Empty/duplicate QA dataset")
    return rows

def build_labels(rows, min_count=2):
    counters = {t: Counter(r["normalized_answer"] for r in rows if r["question_type"] == t) for t in SUPPORTED}
    answers = {t: sorted(a for a, n in c.items() if n >= min_count and a) for t,c in counters.items()}
    if any(not v for v in answers.values()): raise ValueError("All supported branches require training answers")
    aliases = {t: {} for t in SUPPORTED}
    for row in rows:
        t = row["question_type"]
        if t in aliases and row["normalized_answer"] in answers[t]:
            aliases[t][normalize_answer(row["answer"])] = row["normalized_answer"]
    return dict(types=list(TYPES), supported=list(SUPPORTED), answers=answers, aliases=aliases,
                counts={t:dict(c) for t,c in counters.items()}, min_answer_count=min_count,
                normalization="lowercase/whitespace/trailing .?!/grey->gray; strip leading 'the'; hyphen->space; word->digit(0-20); semi- merge; conservative singularize; multi-color preserved")

def assert_split_isolation(train, val):
    evidence = {}
    for field in ("image_id", "parent_image_id", "image_sha256", "pixel_sha256"):
        a = {r.get(field) for r in train if r.get(field)}
        b = {r.get(field) for r in val if r.get(field)}
        evidence[field] = dict(train_groups=len(a), val_groups=len(b), overlap=len(a & b))
        if a & b: raise ValueError(f"Split leakage: {field}, {len(a & b)} groups")
    return evidence

def clean_splits(train_path,val_path,train_out,val_out,report_path):
    """Keep raw prepared artifacts, repair question-type metadata and isolate identical content."""
    datasets=[];retyped=Counter()
    for p in (train_path,val_path):
        rs=[json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]
        for r in rs:
            t=question_type(r["original_type"],r["question"])
            if t!=r["question_type"]:retyped[f'{r["split"]}:{r["question_type"]}->{t}']+=1
            r["question_type"]=t
        datasets.append(rs)
    train,val=datasets;excluded=[]
    train_pixels={r["pixel_sha256"] for r in train};train_files={r["image_sha256"] for r in train}
    clean_val=[]
    for r in val:
        if r["pixel_sha256"] in train_pixels or r["image_sha256"] in train_files:
            excluded.append(dict(sample_id=r["sample_id"],image_id=r["image_id"],reason="identical_image_content_in_train"))
        else:clean_val.append(r)
    duplicate_qa=[];deduplicated=[]
    for rs in (train,clean_val):
        groups=defaultdict(list)
        for r in rs:groups[(r["image_id"],r["question"])].append(r)
        kept=[]
        for group in groups.values():
            if len({r["normalized_answer"] for r in group})>1:
                duplicate_qa.extend(dict(sample_id=r["sample_id"],reason="conflicting_same_image_question") for r in group)
            else:
                kept.append(group[0])
                duplicate_qa.extend(dict(sample_id=r["sample_id"],reason="duplicate_same_image_question_answer") for r in group[1:])
        deduplicated.append(kept)
    train,clean_val=deduplicated
    isolation=assert_split_isolation(train,clean_val)
    for rs,src,out in [(train,train_path,train_out),(clean_val,val_path,val_out)]:
        mapped_paths={}
        with Path(out).open("w",encoding="utf-8") as f:
            for r in rs:
                raw_path=r["image_path"]
                if raw_path not in mapped_paths:mapped_paths[raw_path]=os.path.relpath(resolve_path(raw_path,Path(src).parent),Path(out).parent).replace("\\","/")
                r["image_path"]=mapped_paths[raw_path]
                f.write(json.dumps(r,ensure_ascii=False)+"\n")
    report=dict(train_qa=len(train),raw_val_qa=len(val),val_qa=len(clean_val),excluded_validation=excluded,
                retyped=dict(retyped),split_isolation=isolation,duplicate_qa=duplicate_qa,
                final_type_counts={name:dict(Counter(r["question_type"] for r in rs)) for name,rs in [("train",train),("val",clean_val)]},
                annotation_quarantine=[dict(file="02726_0000.json",image_field="02725_0000.png",qa_count=3),dict(file="P2708_0017.json",image_field="P2708_0016.png",qa_count=6)],
                scope="inferred public QA types; complete question/answer untouched; no object association or synthetic negative labels",
                source_count_note="Downloaded archives contain 85813 train and 37409 val QA. One duplicated val image/question/answer explains the extra QA relative to README 123221; 9 train QA with mismatched image fields are separately quarantined.")
    write_json(report_path,report);print(json.dumps(report,ensure_ascii=False),flush=True)
    return report

def prepare(annotations_dir, images_dir, out, split, xml_dir=None):
    source = resolve_path(annotations_dir); images = resolve_path(images_dir); out = resolve_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skipped = Counter(); rows = []; audit = []; types = Counter(); raw_types = Counter(); ids = set()
    image_index = defaultdict(list)
    for p in images.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"): image_index[p.stem].append(p)
    archive = zipfile.ZipFile(source) if source.is_file() else None
    files = sorted(n for n in archive.namelist() if n.endswith(".json") and not n.startswith("__MACOSX/") and not Path(n).name.startswith("._")) if archive else sorted(source.rglob("*.json"))
    for n, name in enumerate(files):
        d = json.loads(archive.read(name) if archive else Path(name).read_bytes())
        original = d.get("image")
        if not original: skipped["missing_image_field"] += 1; continue
        image_id = Path(original).stem
        if Path(name).stem != image_id:
            skipped["annotation_filename_image_field_mismatch"] += 1
            continue
        if image_id in ids: raise ValueError("Duplicate image annotation")
        ids.add(image_id)
        candidates = image_index[image_id]
        if len(candidates) != 1: raise ValueError(f"Image missing or ambiguous: {image_id}")
        p = candidates[0]
        with Image.open(p) as im:
            im.load(); rgb = im.convert("RGB"); w,h = rgb.size
            pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        digest = sha256(p)
        xml_match = None
        if xml_dir:
            xp = resolve_path(xml_dir) / (image_id+".xml")
            if not xp.is_file(): raise FileNotFoundError(xp)
            x = ET.parse(xp).getroot()
            if x.findtext("filename") != p.name or (int(x.findtext("size/width")),int(x.findtext("size/height"))) != (w,h):
                raise ValueError(f"XML image dimension mismatch: {image_id}")
            xml_desc = [o.findtext("description") for o in x.findall("object")]
            json_desc = [o.get("referring_sentence") for o in d.get("objects",[])]
            xml_match = xml_desc == json_desc
            if not xml_match: raise ValueError(f"Original JSON vs local XML descriptions mismatch: {image_id}")
        parent = image_id.split("_")[0]
        audit.append(dict(image_id=image_id, original_image=original, local_image=p.name, width=w,height=h,
                          image_sha256=digest,pixel_sha256=pixel_hash,parent_image_id=parent,xml_descriptions_match=xml_match))
        for qi,qa in enumerate(d.get("qa_pairs", [])):
            q,a = qa.get("question"),qa.get("answer")
            if not isinstance(q,str) or not q.strip() or not isinstance(a,str) or not a.strip():
                skipped["empty_or_nontext_qa"] += 1; continue
            t = question_type(str(qa.get("type","")),q); types[t]+=1; raw_types[str(qa.get("type"))]+=1
            rows.append(dict(sample_id=f"{split}:{image_id}:{qi}",image_id=image_id,
                             image_path=os.path.relpath(p,out.parent).replace("\\","/"),question=q,answer=a,
                             normalized_answer=normalize_answer(a),question_type=t,original_type=qa.get("type"),
                             split=split,source="VRSBench",parent_image_id=parent,width=w,height=h,
                             image_sha256=digest,pixel_sha256=pixel_hash,target_boxes=[],reference_boxes=[]))
        if (n+1)%1000 == 0: print(f"{split}: audited {n+1}/{len(files)} images",flush=True)
    if archive: archive.close()
    with out.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    summary=dict(source="xiang709/VRSBench original qa_pairs",source_path=str(source),
                 source_sha256=sha256(source) if source.is_file() else None,annotation_images=len(ids),
                 qa_count=len(rows),types=dict(types),original_types=dict(raw_types),discarded=dict(skipped),
                 png_jpg_mapping="exact image stem, unique local extension, actual decode/dimensions/hash and original JSON/local XML descriptions",
                 parent_group_rule="prefix before first underscore; original source-image grouping inferred from names",
                 pixel_content_verification="local decoded hashes and XML identity; remote PNG sample checks recorded separately",
                 source_geolocation_isolation="geolocation/site identity unavailable; parent grouping is not proof of geographical disjointness")
    write_json(out.with_suffix(".summary.json"),summary)
    with out.with_suffix(".images.jsonl").open("w",encoding="utf-8") as f:
        for row in audit: f.write(json.dumps(row)+"\n")
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations-dir",help="Original annotation directory OR downloaded ZIP")
    p.add_argument("--images-dir");p.add_argument("--out")
    p.add_argument("--clean-splits",nargs=2,metavar=("TRAIN_RAW","VAL_RAW"));p.add_argument("--out-dir")
    p.add_argument("--split",choices=("train","val"),default="train");p.add_argument("--xml-dir")
    a=p.parse_args()
    if a.clean_splits:
        if not a.out_dir:p.error("--out-dir required for --clean-splits")
        out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
        clean_splits(*a.clean_splits,out/"train.jsonl",out/"val.jsonl",out/"data_summary.json")
    else:
        if not all((a.annotations_dir,a.images_dir,a.out)):p.error("--annotations-dir, --images-dir and --out required")
        prepare(a.annotations_dir,a.images_dir,a.out,a.split,a.xml_dir)
if __name__ == "__main__": main()

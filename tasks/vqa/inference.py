"""Local checkpoint-owning VQA interface; no API, Web, Ollama or training-data lookups."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import threading
import torch
from .config import ARCHITECTURE_IDS, FORMAT_VERSION, SUPPORTED, architecture_id, read_json, resolve_device, resolve_path, sha256
from .data import normalize_answer,comparison_intent
from .model import FrozenText, FrozenVisual, VQAHead

def validate_modes(choices, allowed_answers):
    if choices is not None and allowed_answers is not None: raise ValueError("choices and allowed_answers are exclusive")
    if choices is not None:
        if not isinstance(choices,dict) or not choices or any(not isinstance(k,str) or not k.strip() or not isinstance(v,str) or not v.strip() for k,v in choices.items()):
            raise ValueError("choices must be a nonempty string-key/string-value mapping")
    if allowed_answers is not None:
        if not isinstance(allowed_answers,list) or not allowed_answers or any(not isinstance(v,str) or not v.strip() for v in allowed_answers) or len(set(allowed_answers)) != len(allowed_answers):
            raise ValueError("allowed_answers must be a nonempty unique string list")

def scope_guard(question):
    q=question.lower()
    if re.search(r"[\u4e00-\u9fff]",q):return "other","unsupported_language"
    if re.search(r"\b(how many|number of|count the|count all)\b",q):return "quantity","unsupported_task"
    if comparison_intent(question):return "comparison","unsupported_task"
    if re.search(r"\b(where|which direction|what direction|what position|which position)\b",q):return "position","unsupported_task"
    if re.search(r"\b(before and after|two images|second image|what has changed)\b",q):return "other","unsupported_task"
    return None

def failure(t,reason,status="unsupported"):
    return dict(status=status,answer=None,answer_text=None,question_type=t,reason=reason)

def decode_prediction(type_logits,answer_logits,labels,question,choices=None,allowed_answers=None,image_size=None):
    validate_modes(choices,allowed_answers)
    guarded=scope_guard(question)
    if guarded:return failure(*guarded)
    if not torch.isfinite(type_logits).all() or any(not torch.isfinite(v).all() for v in answer_logits.values()):
        return failure(None,"nonfinite_prediction","error")
    t=labels["types"][int(type_logits.argmax())]
    if t not in SUPPORTED:return failure(t,"unsupported_task")
    scores=answer_logits[t].softmax(-1)
    names=labels["answers"][t]
    choice_coverage=None
    if choices is not None or allowed_answers is not None:
        options=list(choices.items()) if choices is not None else [(v,v) for v in allowed_answers]
        mapped=[];unmatched=[]
        for key,value in options:
            n=normalize_answer(value);n=labels["aliases"].get(t,{}).get(n,n)
            if n not in names:
                # Single-choice scoring can use the trained subset. An unknown
                # option has no learned score and must never become a fallback.
                if choices is None:return failure(t,"choice_unmatched")
                unmatched.append(key)
                continue
            mapped.append((key,n,names.index(n)))
        if not mapped:return failure(t,"choice_unmatched")
        if len({n for _,n,_ in mapped}) != len(mapped):raise ValueError("Ambiguous normalized choices")
        if unmatched:
            choice_coverage=dict(provided=len(options),matched=len(mapped),unmatched_keys=unmatched)
        key,text,ix=max(mapped,key=lambda z:float(scores[z[2]]))
        answer=key
    else:
        ix=int(scores.argmax());text=names[ix];answer=text
    w,h=image_size or (0,0)
    result=dict(status="ok",answer=answer,answer_text=text,question_type=t,confidence=float(scores[ix]),
                evidence=dict(image_size=dict(width=w,height=h),coordinate_system="pixel_xyxy",target_boxes=[],reference_boxes=[]))
    if choice_coverage is not None:result["choice_coverage"]=choice_coverage
    return result

def load_bundle(bundle_dir,asset_paths):
    root=resolve_path(bundle_dir);manifest=read_json(root/"manifest.json")
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("architecture_id") not in ARCHITECTURE_IDS:
        raise ValueError("Incompatible VQA bundle architecture/version")
    for rel,digest in manifest["file_sha256"].items():
        p=resolve_path(rel,root)
        if not p.is_relative_to(root):raise ValueError("Unsafe bundle file path")
        if not p.is_file() or sha256(p)!=digest:raise ValueError(f"Missing/mismatched bundle file: {rel}")
    if not isinstance(asset_paths,dict):raise ValueError("asset_paths is required")
    paths={}
    for a in manifest["external_assets"]:
        key=a["asset_id"]
        if key not in asset_paths:raise ValueError(f"Missing asset path: {key}")
        p=resolve_path(asset_paths[key])
        if not p.is_file():raise FileNotFoundError(p)
        if sha256(p) != a["sha256"]:raise ValueError(f"External asset hash mismatch: {key}")
        paths[key]=p
    config=read_json(root/manifest["configs"]["inference"])
    if manifest["architecture_id"] != architecture_id(config):
        raise ValueError("VQA architecture does not match fusion configuration")
    labels=read_json(root/manifest["configs"]["labels"])
    payload=torch.load(root/manifest["checkpoint"]["path"],map_location="cpu",weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("architecture_id") != manifest["architecture_id"] or payload.get("labels") != labels or payload.get("config") != config:
        raise ValueError("Checkpoint metadata/config/labels mismatch")
    head=VQAHead(config,labels);head.load_state_dict(payload["state_dict"],strict=True);head.eval()
    return manifest,config,labels,head,paths

class VQAPredictor:
    def __init__(self,bundle_dir,device="auto",asset_paths=None):
        self.device=resolve_device(device)
        self.manifest,self.config,self.labels,self.head,paths=load_bundle(bundle_dir,asset_paths)
        self.head.to(self.device)
        self.visual=FrozenVisual(paths["visual_backbone"],self.config,self.device)
        self.text=FrozenText(paths["text_encoder"],self.device)
        self.lock=threading.Lock()

    @torch.inference_mode()
    def predict(self,image,question,choices=None,allowed_answers=None):
        if not isinstance(question,str) or not question.strip():raise ValueError("question must be nonempty string")
        validate_modes(choices,allowed_answers)
        if not isinstance(image,(str,Path)):
            from PIL import Image
            if not isinstance(image,Image.Image):raise ValueError("image must be PIL, str or Path")
        with self.lock:
            # Decode invalid/missing images explicitly even for an unsupported question.
            x,size=self.visual.image_tensor(image)
            guarded=scope_guard(question)
            if guarded:return failure(*guarded)
            try:
                q=self.text([question]);v=[f.half().float() for f in self.visual(x[None].to(self.device))]
                types,answers=self.head(v,q,self.config.get("train_ablation"))
                return decode_prediction(types[0],{t:a[0] for t,a in answers.items()},self.labels,question,choices,allowed_answers,size)
            except ValueError as e:
                if "budget" in str(e):return failure("other","text_context_exceeded")
                raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("bundle-dir","visual-backbone","text-encoder","image","question"):p.add_argument("--"+n,required=True)
    p.add_argument("--device",default="auto");p.add_argument("--choices-json");p.add_argument("--allowed-answers-json")
    a=p.parse_args();predictor=VQAPredictor(a.bundle_dir,a.device,dict(visual_backbone=a.visual_backbone,text_encoder=a.text_encoder))
    r=predictor.predict(a.image,a.question,json.loads(a.choices_json) if a.choices_json else None,json.loads(a.allowed_answers_json) if a.allowed_answers_json else None)
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__ == "__main__":main()

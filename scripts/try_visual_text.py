"""Same candidates/full phrase maps; source-bound visual crop text ablation."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
sys.path.insert(0,str(CODE/"participant_api_starter-main"))
from scripts.try_crop_candidate_thresholds import load_inputs,ASSETS
from scripts.try_crop_identity import sha256,write_new_json
from tasks.detection.text_conditioning import parse_grounding_text,visual_description
from tasks.detection.reference_plan import parse_reference_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest115","manifest80","data-root","baseline","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    members,manifests=load_inputs(args.manifest115,args.manifest80,args.data_root)
    old={r["row_key"]:r for r in map(json.loads,args.baseline.read_text().splitlines()) if r["candidate_confidence"]==.25}
    if set(old)!={r["row_key"] for r in members}:
        raise ValueError("Baseline denominator differs")
    args.output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),CODE/"tasks/detection/text_conditioning.py",CODE/"tasks/detection/reference_plan.py",
           CODE/"tasks/detection/crop_identity.py",CODE/"participant_api_starter-main/native_detection.py",
           args.baseline,CODE/"openclip/ViT-L-14.pt"]
    hashes={str(f):sha256(f) for f in files}
    if hashes[str(CODE/"openclip/ViT-L-14.pt")] != ASSETS["clip"][1]:
        raise ValueError("Original CLIP differs")
    images={r["image"]:sha256(Path(r["image"])) for r in members}
    write_new_json(args.output/"frozen.json",{"protocol":"same_pool_visual_text_v1",
        "registered_utc":datetime.now(timezone.utc).isoformat(),"manifests":manifests,"count":len(members),
        "sources_assets_baseline":hashes,"images":images,"device":args.device,
        "configuration":"original adjusted score + .15 minmax target-visual minus reference-visual crop score; unchanged text uses G",
        "prediction_reads_gold":False})
    import cv2
    import numpy as np
    import torch
    from native_detection import NativeTextEncoder
    from tasks.detection.crop_identity import FrozenCLIPCropScorer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started=time.perf_counter()
    encoder=NativeTextEncoder(args.device,pretrained=str(CODE/"openclip/ViT-L-14.pt"))
    scorer=FrozenCLIPCropScorer(encoder.encoder.model,encoder.encode_descriptions)
    changed=0
    with (args.output/"predictions.jsonl").open("x") as out:
        for index,row in enumerate(members):
            prior=old[row["row_key"]]
            plan=parse_grounding_text(row["description"])
            legacy=parse_reference_plan(row["description"])
            target=visual_description(plan.target_description) if plan else row["description"]
            refs=[visual_description(r.description) for r in plan.references] if plan else []
            old_target=legacy.target_description if legacy else row["description"]
            old_refs=[r.description for r in legacy.references] if legacy else []
            modified=plan is not None and (target!=old_target or refs!=old_refs)
            baseline=prior["arms"]["G"]["selected_index"]
            selected,records=baseline,[]
            if modified and prior["candidate_boxes"]:
                image=cv2.imread(row["image"])
                records=scorer.score(image,prior["candidate_boxes"],target,refs)["records"]
                if all(r["status"]=="succeeded" for r in records):
                    contrast=np.array([r["contrast_score"] for r in records])
                    span=contrast.max()-contrast.min()
                    adjustment=(contrast-contrast.min())/span if span>1e-12 else np.zeros_like(contrast)
                    scores=np.asarray(prior["baseline_adjusted_scores"])+.15*adjustment
                    tied=np.flatnonzero(scores.max()-scores<=1e-12).tolist()
                    selected=baseline if baseline in tied else tied[0]
            changed+=selected!=baseline
            record={"cohort":row["cohort"],"id":row["id"],"modified":modified,"target":target,"references":refs,
                "crop_scores":records,"baseline_index":baseline,"selected_index":selected,"arms":{
                    "A":prior["arms"]["A"]["selected_box_xyxy"],"G":prior["arms"]["G"]["selected_box_xyxy"],
                    "visual_text":prior["candidate_boxes"][selected] if selected is not None else None}}
            out.write(json.dumps(record,allow_nan=False)+"\n")
            out.flush()
            if modified:
                print(json.dumps({"progress":index+1,"modified":True,"selected_changed":selected!=baseline}),flush=True)
    if any(sha256(Path(path))!=expected for path,expected in {**hashes,**images}.items()):
        raise ValueError("Registered file changed")
    write_new_json(args.output/"completion.json",{"completed":True,"count":len(members),"changed":changed,
        "prediction_sha256":sha256(args.output/"predictions.jsonl"),"frozen_sha256":sha256(args.output/"frozen.json"),
        "seconds":time.perf_counter()-started})


if __name__=="__main__":
    main()

"""Real public image check: bare/new wrapped/old wrapped, then separate scoring."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
sys.path.insert(0,str(CODE/"participant_api_starter-main"))
from scripts.try_crop_candidate_thresholds import load_inputs,ASSETS
from scripts.try_crop_identity import sha256,write_new_json
DIRECTIVES=("Return a bbox.","Return the bounding box as [x1,y1,x2,y2].",
            "Please provide the bounding box in pixel coordinates.","Report box coordinates only.",
            "Your response must be a bbox.","请只返回坐标，不要包含解释。")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest115","manifest80","data-root","old-module","output"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    members,manifests=load_inputs(args.manifest115,args.manifest80,args.data_root)
    selected=[row for cohort in ("public115","public80")
              for row in [r for r in members if r["cohort"]==cohort][:6]]
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),args.old_module,CODE/"participant_api_starter-main/api_task_text.py",
             CODE/"participant_api_starter-main/native_detection.py",CODE/"tasks/detection/bbox_grounding.py",
             CODE/"tasks/detection/crop_identity.py",CODE/"tasks/detection/reference_plan.py"]
    for relative,expected in ASSETS.values():
        asset=CODE/relative
        if sha256(asset)!=expected:
            raise ValueError("Original asset changed")
        sources.append(asset)
    hashes={str(path):sha256(path) for path in sources}
    images={r["image"]:sha256(Path(r["image"])) for r in selected}
    write_new_json(args.output/"frozen.json",{"protocol":"public_grounding_wrap_parity_v1",
        "registered_utc":datetime.now(timezone.utc).isoformat(),"manifests":manifests,"count":12,
        "selection":"first 6 frozen members in each cohort; 6 formatting directives in fixed order",
        "sources_assets_baseline":hashes,"images":images,"prediction_reads_gold":False})
    old=types.ModuleType("old_grounding_task_text")
    exec(compile(args.old_module.read_text(encoding="utf-8-sig"),str(args.old_module),"exec"),old.__dict__)
    import numpy as np
    import torch
    import native_detection
    from native_detection import NativeDetection,NativeTextEncoder
    from api_task_text import detection_text
    from tasks.detection.bbox_grounding import BBoxGroundingExecutor
    from tasks.detection.crop_identity import FrozenCLIPCropScorer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started=time.perf_counter()
    encoder=NativeTextEncoder("cpu",pretrained=str(CODE/"openclip/ViT-L-14.pt"))
    native=NativeDetection(CODE/"runs/detect/DIOR-RSVG-ViT-L-14/weights/best.pt","cpu",
                          encoder=encoder,max_image_size=1600,spatial_weight=.15)
    scorer=FrozenCLIPCropScorer(encoder.encoder.model,encoder.encode_descriptions)
    def executor(extract):
        return BBoxGroundingExecutor(native,scorer,description_extractor=extract,image_converter=native._image)
    fresh,previous=executor(detection_text),executor(old.detection_text)
    def box(result):
        index=result.grounding_index
        return result.boxes[index].tolist() if index is not None else None
    exact=0
    with (args.output/"predictions.jsonl").open("x") as out:
        for index,row in enumerate(selected):
            question="Locate the described object.\nDescription:\n"+row["description"]+"\n"+DIRECTIVES[index%6]
            bare=fresh.predict(row["image"],row["description"])
            wrapped=fresh.predict(row["image"],question)
            passed=(detection_text(question)==row["description"] and bare.model_prompt==wrapped.model_prompt
                and bare.grounding_index==wrapped.grounding_index and np.array_equal(bare.boxes,wrapped.boxes)
                and np.array_equal(bare.scores,wrapped.scores)
                and np.array_equal(bare.phrase_scores,wrapped.phrase_scores)
                and np.array_equal(bare.baseline_adjusted_scores,wrapped.baseline_adjusted_scores))
            if not passed:
                raise ValueError("Bare/wrapped input or actual outputs differ")
            exact+=1
            old_box,old_reason=None,"succeeded"
            try:
                with patch.object(native_detection,"detection_text",old.detection_text):
                    before=previous.predict(row["image"],question)
                old_box=box(before)
            except Exception as error:
                old_reason=getattr(error,"code",type(error).__name__)
            record={"cohort":row["cohort"],"id":row["id"],"bare_wrapped_exact":passed,
                    "old_reason":old_reason,"old_extraction_contaminated":old.detection_text(question)!=row["description"],
                    "arms":{"G":box(bare),"new_wrapped":box(wrapped),"old_wrapped":old_box}}
            out.write(json.dumps(record,allow_nan=False)+"\n")
            out.flush()
            print(json.dumps({"progress":index+1,"exact":passed,"old_reason":old_reason}),flush=True)
    if any(sha256(Path(path))!=expected for path,expected in {**hashes,**images}.items()):
        raise ValueError("Registered file changed")
    write_new_json(args.output/"completion.json",{"completed":True,"count":12,"exact":exact,
        "prediction_sha256":sha256(args.output/"predictions.jsonl"),"frozen_sha256":sha256(args.output/"frozen.json"),
        "seconds":time.perf_counter()-started})


if __name__=="__main__":
    main()

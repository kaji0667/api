"""No-gold agreement arm for a complete, SHA-verified text trial."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from scripts.try_crop_identity import sha256,write_new_json
from tasks.detection.text_consensus import consensus_selection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    receipt=json.loads((args.trial/"completion.json").read_text())
    if receipt.get("completed") is not True or sha256(args.trial/"predictions.jsonl") != receipt["prediction_sha256"]:
        raise ValueError("Incomplete or changed trial")
    old={r["row_key"]:r for r in map(json.loads,args.baseline.read_text().splitlines()) if r["candidate_confidence"]==.25}
    rows=list(map(json.loads,(args.trial/"predictions.jsonl").read_text().splitlines()))
    if len(rows)!=receipt["count"]:
        raise ValueError("Denominator differs")
    args.output.mkdir(parents=True,exist_ok=False)
    write_new_json(args.output/"frozen.json",{"source_assets_baseline":{str(args.baseline):sha256(args.baseline),
        str(Path(__file__)):sha256(Path(__file__)),str(CODE/"tasks/detection/text_consensus.py"):sha256(CODE/"tasks/detection/text_consensus.py")},
        "source_trial_prediction_sha256":receipt["prediction_sha256"],"prediction_reads_gold":False,
        "policy":"both original-pool mix winners agree AND both crop target and contrast improve"})
    reasons={}
    with (args.output/"predictions.jsonl").open("x") as out:
        for row in rows:
            prior=old[row["cohort"]+":"+row["id"]]
            boxes=prior["candidate_boxes"]
            baseline=prior["arms"]["G"]["selected_index"]
            winners=[]
            for label in ("0.25:original_pool","0.5:original_pool"):
                box=row["arms"][label]
                winners.append(boxes.index(box) if box in boxes else None)
            selected,reason=consensus_selection(baseline,*winners,prior["crop_scores"])
            reasons[reason]=reasons.get(reason,0)+1
            row["arms"]["consensus"]=boxes[selected] if selected is not None else None
            out.write(json.dumps(row,allow_nan=False)+"\n")
    write_new_json(args.output/"completion.json",{"completed":True,"count":len(rows),
        "prediction_sha256":sha256(args.output/"predictions.jsonl"),"frozen_sha256":sha256(args.output/"frozen.json"),"reasons":reasons})
    print(json.dumps(reasons))


if __name__=="__main__":
    main()

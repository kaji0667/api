"""Complete-denominator exact-match evaluation with unknown-answer and format diagnostics."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .cache import FeatureCache
from .config import SUPPORTED,resolve_device,resolve_path,write_json,sha256
from .data import normalize_answer,read_rows
from .inference import VQAPredictor,decode_prediction,load_bundle
from .model import unpack_visual

def summarize(predictions,labels):
    groups=defaultdict(list)
    for p in predictions:groups[p["question_type"]].append(p)
    per={}
    for t,ps in groups.items():
        n=len(ps);supported=t in SUPPORTED
        correct=sum(p["correct"] for p in ps);valid=sum(p["result"]["status"]=="ok" for p in ps)
        unknown=sum(supported and p["normalized_answer"] not in labels["answers"][t] for p in ps)
        known=[p for p in ps if supported and p["normalized_answer"] in labels["answers"][t]]
        per[t]=dict(n=n,correct=correct,accuracy=correct/n,valid_return_rate=valid/n,
                    unknown_answer_count=unknown if supported else None,unknown_answer_rate=unknown/n if supported else None,
                    known_answer_accuracy=sum(p["correct"] for p in known)/len(known) if known else None,
                    type_accuracy=sum(p["result"].get("question_type")==t for p in ps)/n,
                    failure_reasons=dict(Counter(p["result"].get("reason") for p in ps if p["result"]["status"]!="ok")))
    n=len(predictions);stage=[p for p in predictions if p["question_type"] in SUPPORTED]
    outside=[p for p in predictions if p["question_type"] not in SUPPORTED]
    yes_no={}
    for answer in ("yes","no"):
        ps=[p for p in predictions if p["question_type"]=="existence" and p["normalized_answer"]==answer]
        yes_no[answer]=dict(n=len(ps),accuracy=sum(p["correct"] for p in ps)/max(1,len(ps)))
    return dict(n=n,accuracy=sum(p["correct"] for p in predictions)/max(1,n),existence_yes_no=yes_no,
                out_of_scope_diagnostic=dict(n=len(outside),valid_return_rate=sum(p['result']['status']=='ok' for p in outside)/max(1,len(outside)),
                    exact_matches=sum(p['correct'] for p in outside),note='gold source type outside trained answer heads can be misrouted into a supported type; accidental exact matches do not establish that missing task capability'),
                valid_return_rate=sum(p["result"]["status"]=="ok" for p in predictions)/max(1,n),per_type=per,
                macro_accuracy=float(np.mean([v["accuracy"] for v in per.values()])),
                first_stage_n=len(stage),first_stage_accuracy=sum(p["correct"] for p in stage)/max(1,len(stage)),
                first_stage_macro_accuracy=float(np.mean([per[t]["accuracy"] for t in SUPPORTED if t in per])),
                first_stage_unknown_answer_rate=sum(p["normalized_answer"] not in labels["answers"][p["question_type"]] for p in stage)/max(1,len(stage)))

def record(row,result):
    return {**row,"result":result,"correct":bool(result["status"]=="ok" and normalize_answer(result["answer_text"])==row["normalized_answer"])}

def evaluate_cached(head,config,labels,cache,rows,device,batch,ablation=None,oracle_type=False):
    preds=[];head.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            ix=np.arange(start,min(start+batch,len(rows)));flat,q=cache.batch(ix,device)
            tl,al=head(unpack_visual(flat,config),q,ablation or config.get("train_ablation"))
            tl=tl.cpu();al={t:x.cpu() for t,x in al.items()}
            for j,i in enumerate(ix):
                r=rows[int(i)];route=tl[j]
                if oracle_type:
                    route=torch.full_like(route,-10000);route[labels["types"].index(r["question_type"])]=10000
                res=decode_prediction(route,{t:a[j] for t,a in al.items()},labels,r["question"],image_size=(r["width"],r["height"]))
                preds.append(record(r,res))
    return preds

def format_diagnostics(head,config,labels,cache,rows,device,batch):
    """Public QA-derived synthetic options, not native VRSBench or official multiple-choice tasks."""
    rng=random.Random(42);selected=[];counts=Counter()
    for i,r in enumerate(rows):
        t=r["question_type"]
        if t in SUPPORTED and r["normalized_answer"] in labels["answers"][t] and counts[t]<200:
            selected.append(i);counts[t]+=1
    modes={"single_choice":[],"enum":[]}
    for start in range(0,len(selected),batch):
        ix=np.array(selected[start:start+batch]);flat,q=cache.batch(ix,device)
        with torch.inference_mode():tl,al=head(unpack_visual(flat,config),q)
        tl=tl.cpu();al={t:x.cpu() for t,x in al.items()}
        for j,i in enumerate(ix):
            r=rows[int(i)];t=r["question_type"];gold=r["normalized_answer"]
            pool=[v for v in labels["answers"][t] if v!=gold]
            values=[gold]+rng.sample(pool,min(3,len(pool)));rng.shuffle(values)
            choices={chr(65+k):v for k,v in enumerate(values)};gold_key=next(k for k,v in choices.items() if v==gold)
            res=decode_prediction(tl[j],{t:a[j] for t,a in al.items()},labels,r["question"],choices=choices,image_size=(r["width"],r["height"]))
            modes["single_choice"].append(dict(sample_id=r["sample_id"],question_type=t,choices=choices,gold_key=gold_key,result=res,correct=res["status"]=="ok" and res["answer"]==gold_key))
            if t=="existence" and gold in {"yes","no"}:
                res=decode_prediction(tl[j],{t:a[j] for t,a in al.items()},labels,r["question"],allowed_answers=["Yes","No"],image_size=(r["width"],r["height"]))
                modes["enum"].append(dict(sample_id=r["sample_id"],allowed_answers=["Yes","No"],gold=gold.title(),result=res,correct=res["status"]=="ok" and res["answer"]==gold.title()))
    metrics={m:dict(n=len(ps),accuracy=sum(p["correct"] for p in ps)/max(1,len(ps)),valid_return_rate=sum(p["result"]["status"]=="ok" for p in ps)/max(1,len(ps))) for m,ps in modes.items()}
    return metrics,modes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("bundle-dir","jsonl","visual-backbone","text-encoder","out"):p.add_argument("--"+name,required=True)
    p.add_argument("--device",default="auto");p.add_argument("--cache-dir");p.add_argument("--batch",type=int,default=128)
    p.add_argument("--ablations",action="store_true");p.add_argument("--formats",action="store_true")
    p.add_argument("--oracle-type",action="store_true",help="Secondary head diagnostic using gold type; never deployment accuracy")
    a=p.parse_args();torch.set_num_threads(4);out=resolve_path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=read_rows(a.jsonl);assets=dict(visual_backbone=a.visual_backbone,text_encoder=a.text_encoder);device=resolve_device(a.device)
    manifest,config,labels,head,_=load_bundle(a.bundle_dir,assets);head.to(device);start=time.time()
    if a.cache_dir:
        cache=FeatureCache(a.cache_dir,rows,config)
        for key,path in assets.items():
            if cache.metadata["external_assets"][key]!=sha256(path):raise ValueError("Cache asset hash mismatch")
        write_json(out/"cache_data_binding.json",cache.verify_datasets([a.jsonl]))
        preds=evaluate_cached(head,config,labels,cache,rows,device,a.batch)
    else:
        predictor=VQAPredictor(a.bundle_dir,a.device,assets)
        preds=[record(r,predictor.predict(r["image_path"],r["question"],r.get("choices"),r.get("allowed_answers"))) for r in rows]
    metrics=summarize(preds,labels);metrics.update(protocol="canonical exact match; unknown, error, unsupported counted incorrect; full original val and first-stage subsets separately",
                                                   dataset_sha256=sha256(a.jsonl),seconds=time.time()-start,checkpoint_sha256=manifest["checkpoint"]["sha256"],
                                                   inference_path="verified frozen cache" if a.cache_dir else "online local predictor",
                                                   device=str(device),precision=config.get("precision"))
    majority={t:max(labels["counts"][t],key=labels["counts"][t].get) for t in SUPPORTED}
    maj=[record(r,dict(status="ok",answer=majority[r["question_type"]],answer_text=majority[r["question_type"]],question_type=r["question_type"])) if r["question_type"] in SUPPORTED else record(r,dict(status="unsupported",answer=None,answer_text=None,question_type=r["question_type"],reason="unsupported_task")) for r in rows]
    metrics["majority_answer_oracle_type_baseline"]=summarize(maj,labels)
    metrics["majority_baseline_note"]="training-only majority per gold type; oracle type provided, no image or learned question routing"
    if a.oracle_type:
        if not a.cache_dir:raise ValueError("Oracle type diagnostic requires cache")
        metrics["oracle_type_head_diagnostic"]=summarize(evaluate_cached(head,config,labels,cache,rows,device,a.batch,oracle_type=True),labels)
        metrics["oracle_type_note"]="Gold question type selects answer head for analysis only; complete-question scope guard retained; not deployable accuracy"
    if a.ablations:
        if not a.cache_dir:raise ValueError("Batched ablations require verified cache")
        metrics["inference_ablations"]={}
        for mode in ("question_only","image_only"):
            aps=evaluate_cached(head,config,labels,cache,rows,device,a.batch,mode)
            metrics["inference_ablations"][mode]=summarize(aps,labels)
        metrics["ablation_note"]="input removal at inference, not independently trained baselines; guards still inspect complete question"
    if a.formats:
        if not a.cache_dir:raise ValueError("Format diagnostics require cache")
        fm,fp=format_diagnostics(head,config,labels,cache,rows,device,a.batch)
        metrics["public_qa_synthetic_format_diagnostics"]=fm
        write_json(out/"format_predictions.json",fp)
    grouped=defaultdict(list)
    for pred in preds:grouped[pred["image_id"]].append(pred)
    multi=[v for v in grouped.values() if len(v)>1]
    metrics["same_image_multiple_questions"]=dict(images=len(multi),images_with_multiple_predicted_answers=sum(len({p["result"].get("answer_text") for p in v if p["result"]["status"]=="ok"})>1 for v in multi),note="different answers alone do not imply correctness")
    with (out/"val_predictions.jsonl").open("w",encoding="utf-8") as f:
        for r in preds:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    write_json(out/"metrics.json",metrics)
    print(json.dumps({k:metrics[k] for k in ("n","first_stage_n","accuracy","first_stage_macro_accuracy","valid_return_rate","first_stage_unknown_answer_rate")}),flush=True)
if __name__ == "__main__":main()

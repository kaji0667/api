"""Train only the new VQA modules on isolated original QA splits."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import importlib.metadata as md
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .cache import FeatureCache,prepare_cache
from .config import FORMAT_VERSION,SUPPORTED,QFORMER_DEFAULTS,architecture_id,config_from_file,resolve_device,resolve_path,sha256,write_json,validate_fusion_config
from .data import assert_split_isolation,build_labels,read_rows
from .inference import decode_prediction
from .model import VQAHead,unpack_visual

def inference_config(c):
    fields=("imgsz","grids","channels","text_dim","hidden_dim","dropout","text_model","text_chunk_tokens","model_yaml")
    result={**{k:c[k] for k in fields},"train_ablation":c.get("train_ablation"),"image_preprocessing":"PIL RGB, bilinear resize to square, /255, ImageNet mean/std; no crop/letterbox",
            "text_preprocessing":"complete original question; standard CLIP BPE; continuous 75-token windows; length-weighted pooled vectors normalized",
            "text_max_characters":10000,"text_max_content_tokens":750,"precision":"fp32 encoders/head; pooled visual fp16 rounding matches cache",
            "spatial_evidence":"no explicit localization; target/reference boxes empty"}
    validate_fusion_config(c)
    result.update(fusion_backend="blip2_qformer",**{k:c.get(k,v) for k,v in QFORMER_DEFAULTS.items()})
    return result

def save_bundle(root,head,config,labels,epoch,score,assets,environment):
    root=Path(root);(root/"weights").mkdir(parents=True,exist_ok=True)
    inf=inference_config(config)
    write_json(root/"config/train.json",config);write_json(root/"config/inference.json",inf);write_json(root/"config/labels.json",labels)
    selected_architecture=architecture_id(inf)
    payload=dict(format_version=FORMAT_VERSION,architecture_id=selected_architecture,
                 config=inf,labels=labels,epoch=int(epoch),validation_macro_accuracy=float(score),
                 state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()})
    tmp=root/"weights/vqa_head.tmp";torch.save(payload,tmp);tmp.replace(root/"weights/vqa_head.pt")
    paths=["weights/vqa_head.pt","config/train.json","config/inference.json","config/labels.json"]
    manifest=dict(format_version=FORMAT_VERSION,task="remote_sensing_vqa",architecture_id=selected_architecture,
                  checkpoint=dict(path=paths[0],sha256=sha256(root/paths[0]),serialization="state_dict plus primitive metadata; weights_only=True"),
                  configs=dict(train=paths[1],inference=paths[2],labels=paths[3]),external_assets=assets,
                  file_sha256={p:sha256(root/p) for p in paths},supported_question_types=list(SUPPORTED),
                  unsupported_question_types=[t for t in labels["types"] if t not in SUPPORTED],
                  training_sources=["VRSBench original public qa_pairs"],
                  split_grouping="original train/val, disjoint image ID/parent prefix/file hash/decoded RGB hash; site/geolocation identity unavailable",
                  verified_devices=[],verified_precisions=[],python_version=environment["python"],
                  metrics="best_validation.json",environment="environment.json",
                  validation_predictions="evaluation/val_predictions.jsonl",evaluation_status="pending",
                  limitations=["finite training-only answer vocabulary","640 full-image baseline; no high-resolution regions or explicit localization","English questions only","softmax confidence is not calibrated accuracy","same parent name isolation does not prove geographical isolation"])
    from .qformer import UPSTREAM
    manifest.update(fusion_backend="blip2_qformer",fusion_source=UPSTREAM)
    manifest["limitations"].append("Adapted Q-Former structure; no pretrained BLIP-2/BERT/LLM weights or BLIP-2 pretraining objectives")
    write_json(root/"manifest.json",manifest)

def validate(head,cache,rows,labels,config,device,max_batches=0):
    head.eval();counts=Counter();correct=Counter();valid=Counter();known=Counter();preds=[]
    with torch.no_grad():
        for step,start in enumerate(range(0,len(rows),config["batch"])):
            if max_batches and step>=max_batches:break
            ix=np.arange(start,min(start+config["batch"],len(rows)));flat,q=cache.batch(ix,device)
            tl,al=head(unpack_visual(flat,config),q,config.get("train_ablation"))
            tl=tl.cpu();al={t:x.cpu() for t,x in al.items()}
            for j,i in enumerate(ix):
                row=rows[int(i)];t=row["question_type"];counts[t]+=1
                known[t]+=int(t in SUPPORTED and row["normalized_answer"] in labels["answers"][t])
                result=decode_prediction(tl[j],{t:x[j] for t,x in al.items()},labels,row["question"],image_size=(row["width"],row["height"]))
                ok=result["status"]=="ok";valid[t]+=int(ok)
                hit=ok and result["answer_text"]==row["normalized_answer"];correct[t]+=int(hit)
                preds.append(dict(sample_id=row["sample_id"],question_type=t,gold=row["normalized_answer"],result=result,correct=bool(hit)))
    per={t:dict(n=counts[t],correct=correct[t],accuracy=correct[t]/counts[t],valid_return_rate=valid[t]/counts[t],
                known_answer_rate=known[t]/counts[t] if t in SUPPORTED else None) for t in counts}
    macro=float(np.mean([per[t]["accuracy"] for t in SUPPORTED if t in per]))
    return dict(per_type=per,first_stage_macro_accuracy=macro,n=sum(counts.values()),
                accuracy=sum(correct.values())/max(1,sum(counts.values())),valid_return_rate=sum(valid.values())/max(1,sum(counts.values()))),preds

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("train-jsonl","val-jsonl","config","save-dir"):p.add_argument("--"+n,required=True)
    p.add_argument("--device",default="auto");p.add_argument("--cache-dir")
    p.add_argument("--max-train-batches",type=int,default=0);p.add_argument("--max-val-batches",type=int,default=0)
    a=p.parse_args();c=config_from_file(a.config);root=resolve_path(a.save_dir)
    if (root/"manifest.json").exists():raise FileExistsError("Choose an independent training output directory")
    root.mkdir(parents=True,exist_ok=True);device=resolve_device(a.device);torch.set_num_threads(4)
    random.seed(c["seed"]);np.random.seed(c["seed"]);torch.manual_seed(c["seed"])
    torch.backends.cudnn.benchmark=False
    train=read_rows(a.train_jsonl);val=read_rows(a.val_jsonl);isolation=assert_split_isolation(train,val)
    write_json(root/"split_isolation.json",isolation)
    labels=build_labels(train,c["min_answer_count"])
    c.update(train_jsonl_sha256=sha256(a.train_jsonl),val_jsonl_sha256=sha256(a.val_jsonl),command=sys.argv,
             max_train_batches=a.max_train_batches,max_val_batches=a.max_val_batches)
    environment=dict(python=sys.version,device=str(device),cuda=torch.version.cuda,
                     packages={k:md.version(k) for k in ("torch","torchvision","open-clip-torch","numpy","Pillow","PyYAML","opencv-python")})
    write_json(root/"environment.json",environment)
    cache_dir=resolve_path(a.cache_dir or c.get("cache_dir",root/"feature_cache"))
    if not (cache_dir/"metadata.json").is_file():prepare_cache([a.train_jsonl,a.val_jsonl],c,cache_dir,device,batch=2,text_batch=64)
    rows=train+val;cache=FeatureCache(cache_dir,rows,c)
    write_json(root/"cache_data_binding.json",cache.verify_datasets([a.train_jsonl,a.val_jsonl]))
    # Validation view shares RAM but owns independent index arrays.
    import copy
    val_cache=copy.copy(cache);val_cache.row_images=cache.row_images[len(train):];val_cache.row_text=cache.row_text[len(train):]
    cache.row_images=cache.row_images[:len(train)];cache.row_text=cache.row_text[:len(train)]
    head=VQAHead(c,labels).to(device);optimizer=torch.optim.AdamW(head.parameters(),lr=c["lr"],weight_decay=c["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,c["epochs"])
    type_ids=np.array([labels["types"].index(r["question_type"]) for r in train])
    answer_ids=np.array([labels["answers"][r["question_type"]].index(r["normalized_answer"]) if r["question_type"] in SUPPORTED and r["normalized_answer"] in labels["answers"][r["question_type"]] else -1 for r in train])
    cnt=Counter(type_ids);prob=np.array([1/cnt[i] for i in type_ids]);prob/=prob.sum()
    assets=[dict(asset_id=k,family="YOLOv12m" if k=="visual_backbone" else "OpenCLIP",model_name="YOLOv12m" if k=="visual_backbone" else "ViT-L-14/openai",sha256=sha256(c[k]),modified=False) for k in ("visual_backbone","text_encoder")]
    write_json(root/"labels_audit.json",dict(train_qa=len(train),val_qa=len(val),vocabulary_sizes={t:len(v) for t,v in labels["answers"].items()},
               learned_parameters=sum(p.numel() for p in head.parameters()),frozen_feature_report=cache.metadata["backbone_load_report"]))
    best=-1;best_epoch=0;stale=0;beg=time.time();rng=np.random.default_rng(c["seed"])
    with (root/"history.csv").open("w",newline="") as log:
        writer=csv.DictWriter(log,fieldnames=("epoch","train_loss","answer_loss","type_loss","macro_accuracy","full_accuracy","lr","seconds"));writer.writeheader()
        for epoch in range(1,c["epochs"]+1):
            head.train();order=rng.choice(len(train),len(train),replace=True,p=prob) if c["sampling"]=="type_balanced" else rng.permutation(len(train))
            total=0;ans_total=0;type_total=0;steps=0
            for step,start in enumerate(range(0,len(order),c["batch"])):
                if a.max_train_batches and step>=a.max_train_batches:break
                ix=order[start:start+c["batch"]];flat,q=cache.batch(ix,device)
                types,answers=head(unpack_visual(flat,c),q,c.get("train_ablation"));target=torch.from_numpy(type_ids[ix]).to(device)
                type_loss=F.cross_entropy(types,target);terms=[]
                for t in SUPPORTED:
                    mask=(type_ids[ix]==labels["types"].index(t))&(answer_ids[ix]>=0)
                    if mask.any():
                        m=torch.from_numpy(mask).to(device);y=torch.from_numpy(answer_ids[ix][mask]).to(device)
                        terms.append(F.cross_entropy(answers[t][m],y))
                answer_loss=sum(terms)/max(1,len(terms))
                loss=answer_loss+c["type_loss_weight"]*type_loss
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.0);optimizer.step()
                total+=float(loss.detach());ans_total+=float(answer_loss.detach()) if terms else 0;type_total+=float(type_loss.detach());steps+=1
            metrics,_=validate(head,val_cache,val,labels,c,device,a.max_val_batches)
            score=metrics["first_stage_macro_accuracy"]
            item=dict(epoch=epoch,train_loss=total/steps,answer_loss=ans_total/steps,type_loss=type_total/steps,
                      macro_accuracy=score,full_accuracy=metrics["accuracy"],lr=optimizer.param_groups[0]["lr"],seconds=time.time()-beg)
            writer.writerow(item);log.flush();print(json.dumps(item),flush=True)
            if score>best:
                best=score;best_epoch=epoch;stale=0;save_bundle(root,head,c,labels,epoch,score,assets,environment)
                write_json(root/"best_validation.json",metrics)
            else:stale+=1
            scheduler.step()
            if stale>=c["patience"]:break
    write_json(root/"training_result.json",dict(best_epoch=best_epoch,best_macro_accuracy=best,epochs_completed=epoch,
              seconds=time.time()-beg,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
              evaluated_val_qa=metrics["n"],validation_scope="limited_batches" if a.max_val_batches else "full_split",
              checkpoint_selection=("limited-batch" if a.max_val_batches else "full")+" first-stage macro validation exact match incl unknown/invalid; no test tuning"))
if __name__ == "__main__":main()

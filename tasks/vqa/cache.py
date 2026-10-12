"""Reproducible frozen-feature cache; also used for batched evaluation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .config import config_from_file, resolve_device, resolve_path, sha256, write_json
from .data import read_rows,question_type
from .model import FrozenVisual, FrozenText, flatten_visual

def prepare_cache(jsonls, config, out, device, batch=4, text_batch=128):
    out=resolve_path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[r for p in jsonls for r in read_rows(p)]
    images={r["image_id"]:r["image_path"] for r in rows}
    questions=list(dict.fromkeys(r["question"] for r in rows));ids=sorted(images)
    image_hashes={r["image_id"]:r["image_sha256"] for r in rows}
    metadata=dict(format_version="1.0",config=config,images=ids,questions=questions,
                  data_sha256={str(p):sha256(p) for p in jsonls},
                  external_assets={k:sha256(config[k]) for k in ("visual_backbone","text_encoder")},
                  storage="float16 pooled YOLO / float32 normalized CLIP question",
                  ordered_image_hashes=[image_hashes[i] for i in ids])
    meta_path=out/"metadata.json"
    if meta_path.is_file():
        old=json.loads(meta_path.read_text())
        if all(old[k] == metadata[k] for k in ("config","images","questions","data_sha256","external_assets")):
            print("validated existing cache",flush=True);return out
        raise ValueError("Existing cache differs; choose an independent output directory")
    flat_dim=sum(c*g*g for c,g in zip(config["channels"],config["grids"]))
    visual=np.lib.format.open_memmap(out/"visual.npy",mode="w+",dtype=np.float16,shape=(len(ids),flat_dim))
    text=np.lib.format.open_memmap(out/"text.npy",mode="w+",dtype=np.float32,shape=(len(questions),768))
    device=resolve_device(device);torch.set_num_threads(4)
    started=time.time();net=FrozenVisual(config["visual_backbone"],config,device)
    metadata["backbone_load_report"]=net.load_report
    for start in range(0,len(ids),batch):
        batch_ids=ids[start:start+batch]
        tensors=[net.image_tensor(images[i])[0] for i in batch_ids]
        with torch.no_grad():
            result=flatten_visual(net(torch.stack(tensors).to(device))).cpu().numpy()
        visual[start:start+len(batch_ids)]=result
        if start%(batch*100)==0:print(f"visual {start}/{len(ids)} elapsed={time.time()-started:.1f}s",flush=True)
    visual.flush();del net
    if device.type=="cuda":torch.cuda.empty_cache()
    encoder=FrozenText(config["text_encoder"],device)
    for start in range(0,len(questions),text_batch):
        qs=questions[start:start+text_batch]
        text[start:start+len(qs)]=encoder(qs).cpu().numpy()
        if start%(text_batch*20)==0:print(f"text {start}/{len(questions)} elapsed={time.time()-started:.1f}s",flush=True)
    text.flush();metadata["seconds"]=time.time()-started
    metadata["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0
    write_json(meta_path,metadata);return out

class FeatureCache:
    def __init__(self,path,rows,config,load_ram=True):
        self.path=resolve_path(path)
        self.metadata=json.loads((self.path/"metadata.json").read_text())
        for k in ("imgsz","grids","channels","text_model","text_chunk_tokens"):
            if self.metadata["config"][k] != config[k]:raise ValueError("Cache preprocessing mismatch")
        for key in ("visual_backbone","text_encoder"):
            if key in config and sha256(config[key])!=self.metadata["external_assets"][key]:
                raise ValueError("Cache frozen asset mismatch")
        self.image_index={i:n for n,i in enumerate(self.metadata["images"])}
        self.text_index={q:n for n,q in enumerate(self.metadata["questions"])}
        hashes=dict(zip(self.metadata["images"],self.metadata["ordered_image_hashes"]))
        if any(hashes.get(r["image_id"]) != r["image_sha256"] for r in rows):raise ValueError("Cache image identity mismatch")
        self.visual=np.load(self.path/"visual.npy",mmap_mode="r")
        self.text=np.load(self.path/"text.npy",mmap_mode="r")
        if load_ram:
            self.visual=np.array(self.visual);self.text=np.array(self.text)
        self.row_images=np.array([self.image_index[r["image_id"]] for r in rows])
        self.row_text=np.array([self.text_index[r["question"]] for r in rows])
    def verify_datasets(self,jsonls):
        """Allow only an audited type-metadata correction; keep cache provenance intact."""
        original={}
        for path,digest in self.metadata["data_sha256"].items():
            if sha256(path)!=digest:raise ValueError("Original cache dataset changed")
            for row in read_rows(path):
                if row["sample_id"] in original:raise ValueError("Duplicate cache sample identity")
                original[row["sample_id"]]=row
        report=[]
        for path in jsonls:
            rows=read_rows(path);changed=0
            for row in rows:
                old=original.get(row["sample_id"])
                if old is None:raise ValueError("Unknown cache sample identity")
                if {k:v for k,v in row.items() if k!="question_type"}!={k:v for k,v in old.items() if k!="question_type"}:
                    raise ValueError("Cache sample changed beyond type metadata")
                if row["question_type"]!=old["question_type"]:
                    if row["question_type"]!=question_type(row["original_type"],row["question"]):
                        raise ValueError("Unverified question-type correction")
                    changed+=1
            report.append(dict(dataset=str(path),sha256=sha256(path),samples=len(rows),type_only_corrections=changed))
        return dict(original_data_sha256=self.metadata["data_sha256"],verified_datasets=report,
                    invariant="every sample field except audited question_type equals original; original file hashes verified")
    def batch(self,ix,device):
        v=torch.from_numpy(np.array(self.visual[self.row_images[ix]],dtype=np.float32)).to(device)
        q=torch.from_numpy(np.array(self.text[self.row_text[ix]],dtype=np.float32)).to(device)
        return v,q

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--jsonl",nargs="+",required=True)
    p.add_argument("--config",required=True);p.add_argument("--out",required=True)
    p.add_argument("--device",default="auto");p.add_argument("--batch",type=int,default=4)
    p.add_argument("--text-batch",type=int,default=128)
    a=p.parse_args();prepare_cache(a.jsonl,config_from_file(a.config),a.out,a.device,a.batch,a.text_batch)
if __name__ == "__main__":main()

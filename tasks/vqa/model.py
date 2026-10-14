"""Frozen YOLO/OpenCLIP adapters and task-owned BLIP-2 Q-Former fusion."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from .config import SUPPORTED, validate_fusion_config

class FrozenVisual(nn.Module):
    def __init__(self, asset, config, device):
        super().__init__()
        from tasks.classification.model import SceneClassificationNetwork
        from tasks.classification.data import SceneDataModule
        net = SceneClassificationNetwork(config["model_yaml"], 21, asset, "cpu")
        source = net._state_dict_from_checkpoint(torch.load(asset,map_location="cpu",weights_only=False))
        target = net.model.state_dict()
        required = {k:v for k,v in target.items() if not k.startswith("model.21.")}
        missing = [k for k,v in required.items() if k not in source or source[k].shape != v.shape]
        if missing: raise ValueError(f"Incomplete YOLO backbone/neck weights: {missing[:5]}")
        self.load_report = dict(required_tensors=len(required),matched_tensors=len(required),skipped_head=True)
        self.layers = nn.ModuleList(net.model.model[:-1])
        self.grids = config["grids"]
        self.channels = config["channels"]
        self.transform = SceneDataModule.build_transform(config["imgsz"],train=False)
        self.requires_grad_(False);self.eval();self.to(device)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        saved=[];x=images
        for m in self.layers:
            if m.f != -1:
                x=saved[m.f] if isinstance(m.f,int) else [x if j == -1 else saved[j] for j in m.f]
            x=m(x);saved.append(x)
        result=[F.adaptive_avg_pool2d(saved[i],g) for i,g in zip((14,17,20),self.grids)]
        if [x.shape[1] for x in result] != self.channels: raise ValueError("Unexpected YOLO feature channels")
        return result

    def image_tensor(self,image):
        if isinstance(image, Image.Image):
            rgb=image.convert("RGB")
        elif isinstance(image,(str,Path)):
            with Image.open(image) as im:
                im.load();rgb=im.convert("RGB")
        else: raise ValueError("image must be PIL, str or Path")
        return self.transform(rgb),rgb.size

class FrozenText(nn.Module):
    """Continuous BPE chunks retain every original question token; no text templates."""
    def __init__(self,asset,device):
        super().__init__()
        import open_clip
        self.encoder = open_clip.create_model("ViT-L-14",pretrained=str(asset),device="cpu")
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.encoder.requires_grad_(False);self.encoder.eval();self.encoder.to(device)
        self.device=torch.device(device)

    def train(self,mode=True): return super().train(False)

    def chunks(self, question):
        if not isinstance(question,str) or not question.strip(): raise ValueError("question must be nonempty text")
        if len(question)>10000: raise ValueError("question exceeds declared character budget")
        # SimpleTokenizer.encode has no truncation. Standard CLIP start/end are retained in each window.
        ids=self.tokenizer.encode(question)
        if not ids: raise ValueError("question has no content tokens")
        if len(ids)>750: raise ValueError("question exceeds declared 10-window token budget")
        windows=[];lengths=[]
        for start in range(0,len(ids),75):
            part=ids[start:start+75];x=torch.zeros(77,dtype=torch.long)
            x[0]=self.tokenizer.sot_token_id;x[1:1+len(part)]=torch.tensor(part);x[1+len(part)]=self.tokenizer.eot_token_id
            windows.append(x);lengths.append(len(part))
        return windows,lengths

    @torch.no_grad()
    def forward(self,questions):
        tokens=[];groups=[];weights=[]
        for i,q in enumerate(questions):
            xs,ns=self.chunks(q);tokens.extend(xs);weights.extend(ns);groups.extend([i]*len(xs))
        embs=self.encoder.encode_text(torch.stack(tokens).to(self.device)).float()
        out=torch.zeros(len(questions),768,device=self.device)
        mass=torch.zeros(len(questions),1,device=self.device)
        ix=torch.tensor(groups,device=self.device);w=torch.tensor(weights,device=self.device).float().unsqueeze(1)
        out.index_add_(0,ix,embs*w);mass.index_add_(0,ix,w)
        return F.normalize(out/mass.clamp_min(1),dim=-1)

def flatten_visual(features):
    return torch.cat([f.flatten(1) for f in features],dim=1)

def unpack_visual(flat, config):
    result=[];offset=0
    for c,g in zip(config["channels"],config["grids"]):
        n=c*g*g;result.append(flat[:,offset:offset+n].view(-1,c,g,g));offset+=n
    if offset != flat.shape[1]: raise ValueError("Invalid cached visual shape")
    return result

class ResidualAnswerHead(nn.Module):
    """Task-owned residual MLP followed by the trained answer classifier."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x + self.mlp(x))


class VQAHead(nn.Module):
    """Every trainable parameter, including question routing, belongs to this state_dict."""
    def __init__(self, config, labels):
        super().__init__()
        self.fusion_backend=validate_fusion_config(config)
        d=config["hidden_dim"];self.config=config
        self.projections=nn.ModuleList(nn.Conv2d(c,d,1) for c in config["channels"])
        self.position=nn.Sequential(nn.Linear(3,d),nn.GELU(),nn.Linear(d,d))
        self.question=nn.Sequential(nn.Linear(768,d),nn.LayerNorm(d),nn.GELU())
        self.type_classifier=nn.Sequential(nn.Linear(768,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(config["dropout"]),nn.Linear(d,len(labels["types"])))
        self.visual_norm=nn.LayerNorm(d)
        self.fusion=nn.Sequential(nn.Linear(d*4,d*2),nn.LayerNorm(d*2),nn.GELU(),nn.Dropout(config["dropout"]))
        self.answers=nn.ModuleDict({t:ResidualAnswerHead(d*2,len(labels["answers"][t]),dropout=config["dropout"]) for t in SUPPORTED})
        from .qformer import QuestionQueryFormer
        self.query_former=QuestionQueryFormer(config)
        coords=[]
        for level,g in enumerate(config["grids"]):
            yy,xx=torch.meshgrid(torch.linspace(-1,1,g),torch.linspace(-1,1,g),indexing="ij")
            coords.append(torch.stack([xx.flatten(),yy.flatten(),torch.full((g*g,),float(level-1))],-1))
        self.register_buffer("coords",torch.cat(coords),persistent=False)

    def forward(self,visual,question,ablation=None):
        if ablation == "image_only": question=torch.zeros_like(question)
        if ablation == "question_only": visual=[torch.zeros_like(v) for v in visual]
        q=self.question(question)
        tokens=torch.cat([p(v).flatten(2).transpose(1,2) for p,v in zip(self.projections,visual)],dim=1)
        tokens=self.visual_norm(tokens+self.position(self.coords)[None])
        readout_tokens,q=self.query_former(tokens,q)
        attention=(readout_tokens*q[:,None]).sum(-1)/(q.shape[-1]**0.5)
        attended=(attention.softmax(-1)[:,:,None]*readout_tokens).sum(1)
        global_v=tokens.mean(1)
        joint=self.fusion(torch.cat([q,attended,global_v,q*attended],-1))
        return self.type_classifier(question),{t:h(joint) for t,h in self.answers.items()}

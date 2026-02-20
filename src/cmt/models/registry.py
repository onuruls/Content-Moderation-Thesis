from typing import Any
from cmt.models.base import Model

def get_model(cfg: dict[str, Any]) -> Model:
    name = str(cfg.get("model_name", "")).lower()
    
    if name == "clip_multilabel":
        from cmt.models.clip_multilabel import CLIPMultiLabelTrained
        return CLIPMultiLabelTrained(
            head_path=cfg["model_path"],
            dtype=cfg.get("dtype", "float16")
        )
    
    elif name == "clip_lp":
        from cmt.models.clip_lp import CLIPLinearProbe
        return CLIPLinearProbe(
            model_id=cfg.get("model_id", "ViT-L-14"),
            pretrained=cfg.get("pretrained", "openai"),
            head_path=cfg["linear_head_path"],
            l2norm=bool(cfg.get("l2norm", True)),
            dtype=cfg.get("dtype", "float16")
        )
        
    elif name == "siglip_lp":
        from cmt.models.siglip_lp import SigLIPLinearProbe
        return SigLIPLinearProbe(
            model_id=cfg.get("model_id", "google/siglip-so400m-patch14-384"),
            head_path=cfg.get("linear_head_path"),
            l2norm=bool(cfg.get("l2norm", True)),
            dtype=cfg.get("dtype", "float16")
        )
        
    elif name == "wdtagger":
        from cmt.models.eva_multilabel import WDEva02Tagger
        return WDEva02Tagger(
            repo_id=cfg.get("repo_id", "SmilingWolf/wd-eva02-large-tagger-v3"),
            dtype=cfg.get("dtype", "float16")
        )
        
    elif name == "animetimm":
        from cmt.models.eva_multilabel import AnimetimmEva02Tagger
        return AnimetimmEva02Tagger(
            repo_id=cfg.get("repo_id", "animetimm/eva02_large_patch14_448.dbv4-full"),
            dtype=cfg.get("dtype", "float16")
        )
        
    elif name == "hysac":
        from cmt.models.hysac import HySACWrapper
        return HySACWrapper(
            model_id=cfg.get("model_id", "aimagelab/HySAC"),
            dtype=cfg.get("dtype", "float16")
        )
        
    else:
        raise ValueError(f"Unknown model_name: {name}")

from typing import Generator, Dict, Any

def get_dataset(name: str, split: str = "test", cfg: dict = None) -> Generator[Dict[str, Any], None, None]:
    name = name.lower()
    cfg = cfg or {}
    
    if name == "nudenet":
        from cmt.data.nudenet import iter_nudenet
        return iter_nudenet(split=split, sexy_policy=cfg.get("sexy_policy", "exclude"))
        
    elif name == "lspd":
        from cmt.data.lspd import iter_lspd
        return iter_lspd(split=split, domain=cfg.get("lspd_domain", "both"))
        
    elif name == "unsafebench":
        from cmt.data.unsafebench import iter_unsafebench
        return iter_unsafebench(split=split)
        
    elif name == "internal":
        from cmt.data.internal import iter_internal
        return iter_internal(split=split)
        
    else:
        raise ValueError(f"Unknown dataset: {name}")

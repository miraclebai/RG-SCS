"""Read RG-SCS NPZ data. PNGs are inspection-only; real records have no clean labels."""
from pathlib import Path
import csv,json
import numpy as np
from radar_common import CORE,phase_labels
ROOT=Path(__file__).resolve().parents[1]
def manifest(root=ROOT,split=None):
    with (Path(root)/"reports/manifest.csv").open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"]==split]
def read_sample(row,root=ROOT):
    root=Path(root)
    with np.load(root/row["input_path"],allow_pickle=False) as q:
        out=dict(x=q["x"],refs_left=q["refs_left"],refs_right=q["refs_right"],refs=q["refs"],
                 scale=float(q["scale"]),metadata=json.loads(q["metadata_json"].item()),target=None,background=None,
                 relation_labels=None)
        if "background_raw" in q:out["background"]=(q["background_raw"][:,CORE]/float(q["scale"])).astype("c8")
    if row["label_path"]:
        with np.load(root/row["label_path"],allow_pickle=False) as q:
            out["target"]=q["s"]
            out["relation_labels"]={k:q[k] for k in q.files if k.startswith("phase_") or k in ["relation_frequency_hz","frequency_bin","frequency_offset_hz"]}
    elif row["kind"]=="background":
        # Reviewed absence assumption; there is intentionally no label file.
        out["target"]=np.zeros((256,16),"c8")
    elif row["kind"]!="real":raise ValueError("Unknown sample kind")
    if row["kind"]=="real" and out["target"] is not None:raise AssertionError("Invented real label")
    return out
try:
    import torch
    from torch.utils.data import Dataset
    class RadarDataset(Dataset):
        def __init__(self,split="train",root=ROOT,mode="supervised"):
            self.root=Path(root);self.mode=mode
            if mode not in ["supervised","inference"]:raise ValueError("mode")
            self.rows=manifest(root,split)
            if mode=="supervised":self.rows=[r for r in self.rows if r["kind"] in ["synthetic","background"]]
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            row=self.rows[i];q=read_sample(row,self.root)
            out=dict(x=torch.from_numpy(q["x"]),refs_left=torch.from_numpy(q["refs_left"]),
                     refs_right=torch.from_numpy(q["refs_right"]),id=row["id"],kind=row["kind"])
            if self.mode=="supervised":
                out["s"]=torch.from_numpy(q["target"]);out["b"]=torch.from_numpy(q["background"])
                out["is_positive"]=row["kind"]=="synthetic"
                labels=q["relation_labels"] or phase_labels(np.zeros((256,16),"c8"))
                out.update({k:torch.as_tensor(v) for k,v in labels.items()})
                if row["kind"]=="background":
                    out["frequency_bin"]=torch.tensor(-1,dtype=torch.int32)
            return out
except ImportError:
    RadarDataset=None
if __name__=="__main__":
    from collections import Counter
    rows=manifest()
    print(Counter((r["split"],r["kind"]) for r in rows))
    for split in ["train","test"]:
        ds=RadarDataset(split,mode="supervised")
        sample=ds[0]
        print(split,len(ds),{k:list(v.shape) for k,v in sample.items() if hasattr(v,"shape")})

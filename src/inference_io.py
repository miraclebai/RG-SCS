"""Input loading, single-panel RD plots and JSON output for RG-SCS."""
from pathlib import Path
import os,csv,json
import numpy as np
from radar_common import rd,FB,PRF
ROOT=Path(__file__).resolve().parents[1]

def rows(root=ROOT, split='test'):
    with (Path(root)/'reports/manifest.csv').open(newline='',encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r['split']==split]

def load_input(row,root=ROOT):
    # Never load background_raw, target, label-derived frequency/SCR/support.
    with np.load(Path(root)/row['input_path'],allow_pickle=False) as z:
        q={k:z[k].copy() for k in ['x','refs_left','refs_right']}
    if q['x'].shape!=(256,16) or not np.iscomplexobj(q['x']):
        raise ValueError('Expected normalized complex IQ [256,16]')
    if not all(np.isfinite(v).all() for v in q.values()): raise ValueError('Nonfinite input')
    return q

def load_target(row,root=ROOT):
    if row['kind']=='background': return np.zeros((256,16),np.complex64)
    if row['kind']=='real': return None
    if not row['label_path']: raise ValueError('Synthetic sample has no clean label')
    with np.load(Path(root)/row['label_path'],allow_pickle=False) as z: return z['s'].copy()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)

def plot(path,z,title,first,vmax):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    a=rd(z)/max(vmax,1e-20);df=PRF/z.shape[0]
    fig,ax=plt.subplots(figsize=(6.5,4.7),dpi=140)
    im=ax.imshow(a,origin='lower',aspect='auto',interpolation='nearest',cmap='jet',vmin=0,vmax=1,
       extent=[FB[0]-df/2,FB[-1]+df/2,first-.5,first+z.shape[1]-.5])
    ax.set(xlabel='Doppler Frequency (Hz)',ylabel='Range Bin',title=title)
    fig.colorbar(im,ax=ax,label='Amplitude / shared input-label maximum')
    fig.tight_layout();fig.savefig(path);plt.close(fig)

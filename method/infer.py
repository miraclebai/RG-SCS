"""Input-only predictions; labels loaded afterwards for fixed legacy + component metrics."""
from pathlib import Path
import sys,time,json,csv,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inference_io import rows,load_input,load_target,plot,dump
from evaluation_metrics import calculate
from radar_common import rd
from model import RGSCS

def component_metrics(m,x,s,rel,prf=2000):
    target=torch.from_numpy(s)[None].to(x.device)
    bg=x-target
    double={k:torch.cat([rel[k],rel[k]],0) for k in ['omega','a','c']}
    parts,res=m.solve(torch.cat([target,bg],0),double);gs,gb=parts.chunk(2)
    e=lambda z:float(z.abs().square().sum())
    out=dict(target_keep_error=e(gs-target)/max(e(target),1e-30),background_leak_rate=e(gb)/max(e(bg),1e-30),
             component_solve_residual=float(res.max()),target_energy_retention=e(gs)/max(e(target),1e-30))
    def spectrum(z):
        win=torch.hann_window(256,periodic=False,device=z.device,dtype=torch.float64)[None,:,None]
        return torch.fft.fft(z*win,dim=1).abs().square().sum((0,2))
    pin=spectrum(bg);pout=spectrum(gb);f=torch.fft.fftfreq(256,d=1/prf,device=x.device)
    for limit,tag in [(500,'display'),(1000,'full')]:
        for sign,mask in [('positive',(f>0)&(f<limit)),('negative',(f<0)&(f>-limit))]:
            a=float(pin[mask].sum());b=float(pout[mask].sum())
            out[f'{tag}_{sign}_background_suppression_db']=10*np.log10(max(a,1e-30)/max(b,1e-30))
    return out

def infer(run, tag, data_root=ROOT, device="cuda:0", output_root=ROOT/"inference"):
    ckpath=run/(tag+'.pt');ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    cfg=ck['config'].copy();cfg.update(cg_iterations=384,cg_tolerance=1e-6)
    m=RGSCS(cfg).to(device);m.load_state_dict(ck['model']);m.eval()
    dest=output_root/tag;sha=hashlib.sha256(ckpath.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256((data_root/'reports/manifest.csv').read_bytes()).hexdigest()
    dataset_id = str(data_root.resolve())
    if (dest/'completed.json').exists():
        done=json.loads((dest/'completed.json').read_text())
        if (done['checkpoint_sha256']==sha and done.get('manifest_sha256')==manifest_hash
                and done.get('data_root')==dataset_id):return json.loads((dest/'summary.json').read_text())
        raise RuntimeError('Completed inference belongs to a different checkpoint or dataset; choose a new --output-dir')
    dest.mkdir(parents=True,exist_ok=True)
    rr=rows(data_root)
    if not rr: raise ValueError('No test samples in manifest')
    records=[];start=time.time()
    dump(dest/'protocol.json',dict(checkpoint=str(ckpath),checkpoint_sha256=sha,epoch=ck['epoch'],
         legacy_metrics='Hann RD amplitude PSNR; truth-defined target/main-lobe clutter regions',
         component_metrics='G_X estimated once from mixed input and guarded references; same G_X applied to S and B after prediction',
         side_metrics='Hann FFT background component power; symmetric bands excluding zero and endpoints; mean of per-sample dB',
         real_clean_truth_available=False,save_npz=False,one_panel_per_png=True))
    for i,row in enumerate(rr):
        q=load_input(row,data_root)
        to=lambda k:torch.from_numpy(q[k])[None].to(device)
        with torch.no_grad():
            x=to('x');rel=m.relations(x,to('refs_left'),to('refs_right'));y,res=m.solve(x,rel)
            assert bool(torch.isfinite(y).all()) and float(res.max())<5e-5
            output=y[0].cpu().numpy()
        # Labels are never provided to the inference network or relation calibration.
        target=load_target(row,data_root)
        vmax=max(float(rd(q['x']).max()),float(rd(target).max()) if target is not None else 0.,1e-20)
        plot(dest/(row['id']+'.png'),output,f'Output | RG-SCS | {tag} epoch {ck["epoch"]} | {row["kind"]}',
             int(row['core_start_one_based']),vmax)
        rec=dict(id=row['id'],kind=row['kind'],source_id=row['source_id'],scr_support_db=row['scr_support_db'],
                 solve_residual=float(res.max()))
        if row['kind']=='synthetic':
            rec.update(calculate(q['x'],output,target))
            rec['delta_scr_db']=rec['scr_gain_db'];rec['delta_mcp_db']=-rec['mcp_ratio_db'] if rec['mcp_ratio_db'] is not None else None
            freq=float(np.angle(np.sum(target[1:]*target[:-1].conj(),dtype=np.complex128))*2000/(2*np.pi))
            rec['target_frequency_hz']=freq;rec['target_sign']='positive' if freq>=0 else 'negative'
            with torch.no_grad():
                rec.update(component_metrics(m,x,target,rel))
                mirror=m.relations(x.conj(),to('refs_left').conj(),to('refs_right').conj())
                rec['background_conjugate_coeff_max_error']=max(float((mirror[k]-rel[k].conj()).abs().max()) for k in ['a','c'])
                ym,rm=m.solve(x.conj(),mirror)
                rec['mirror_psnr_db']=calculate(q['x'].conj(),ym[0].cpu().numpy(),target.conj())['psnr_db']
                rec['output_conjugate_relative_error']=float((ym-y.conj()).norm()/y.norm().clamp_min(1e-20))
        elif row['kind']=='background':
            rec['background_leak_rate']=float(np.sum(abs(output)**2)/max(np.sum(abs(q['x'])**2),1e-30))
        records.append(rec)
        if (i+1)%50==0:print(json.dumps(dict(event='inference_progress',checkpoint=tag,completed=i+1,total=len(rr))),flush=True)
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (dest/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    syn=[r for r in records if r['kind']=='synthetic']
    selected=['input_scr_db','scr_db','input_mcp','mcp','target_energy_retention','delta_scr_db','delta_mcp_db','psnr_db','target_keep_error','background_leak_rate',
              'display_positive_background_suppression_db','display_negative_background_suppression_db',
              'full_positive_background_suppression_db','full_negative_background_suppression_db',
              'mirror_psnr_db','background_conjugate_coeff_max_error','output_conjugate_relative_error']
    groups={'all':syn,'positive_target':[r for r in syn if r['target_sign']=='positive'],
            'negative_target':[r for r in syn if r['target_sign']=='negative'],
            'low_scr_below_minus10':[r for r in syn if float(r['scr_support_db'])< -10]}
    groups['negative_low_scr_below_minus10']=[r for r in syn if r['target_sign']=='negative' and float(r['scr_support_db'])< -10]
    stats={}
    for name,group in groups.items():
        stats[name]={'count':len(group)}
        for key in selected:
            valid=[r[key] for r in group if r.get(key) is not None and np.isfinite(r[key])]
            stats[name][key]=float(np.mean(valid)) if valid else None
            stats[name][key+'_valid_count']=len(valid)
    summary=dict(epoch=ck['epoch'],checkpoint=tag,checkpoint_sha256=sha,test_count=len(rr),groups=stats,seconds=time.time()-start)
    from PIL import Image
    for row in rr:
        with Image.open(dest/(row['id']+'.png')) as im:im.verify()
    assert len(list(dest.glob('*.png')))==len(rr) and not list(dest.rglob('*.npz'))
    dump(dest/'summary.json',summary)
    dump(dest/'completed.json',dict(success=True,png_count=len(rr),checkpoint_sha256=sha,manifest_sha256=manifest_hash,data_root=dataset_id))
    return summary
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/rg-scs')
    parser.add_argument('--tag', choices=['best', 'last'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'cuda:0'], default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'inference')
    args = parser.parse_args()
    print(json.dumps(infer(args.run_dir, args.tag, args.data_root, args.device, args.output_dir), indent=2))

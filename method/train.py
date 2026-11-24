"""250 epochs; immutable original split; best selected on train+conjugate train only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,math,random,hashlib,gc,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from model import RGSCS
from losses import objective
from augmentation import move
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dataset_reader import RadarDataset

class CachedTrain(Dataset):
    def __init__(self):
        ds=RadarDataset('train',root=ROOT)
        with ThreadPoolExecutor(max_workers=4) as pool:self.data=list(pool.map(ds.__getitem__,range(len(ds))))
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i]

def save(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def log(**kw):print(json.dumps(kw,allow_nan=False),flush=True)
def status(**kw):
    p=ROOT/'reports/training_status.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(updated_unix=time.time(),**kw),indent=2,allow_nan=False));os.replace(t,p)
def check(diag):
    if not all(bool(torch.isfinite(v).all()) for v in diag.values()):raise FloatingPointError('Nonfinite diagnostic')
    if float(diag.get('calibration_change',0))>.75001:raise FloatingPointError('Calibration exceeds trust region')
    if float(diag.get('calibration_objective_delta',0))>1e-5:raise FloatingPointError('Calibration worsened its objective')
    if float(diag.get('solve_residual_max',0))>5e-5:raise FloatingPointError('Recovery solve inaccurate')

def main(resume=False):
    cfg=json.loads((ROOT/'method/config.json').read_text());run=ROOT/'runs/rg-scs';run.mkdir(parents=True,exist_ok=True)
    if (run/'last.pt').exists() and not resume:raise RuntimeError('Use --resume for an existing run')
    if resume and not (run/'last.pt').exists():raise RuntimeError('No checkpoint to resume')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    model=RGSCS(cfg).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=None if resume and cfg.get('continuation_schedule')=='constant_restored_lr' else torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=cfg['learning_rate']*.1)
    prov={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'method').glob('*.py')}
    prov['manifest']=hashlib.sha256((ROOT/'reports/manifest.csv').read_bytes()).hexdigest()
    checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False) if resume else None
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        if cfg.get('continuation_schedule')=='constant_restored_lr':
            scheduler=torch.optim.lr_scheduler.ConstantLR(opt,factor=1.0,total_iters=cfg['epochs'])
            if checkpoint['config'].get('continuation_schedule')=='constant_restored_lr':
                scheduler.load_state_dict(checkpoint['scheduler'])
            assert all(abs(g['lr']-cfg['continuation_learning_rate'])<1e-12 for g in opt.param_groups)
        else:
            scheduler.load_state_dict(checkpoint['scheduler'])
    if scheduler is None:raise RuntimeError('Continuation config requires --resume')
    log(event='optimizer_restored',learning_rates=[g['lr'] for g in opt.param_groups],schedule=cfg.get('continuation_schedule','cosine'))
    log(event='resume' if resume else 'start',resumed_epoch=checkpoint['epoch'] if checkpoint else None,
        previous_batch_size=checkpoint['config']['batch_size'] if checkpoint else None,config=cfg,provenance=prov,gpu=torch.cuda.get_device_name(0),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),test_used=False)
    data=CachedTrain();assert len(data)==3497
    args=dict(batch_size=cfg['batch_size'],num_workers=cfg['num_workers'],pin_memory=True,persistent_workers=True)
    train=DataLoader(data,shuffle=True,**args);evaluate=DataLoader(data,shuffle=False,**args)
    best=checkpoint['best_score'] if checkpoint else math.inf
    best_epoch=checkpoint['best_epoch'] if checkpoint else None
    start_epoch=checkpoint['epoch']+1 if checkpoint else 1
    if checkpoint:
        torch.set_rng_state(checkpoint['rng_cpu'])
        torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
        np.random.set_state(checkpoint['rng_numpy']);random.setstate(checkpoint['rng_python'])
        if checkpoint['config']['batch_size']!=cfg['batch_size']:
            # Batchwise sign means change slightly: reselect best under the new batching.
            best=math.inf;best_epoch=None
            log(event='best_selection_reset',reason='batch_size_changed',retained_checkpoint_epoch=checkpoint['epoch'])
    del checkpoint
    status(state='running',epoch=start_epoch-1,total_epochs=cfg['epochs'],pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for epoch in range(start_epoch,cfg['epochs']+1):
        begin=time.time();model.train();acc={};count=0;maxres=0.
        for step,batch in enumerate(train):
            q=move(batch,augment=True);opt.zero_grad(set_to_none=True)
            loss,diag=objective(model,q,epoch);check(diag)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            if not torch.isfinite(gn):raise FloatingPointError('Nonfinite gradient')
            opt.step();num=len(q['x']);count+=num
            for key,val in diag.items():acc[key]=acc.get(key,0.)+float(val)*num
            maxres=max(maxres,float(diag.get('solve_residual_max',0.)))
            if epoch==start_epoch or step==0:
                log(event='startup_batch',epoch=epoch,step=step+1,loss=float(loss),gradient_norm=float(gn),peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30)
        scheduler.step();training={k:v/count for k,v in acc.items()};selection=None
        if epoch>=cfg['best_start_epoch']:
            model.eval();acc={};count=0
            with torch.no_grad():
                for batch in evaluate:
                    for mirror,target_mirror in [(False,False),(False,True),(True,False),(True,True)]:
                        q=move(batch,mirror=mirror,target_mirror=target_mirror);loss,diag=objective(model,q,epoch,full=True);check(diag)
                        num=len(q['x']);count+=num
                        for key,val in diag.items():acc[key]=acc.get(key,0.)+float(val)*num
                        maxres=max(maxres,float(diag.get('solve_residual_max',0.)))
            selection={k:v/count for k,v in acc.items()}
            if selection['loss']<best:best=selection['loss'];best_epoch=epoch
        payload=dict(epoch=epoch,config=cfg,model=model.state_dict(),optimizer=opt.state_dict(),
                     scheduler=scheduler.state_dict(),best_score=best,best_epoch=best_epoch,
                     train=training,selection=selection,provenance=prov,
                     rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
                     rng_numpy=np.random.get_state(),rng_python=random.getstate())
        save(run/'last.pt',payload)
        if best_epoch==epoch:save(run/'best.pt',payload)
        seconds=time.time()-begin
        log(event='epoch',epoch=epoch,total_epochs=cfg['epochs'],learning_rate=opt.param_groups[0]['lr'],best_epoch=best_epoch,
            best_score=best if math.isfinite(best) else None,seconds=seconds,train=training,selection=selection,
            solve_residual_max=maxres,peak_gpu_gb=torch.cuda.max_memory_allocated()/2**30)
        status(state='running',epoch=epoch,total_epochs=cfg['epochs'],best_epoch=best_epoch,epoch_seconds=seconds,pid=os.getpid())
    del train,evaluate,data,opt,model,payload;gc.collect();torch.cuda.empty_cache()
    status(state='training_complete_inference_running',epoch=cfg['epochs'])
    log(event='training_complete',epoch=cfg['epochs'])
    from infer import infer
    for tag in ['last','best']:
        summary=infer(run,tag);log(event='inference_complete',checkpoint=tag,summary=summary)
    status(state='complete',epoch=cfg['epochs'],best_and_last_inference=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:main(args.resume)
    except BaseException as e:
        status(state='failed',error=repr(e),pid=os.getpid());raise

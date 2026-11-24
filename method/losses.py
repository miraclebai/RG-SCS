"""Same S/B component supervision, full-band balanced fit/leak, sign-balanced keep."""
import torch
from torch.nn import functional as F
from model import LAGS,q_lag,at_apply,ab_apply,coverage_response,band_power

def energy(z):return z.abs().square().mean((1,2))
def masked_mean(x,m):return (x*m).sum()/m.sum().clamp_min(1)
def sign_mean(x,batch):
    p=batch['is_positive'].float();f=batch['relation_frequency_hz']
    masks=[p*(f>=0),p*(f<0)]
    active=torch.stack([(m.sum()>0).to(x.dtype) for m in masks])
    means=torch.stack([masked_mean(x,m) for m in masks])
    return (means*active).sum()/active.sum().clamp_min(1)
def response_target(y,rel):
    return sum(z.abs().square().sum((1,2)) for z in at_apply(y,rel['omega']))/y.abs().square().sum((1,2)).clamp_min(1e-10)
def response_background(y,rel,beta):
    return sum(z.abs().square().sum((1,2)) for z in ab_apply(y,rel['a'],rel['c'],beta))/y.abs().square().sum((1,2)).clamp_min(1e-10)
def reference_fit(ref,coeff,cfg):
    a=coeff[:,8:,:8,None].transpose(2,3).expand(-1,-1,ref.shape[2],-1)
    c=coeff[:,:,8:,None].transpose(2,3).expand(-1,-1,ref.shape[2]-2,-1)
    return coverage_response(ab_apply(ref,a,c,cfg['spatial_weight']),ref,cfg)
def objective(model,batch,epoch,full=False):
    cfg=model.config;p=batch['is_positive'].float();s=batch['s'];b=batch['b'];x=batch['x']
    warmup=epoch<=cfg['relation_warmup_epochs'] and not full
    rel=model.relations(x,batch['refs_left'],batch['refs_right'],batch['frequency_bin'].long() if warmup else None)
    truth=batch['frequency_bin'].long().clamp_min(0)
    ce=sign_mean(F.cross_entropy(rel['logits'],truth,reduction='none'),batch)
    off=rel['offsets'].gather(1,truth[:,None]).squeeze(1)
    off=sign_mean(F.smooth_l1_loss(off/(2000/256/2),batch['frequency_offset_hz']/(2000/256/2),reduction='none'),batch)
    phase=0.
    for lag in LAGS:
        weight=batch[f'phase_weight_lag{lag}'];q=q_lag(rel['omega'],lag)
        per=((q-batch[f'phase_lag{lag}']).abs().square()*weight).sum(1)/weight.sum(1).clamp_min(1e-10)
        phase+=sign_mean(per,batch)/len(LAGS)
    ts=response_target(s,rel);tb=response_target(b,rel)
    bs=response_background(s,rel,cfg['spatial_weight']);bb=response_background(b,rel,cfg['spatial_weight'])
    bb_bal=coverage_response(ab_apply(b,rel['a'],rel['c'],cfg['spatial_weight']),b,cfg)
    ref=(reference_fit(batch['refs_left'],rel['left_coeff'],cfg)+reference_fit(batch['refs_right'],rel['right_coeff'],cfg))/2
    cross=sign_mean(F.relu(cfg['margin']+ts-tb)+F.relu(cfg['margin']+bb-bs),batch)
    smooth=(rel['background_controls'][:,:,1:]-rel['background_controls'][:,:,:-1]).abs().square().mean()
    smooth+=(rel['drift_controls'][:,:,1:]-rel['drift_controls'][:,:,:-1]).square().mean()/64
    terms=dict(frequency_ce=ce,frequency_offset=off,phase=phase,target_fit=sign_mean(ts,batch),
               background_fit=bb_bal.mean(),reference_fit=ref.mean(),cross=cross,smooth=smooth)
    diag=dict(background_response=bb.mean(),background_on_target=masked_mean(bs,p),
              frequency_mae=masked_mean((rel['frequency']-batch['relation_frequency_hz']).abs(),p))
    diag.update({k:v.mean() for k,v in rel.items() if k.startswith('calibration_')})
    if not warmup:
        double={k:torch.cat([rel[k],rel[k]],0) for k in ['omega','a','c']}
        parts,res=model.solve(torch.cat([s,b],0),double);gs,gb=parts.chunk(2);out=gs+gb
        sp=energy(s);bp=energy(b);floor=(1e-6*bp).clamp_min(1e-10)
        perkeep=energy(gs-s)/(sp+floor)
        leak=coverage_response([gb],b,cfg)
        recovery=energy(out-s)/torch.where(p.bool(),sp+floor,bp+1e-10)
        # The same mixed-input target relation is used for both background alternatives.
        prior_rel=dict(omega=double['omega'],a=torch.cat([rel['prior_a']]*2),c=torch.cat([rel['prior_c']]*2))
        prior_parts,prior_res=model.solve(torch.cat([s,b],0),prior_rel)
        gs0,gb0=prior_parts.chunk(2)
        keep0=energy(gs0-s)/(sp+floor)
        leak0=coverage_response([gb0],b,cfg)
        rec0=energy(gs0+gb0-s)/torch.where(p.bool(),sp+floor,bp+1e-10)
        w=cfg['weights']
        prior_task=w['keep']*sign_mean(keep0,batch)+w['leak']*leak0.mean()+w['recovery']*rec0.mean()
        # Detached comparison prevents hinge gradients from worsening the prior branch.
        # Independent prior_task supervision gives that baseline a useful task objective.
        calibration_task=F.relu(leak-leak0.detach()).mean()+sign_mean(F.relu(perkeep-keep0.detach()),batch)
        terms.update(prior_task=prior_task,calibration_task=calibration_task)
        diag.update(prior_keep_error=masked_mean(keep0,p),prior_background_leak=(energy(gb0)/(bp+1e-10)).mean(),
                    calibration_core_leak_worsen=(leak>leak0).float().mean(),
                    calibration_core_keep_worsen=masked_mean((perkeep>keep0).float(),p))
        res=torch.cat([res,prior_res])
        terms.update(keep=sign_mean(perkeep,batch),leak=leak.mean(),recovery=recovery.mean())
        diag.update(solve_residual_max=res.max(),solve_residual_mean=res.mean(),
                    target_keep_error=masked_mean(perkeep,p),background_leak_ratio=(energy(gb)/(bp+1e-10)).mean())
        bi=band_power(b);bo=band_power(gb)
        for i,tag in [(1,'positive'),(2,'negative')]:
            diag['background_leak_'+tag]=(bo[i]/bi[i].clamp_min(cfg['band_energy_floor']*bi[0].clamp_min(1e-10))).mean()
            mask=p*((batch['relation_frequency_hz']>=0) if i==1 else (batch['relation_frequency_hz']<0))
            diag['keep_'+tag]=masked_mean(perkeep,mask)
            diag['target_count_'+tag]=mask.sum()
    loss=sum(cfg['weights'][k]*v for k,v in terms.items())
    return loss,{k:v.detach() for k,v in dict(loss=loss,**terms,**diag).items()}

"""Whole-band energy coverage and reference-only calibration.
"""
import torch
from torch.nn import functional as F
from base_model import *
from base_model import RelationModel

def frequency_masks(n,device):
    f=torch.fft.fftfreq(n,device=device)
    return f>0,(f<0)&(f>-.5)
def band_power(z):
    p=torch.fft.fft(z,dim=1,norm='ortho').abs().square()
    pos,neg=frequency_masks(z.shape[1],z.device)
    return p.sum((1,2)),p[:,pos].sum((1,2)),p[:,neg].sum((1,2))
def coverage_power(z,bands):
    p=torch.fft.fft(z,dim=1,norm='ortho').abs().square().sum(2)
    f=torch.fft.fftfreq(z.shape[1],device=z.device,dtype=p.dtype)
    centers=torch.arange(bands,device=z.device,dtype=p.dtype)/bands
    distance=torch.remainder(f[:,None]-centers[None]+.5,1)-.5
    w=(1-distance.abs()*bands).clamp_min(0)
    w=w/w.sum(1,keepdim=True).clamp_min(1e-12)
    return p@w
def coverage_response(outputs,reference,cfg):
    inp=coverage_power(reference,cfg['coverage_bands'])
    out=sum(coverage_power(v,cfg['coverage_bands']) for v in outputs)
    total=inp.sum(1).clamp_min(1e-10)
    glob=out.sum(1)/total
    floor=cfg['coverage_floor']*total[:,None]/cfg['coverage_bands']
    ratio=out/(inp+floor).clamp_min(1e-10)
    tail=F.relu(ratio-cfg['coverage_threshold']).topk(cfg['coverage_topk'],dim=1).values.mean(1)
    return glob+cfg['coverage_weight']*tail
# Kept solely for compatible inference/diagnostic imports; not used by v2 training.
def balanced_response(outputs,reference,alpha=.5,floor=.05):
    inp=band_power(reference);parts=[band_power(z) for z in outputs]
    out=[sum(p[j] for p in parts) for j in range(3)]
    total=inp[0].clamp_min(1e-10)
    return (1-alpha)*out[0]/total+alpha*sum(out[j]/inp[j].clamp_min(floor*total) for j in (1,2))/2

class AdaptiveBackground(BackgroundRelations):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.register_buffer('basis',F.interpolate(torch.eye(4)[None],size=256,mode='linear',align_corners=True)[0].T)
    def symmetric_prior(self,z):
        return (self.side(z)+self.side(z.conj()).conj())*.5
    def statistics(self,z):
        z=z.to(torch.complex128);target=z[:,8:]
        past=torch.stack([z[:,8-k:256-k] for k in range(1,9)],-1)
        # Statistics are reference-only. Their scale cancels in the normalized objective.
        power=z.abs().square().mean((1,2)).clamp_min(1e-12)
        cov=torch.einsum('btri,btrj->btij',past.conj(),past)/z.shape[2]
        h=(past.conj()*target[...,None]).mean(2)
        p=target.abs().square().mean(2)
        return cov,h,p,power
    def fit_and_grad(self,a,a0,stats):
        cov,h,p,power=stats
        basis=self.basis[8:].to(a.dtype)
        at=torch.einsum('tk,bki->bti',basis,a)
        ca=torch.einsum('btij,btj->bti',cov,at)
        residual=p-2*(at.conj()*h).sum(2).real+(at.conj()*ca).sum(2).real
        den=1+at.abs().square().sum(2)
        factor=(1-self.cfg['spatial_weight'])/power
        fit=(residual/den).mean(1)*factor
        eta=self.cfg['calibration_eta'];delta=a-a0
        obj=fit+eta*delta.abs().square().sum((1,2))
        gt=2*((ca-h)/den[...,None]-residual[...,None]*at/den[...,None].square())
        grad=torch.einsum('tk,bti->bki',basis,gt)*factor[:,None,None]/at.shape[1]+2*eta*delta
        return fit,obj,grad
    def calibrate(self,z,prior):
        stats=self.statistics(z);a0=prior[...,:8].to(torch.complex128);a=a0
        before,obj,_=self.fit_and_grad(a,a0,stats)
        radius=self.cfg['calibration_radius']
        accepts=torch.zeros(len(z),device=z.device,dtype=torch.float64)
        for _ in range(self.cfg['calibration_steps']):
            _,current,grad=self.fit_and_grad(a,a0,stats)
            # Differentiate through the chosen projected gradient step.
            # Discrete backtracking decisions themselves need no autograd graph.
            with torch.no_grad():
                accepted=torch.zeros_like(current,dtype=torch.bool)
                steps=torch.zeros_like(current)
                for j in range(6):
                    step=self.cfg['calibration_step_size']/(2**j)
                    delta=a-step*grad-a0
                    scale=(radius/delta.abs().square().sum((1,2)).sqrt().clamp_min(1e-12)).clamp_max(1)
                    trial=a0+delta*scale[:,None,None]
                    _,candidate,_=self.fit_and_grad(trial,a0,stats)
                    use=(~accepted)&(candidate<=current)&torch.isfinite(candidate)
                    steps=torch.where(use,step,steps);accepted|=use
            delta=a-steps[:,None,None]*grad-a0
            scale=(radius/delta.abs().square().sum((1,2)).sqrt().clamp_min(1e-12)).clamp_max(1)
            a=a0+delta*scale[:,None,None];accepts=accepts+accepted
        after,final,grad=self.fit_and_grad(a,a0,stats)
        return a.to(prior.dtype),dict(calibration_fit_before=before,calibration_fit_after=after,
            calibration_objective_delta=final-obj,
            calibration_change=(a-a0).abs().square().sum((1,2)).sqrt(),
            calibration_accepted_steps=accepts,
            calibration_gradient_norm=grad.abs().square().sum((1,2)).sqrt())
    def expand_pair(self,ctrl):
        lc,rc=ctrl.chunk(2);l=self.expand(lc);r=self.expand(rc)
        w=(torch.arange(24,40,device=l.device,dtype=l.real.dtype)-3.5)/56
        coeff=l[:,:,None,:]*(1-w[None,None,:,None])+r[:,:,None,:]*w[None,None,:,None]
        return dict(a=coeff[:,8:,:,:8],c=coeff[:,:,1:-1,8:],left_coeff=l,right_coeff=r,
                    background_controls=torch.stack([lc,rc],1))
    def forward(self,left,right):
        z=torch.cat([left,right],0);prior=self.symmetric_prior(z)
        a,diag=self.calibrate(z,prior)
        out=self.expand_pair(torch.cat([a,prior[...,8:]],-1))
        p=self.expand_pair(prior)
        out.update(prior_a=p['a'],prior_c=p['c'])
        out.update({k:(v[:len(left)]+v[len(left):])/2 for k,v in diag.items()})
        return out
class RGSCS(RelationModel):
    def __init__(self,config):
        super().__init__(config);self.background=AdaptiveBackground(config)

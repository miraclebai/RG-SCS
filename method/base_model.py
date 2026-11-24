"""RGSCS: learned complex relations, matrix-free Hermitian solve, implicit gradients."""
import math
import torch
from torch import nn
from torch.nn import functional as F

LAGS=(1,4,16)

def pad_time(z,left,right):
    return F.pad(z,(0,0,left,right))

def q_lag(omega,lag):
    phase=omega.unfold(1,lag,1).sum(-1)
    return torch.polar(torch.ones_like(phase),phase)

def at_apply(y,omega):
    n=y.shape[1]
    return [(y[:,l:]-q_lag(omega,l)[...,None]*y[:,:-l])*math.sqrt(n/(3*(n-l)*2)) for l in LAGS]

def at_gram(y,omega):
    n=y.shape[1];out=torch.zeros_like(y)
    for l in LAGS:
        q=q_lag(omega,l)[...,None]
        d=(y[:,l:]-q*y[:,:-l])*(n/(3*(n-l)*2))
        out=out+pad_time(d,l,0)-pad_time(q.conj()*d,0,l)
    return out

def ab_apply(y,a,c,beta=.25):
    n,r=y.shape[1:];p=a.shape[-1]
    an=(1+a.abs().square().sum(-1)).sqrt()
    dt=y[:,p:]
    for k in range(1,p+1):dt=dt-a[...,k-1]*y[:,p-k:n-k]
    dt=dt/an*math.sqrt((1-beta)*n/(n-p))
    cn=(1+c.abs().square().sum(-1)).sqrt()
    ds=(y[:,:,1:-1]-c[...,0]*y[:,:,:-2]-c[...,1]*y[:,:,2:])/cn
    ds=ds*math.sqrt(beta*r/(r-2))
    return dt,ds

def ab_gram(y,a,c,beta=.25):
    n,r=y.shape[1:];p=a.shape[-1]
    an=1+a.abs().square().sum(-1)
    dt=y[:,p:]
    for k in range(1,p+1):dt=dt-a[...,k-1]*y[:,p-k:n-k]
    dt=dt/an*((1-beta)*n/(n-p))
    out=pad_time(dt,p,0)
    for k in range(1,p+1):out=out-pad_time(a[...,k-1].conj()*dt,p-k,k)
    ds=(y[:,:,1:-1]-c[...,0]*y[:,:,:-2]-c[...,1]*y[:,:,2:])/(1+c.abs().square().sum(-1))
    ds=ds*(beta*r/(r-2))
    out=out+F.pad(ds,(1,1))-F.pad(c[...,0].conj()*ds,(0,2))-F.pad(c[...,1].conj()*ds,(2,0))
    return out

def h_apply(y,omega,a,c,lam,eps,beta):
    return at_gram(y,omega)+lam*ab_gram(y,a,c,beta)+eps*y

def diagonal(omega,a,c,lam,eps,beta):
    n=omega.shape[1]+1;r=a.shape[2];p=a.shape[-1]
    d=torch.zeros_like(a[...,0].real)
    full=pad_time(d,p,0)+eps
    for l in LAGS:
        v=torch.ones_like(full[:,:-l])*(n/(3*(n-l)*2))
        full=full+pad_time(v,l,0)+pad_time(v,0,l)
    w=lam*(1-beta)*n/(n-p)/(1+a.abs().square().sum(-1))
    full=full+pad_time(w,p,0)
    for k in range(1,p+1):full=full+pad_time(w*a[...,k-1].abs().square(),p-k,k)
    w=lam*beta*r/(r-2)/(1+c.abs().square().sum(-1))
    full=full+F.pad(w,(1,1))+F.pad(w*c[...,0].abs().square(),(0,2))+F.pad(w*c[...,1].abs().square(),(2,0))
    return full

def inner(x,y):return (x.conj()*y).real.sum((1,2))

# Replaced by compiled callable after numerical checks, when explicitly enabled.
H_CALL=h_apply
AB_CALL=ab_gram

def prepare_h(omega,a,c,lam,eps,beta):
    """Exact sparse Hermitian bands, prepared once per linear solve."""
    n=omega.shape[1]+1;p=a.shape[-1];r=a.shape[2]
    d=diagonal(omega,a,c,lam,eps,beta)
    wt=lam*(1-beta)*n/(n-p)/(1+a.abs().square().sum(-1))
    v=[torch.ones_like(a[...,0])]+[-a[...,k] for k in range(p)]
    bands={}
    for lag in sorted(set(range(1,p+1))|set(LAGS)):
        h=torch.zeros_like(d[:,lag:]).to(a.dtype)
        if lag in LAGS:h=h-q_lag(omega,lag)[...,None]*(n/(3*(n-lag)*2))
        if lag<=p:
            for i in range(p-lag+1):
                h=h+pad_time(wt*v[i].conj()*v[i+lag],p-i-lag,i)
        bands[lag]=h
    wr=lam*beta*r/(r-2)/(1+c.abs().square().sum(-1))
    s1=-F.pad(wr*c[...,0],(0,1))-F.pad(wr*c[...,1].conj(),(1,0))
    s2=wr*c[...,1].conj()*c[...,0]
    return d,bands,s1,s2

def prepared_apply(y,prepared):
    d,bands,s1,s2=prepared;out=d*y
    for lag,h in bands.items():
        out=out+pad_time(h*y[:,:-lag],lag,0)+pad_time(h.conj()*y[:,lag:],0,lag)
    out=out+F.pad(s1*y[:,:,:-1],(1,0))+F.pad(s1.conj()*y[:,:,1:],(0,1))
    return out+F.pad(s2*y[:,:,:-2],(2,0))+F.pad(s2.conj()*y[:,:,2:],(0,2))

def _pcg_once(rhs,omega,a,c,lam,eps,beta,maxiter,tol,x0=None,prepared=None):
    prep=prepare_h(omega,a,c,lam,eps,beta) if prepared is None else prepared;inv=prep[0].reciprocal()
    x=torch.zeros_like(rhs) if x0 is None else x0.clone()
    r=rhs-prepared_apply(x,prep);z=r*inv;p=z.clone()
    rz=inner(r,z);rhsnorm=rhs.abs().square().sum((1,2)).sqrt()
    active=rhsnorm>1e-20
    for k in range(maxiter):
        hp=prepared_apply(p,prep);denom=inner(p,hp)
        alpha=torch.where(active,rz/denom.clamp_min(torch.finfo(rhs.real.dtype).tiny),torch.zeros_like(rz))
        x=x+alpha[:,None,None]*p;r=r-alpha[:,None,None]*hp
        refresh=(k+1)%(256 if rhs.dtype==torch.complex128 else 64)==0
        if refresh:r=rhs-prepared_apply(x,prep)
        active=r.abs().square().sum((1,2)).sqrt()>tol*rhsnorm.clamp_min(1e-20)
        if (k+1)%16==0 and not bool(active.any()):break
        z=r*inv;rznew=inner(r,z)
        coeff=torch.where(active,rznew/rz.clamp_min(torch.finfo(rhs.real.dtype).tiny),torch.zeros_like(rz))
        p=z if refresh else z+coeff[:,None,None]*p
        rz=rznew
    residual=(rhs-prepared_apply(x,prep)).abs().square().sum((1,2)).sqrt()/rhsnorm.clamp_min(1e-20)
    return x,residual

def pcg(rhs,omega,a,c,lam,eps,beta,maxiter,tol):
    x,res=_pcg_once(rhs,omega,a,c,lam,eps,beta,maxiter,tol)
    # Only difficult samples receive extra work; never silently accept a fixed
    # iteration truncation as a converged solution. Double refinement handles
    # float32 cancellation near the regularized joint nullspace.
    bad=res>tol*2
    if bool(bad.any()):
        use_double=rhs.dtype==torch.complex64
        rr=rhs[bad].to(torch.complex128) if use_double else rhs[bad]
        oo=omega[bad].double() if use_double else omega[bad]
        aa=a[bad].to(rr.dtype);cc=c[bad].to(rr.dtype)
        native=prepare_h(omega[bad],a[bad],c[bad],lam,eps,beta)
        exact=(native[0].double(),{k:v.to(rr.dtype) for k,v in native[1].items()},native[2].to(rr.dtype),native[3].to(rr.dtype)) if use_double else native
        yy,_=_pcg_once(rr,oo,aa,cc,lam,eps,beta,2048,min(tol*.1,1e-8),x[bad].to(rr.dtype),prepared=exact)
        x=x.clone();x[bad]=yy.to(rhs.dtype)
    # Residual is checked using the original operator expressions as well.
    res=(rhs-h_apply(x,omega,a,c,lam,eps,beta)).abs().square().sum((1,2)).sqrt()/rhs.abs().square().sum((1,2)).sqrt().clamp_min(1e-20)
    if bool((res>max(5*tol,1e-5)).any()):
        raise FloatingPointError(f'PCG failed after adaptive refinement: {float(res.max()):.6g}')
    return x,res

class ImplicitSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,omega,a,c,lam,eps,beta,maxiter,tol):
        rhs=lam*AB_CALL(x,a,c,beta)
        y,res=pcg(rhs,omega,a,c,lam,eps,beta,maxiter,tol)
        ctx.save_for_backward(x,omega,a,c,y)
        ctx.options=(lam,eps,beta,maxiter,tol)
        ctx.mark_non_differentiable(res)
        return y,res
    @staticmethod
    def backward(ctx,grad_y,grad_res):
        x,omega,a,c,y=ctx.saved_tensors
        lam,eps,beta,maxiter,tol=ctx.options
        with torch.no_grad():u,res=pcg(grad_y,omega,a,c,lam,eps,beta,maxiter,tol)
        # Differentiate the optimality equation, not a truncated solver graph.
        with torch.enable_grad():
            xx=x.detach().requires_grad_(True);oo=omega.detach().requires_grad_(True)
            aa=a.detach().requires_grad_(True);cc=c.detach().requires_grad_(True)
            residual=lam*ab_gram(xx-y,aa,cc,beta)-at_gram(y,oo)-eps*y
            grads=torch.autograd.grad(residual,(xx,oo,aa,cc),u,allow_unused=False)
        return (*grads,None,None,None,None,None)

def time_features(z):
    power=z.abs().square();scale=power.mean((1,2),keepdim=True).sqrt().clamp_min(1e-6)
    zn=z/scale
    channels=[zn.real,zn.imag,torch.log1p(zn.abs())]
    floor=.01*power.mean((1,2),keepdim=True)+1e-8
    for l in LAGS:
        v=z[:,l:]*z[:,:-l].conj()/(z[:,l:].abs()*z[:,:-l].abs()+floor)
        v=pad_time(v,l,0);channels.extend([v.real,v.imag])
    return torch.stack(channels,1)

class ConvBlock(nn.Module):
    def __init__(self,cin,cout,stride=(1,1)):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,stride=stride),nn.GroupNorm(4,cout),nn.SiLU())
    def forward(self,x):return self.net(x)

class TargetRelations(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectral=nn.Sequential(ConvBlock(9,24),ConvBlock(24,32),ConvBlock(32,32))
        self.freq_head=nn.Sequential(nn.Conv1d(64,32,3,padding=1),nn.SiLU(),nn.Conv1d(32,2,1))
        self.temporal=nn.Sequential(ConvBlock(9,24,(2,1)),ConvBlock(24,32,(2,1)),ConvBlock(32,32,(2,1)))
        self.controls=nn.Sequential(nn.Conv1d(64,32,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool1d(8),nn.Conv1d(32,1,1))
        self.register_buffer('frequencies',torch.fft.fftshift(torch.fft.fftfreq(256,d=1/2000)))
        self.register_buffer('hann',torch.hann_window(256,periodic=False)[None,:,None])
        nn.init.zeros_(self.freq_head[-1].weight);nn.init.zeros_(self.freq_head[-1].bias)
        nn.init.zeros_(self.controls[-1].weight);nn.init.zeros_(self.controls[-1].bias)
    def forward(self,x,left,right,teacher_bin=None):
        z=torch.fft.fftshift(torch.fft.fft(x*self.hann,dim=1,norm='ortho'),dim=1)
        zr=torch.fft.fftshift(torch.fft.fft(torch.cat([left,right],2)*self.hann,dim=1,norm='ortho'),dim=1)
        scale=zr.abs().square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-6)
        z=z/scale;zr=zr/scale
        amp=z.abs();ref=zr.abs().square().mean(2,keepdim=True).sqrt().expand_as(amp)
        cross=z[:,:,1:]*z[:,:,:-1].conj()/(amp[:,:,1:]*amp[:,:,:-1]+.01)
        cross=F.pad(cross,(1,0))
        f=torch.stack([z.real,z.imag,torch.log1p(amp),torch.log1p(ref),torch.log1p(amp)-torch.log1p(ref),cross.real,cross.imag,zr.real,zr.imag],1)
        h=self.spectral(f);h=torch.cat([h.mean(3),h.amax(3)],1)
        head=self.freq_head(h);logits=head[:,0];offsets=torch.tanh(head[:,1])*(2000/256/2)
        # Zero initialization starts from a spectral local-contrast score, but the
        # learned logits are unrestricted. This is not a target existence gate.
        contrast=(torch.log1p(amp)-torch.log1p(ref)).amax(2)
        logits=logits+contrast
        prob=logits.softmax(1);hard=F.one_hot(logits.argmax(1),256).to(prob.dtype)
        select=hard+prob-prob.detach() if self.training else hard
        if teacher_bin is not None:
            valid=teacher_bin>=0;truth=F.one_hot(teacher_bin.clamp_min(0),256).to(prob.dtype)
            select=torch.where(valid[:,None],truth,select)
        frequency=(select*(self.frequencies+offsets)).sum(1)
        h=self.temporal(time_features(x));h=torch.cat([h.mean(3),h.amax(3)],1)
        ctrl=8.*torch.tanh(self.controls(h));ctrl=ctrl-ctrl.mean(2,keepdim=True)
        drift=F.interpolate(ctrl,size=255,mode='linear',align_corners=True)[:,0]
        omega=(2*math.pi/2000)*(frequency[:,None]+drift)
        return dict(omega=omega,logits=logits,offsets=offsets,frequency=frequency,drift_controls=ctrl)

class BackgroundRelations(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(ConvBlock(9,24,(2,1)),ConvBlock(24,32,(2,1)),ConvBlock(32,32,(2,1)))
        self.head=nn.Sequential(nn.Conv1d(64,32,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool1d(4),nn.Conv1d(32,20,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def side(self,z):
        h=self.net(time_features(z));h=torch.cat([h.mean(3),h.amax(3)],1)
        v=self.head(h).transpose(1,2)
        # Bounded coefficients prevent arbitrarily large cancellation ratios.
        return torch.complex(torch.tanh(v[...,:10]),torch.tanh(v[...,10:]))*.7
    def expand(self,ctrl):
        return torch.complex(F.interpolate(ctrl.real.transpose(1,2),size=256,mode='linear',align_corners=True),F.interpolate(ctrl.imag.transpose(1,2),size=256,mode='linear',align_corners=True)).transpose(1,2)
    def forward(self,left,right):
        lc=self.side(left);rc=self.side(right);l=self.expand(lc);r=self.expand(rc)
        # Actual reference centers are 3.5 and 59.5 in the 64-cell context.
        w=(torch.arange(24,40,device=l.device,dtype=l.real.dtype)-3.5)/56
        coeff=l[:,:,None,:]*(1-w[None,None,:,None])+r[:,:,None,:]*w[None,None,:,None]
        return dict(a=coeff[:,8:,:,:8],c=coeff[:,:,1:-1,8:],left_coeff=l,right_coeff=r,background_controls=torch.stack([lc,rc],1))

class RelationModel(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.target=TargetRelations();self.background=BackgroundRelations()
    def relations(self,x,left,right,teacher_bin=None):
        return dict(**self.target(x,left,right,teacher_bin),**self.background(left,right))
    def solve(self,x,rel):
        c=self.config
        # The small neural encoders stay float32; ill-conditioned linear solves
        # and their implicit backward equations use full complex128 precision.
        return ImplicitSolve.apply(x.to(torch.complex128),rel['omega'].double(),rel['a'].to(torch.complex128),rel['c'].to(torch.complex128),c['lambda'],c['epsilon'],c['spatial_weight'],c['cg_iterations'],c['cg_tolerance'])
    def forward(self,x,left,right):
        rel=self.relations(x,left,right);y,res=self.solve(x,rel)
        return dict(s_hat=y,solve_residual=res,**rel)

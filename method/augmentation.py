"""Training-only synchronized IQ augmentation; no change to the stored dataset."""
import math
import torch
def move(batch,augment=False,mirror=False,target_mirror=False,device='cuda:0'):
    q={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
    size=len(q['x'])
    mask=torch.full((size,),mirror,dtype=torch.bool,device=device)
    tm=torch.full((size,),target_mirror,dtype=torch.bool,device=device)
    if augment:
        mask[torch.randperm(size,device=device)[:size//2]]=True
        tm[torch.randperm(size,device=device)[:size//2]]=True
    if 's' in q:
        tm=tm&q['is_positive']
        q['s']=torch.where(tm[:,None,None],q['s'].conj(),q['s'])
        q['x']=torch.where(tm[:,None,None],q['b']+q['s'],q['x'])
    else:tm=torch.zeros_like(mask)
    for key in ['x','s','b','refs_left','refs_right']:
        if key in q:q[key]=torch.where(mask[:,None,None],q[key].conj(),q[key])
    if 'relation_frequency_hz' in q:
        flip=mask^tm
        f=torch.where(flip,-q['relation_frequency_hz'],q['relation_frequency_hz'])
        q['relation_frequency_hz']=f
        grid=torch.fft.fftshift(torch.fft.fftfreq(256,d=1/2000,device=device))
        idx=(f[:,None]-grid).abs().argmin(1)
        q['frequency_bin']=torch.where(q['is_positive'],idx,torch.full_like(idx,-1))
        q['frequency_offset_hz']=f-grid[idx]
        for lag in [1,4,16]:
            key=f'phase_lag{lag}'
            q[key]=torch.where(flip[:,None],q[key].conj(),q[key])
    if augment:
        phase=torch.rand(size,1,1,device=device)*2*math.pi
        rotation=torch.polar(torch.ones_like(phase),phase)
        for key in ['x','s','b','refs_left','refs_right']:
            if key in q:q[key]=q[key]*rotation
    return q

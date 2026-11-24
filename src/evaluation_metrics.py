"""Fixed synthetic-test RD metrics; clean labels are used only for evaluation."""
import numpy as np
from scipy.ndimage import gaussian_filter1d,binary_dilation
from radar_common import rd,FB,PRF,BAND

def power(z):
    h=np.hanning(len(z))[:,None]
    return (abs(np.fft.fftshift(np.fft.fft(z.astype(np.complex128)*h,axis=0),axes=0)[BAND]).T**2/np.sum(h*h))

def regions(x,s):
    # Evaluation only: known additive background; never passed to a predictor.
    b=x.astype(np.complex128)-s.astype(np.complex128)
    pb=power(b);ps=power(s);spectrum=gaussian_filter1d(pb.mean(0),1.)
    peak=int(spectrum.argmax());lo=hi=peak
    while lo>0 and spectrum[lo-1]>=.1*spectrum[peak]:lo-=1
    while hi<len(spectrum)-1 and spectrum[hi+1]>=.1*spectrum[peak]:hi+=1
    target=ps>=.01*ps.max() # clean target -20dB power support
    guard=binary_dilation(target,structure=np.ones((3,5),bool))
    band=np.zeros_like(target);band[:,lo:hi+1]=True
    clutter=band & ~guard
    return target,clutter,dict(clutter_band_low_hz=float(FB[lo]),clutter_band_high_hz=float(FB[hi]),
       target_pixels=int(target.sum()),clutter_pixels=int(clutter.sum()))

def calculate(x,y,s):
    target,clutter,info=regions(x,s);ay=rd(y);ax=rd(x);truth=rd(s)
    scale=max(float(ax.max()),float(truth.max()),1e-20)
    info['psnr_db']=float(-10*np.log10(max(float(np.mean(((ay-truth)/scale).astype(np.float64)**2)),1e-30)))
    info['input_psnr_db']=float(-10*np.log10(max(float(np.mean(((ax-truth)/scale).astype(np.float64)**2)),1e-30)))
    py=power(y);px=power(x)
    for prefix,p in [('input_',px),('',py)]:
        pt=float(p[target].mean());pc=float(p[clutter].mean()) if clutter.any() else None
        info[prefix+'target_region_power']=pt;info[prefix+'mcp']=pc
        info[prefix+'scr_db']=float(10*np.log10(pt/pc)) if pc is not None and pc>0 and pt>0 else None
    if info['mcp'] is not None and info['mcp']>0 and info['input_mcp']>0:
        info['mcp_ratio_db']=float(10*np.log10(info['mcp']/info['input_mcp']))
    else:info['mcp_ratio_db']=None
    info['scr_gain_db']=info['scr_db']-info['input_scr_db'] if info['scr_db'] is not None and info['input_scr_db'] is not None else None
    info['metric_valid']=bool(clutter.any() and info['scr_db'] is not None)
    return info

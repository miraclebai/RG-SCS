"""Shared radar dataset geometry and physically coherent target simulation."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_filter1d
N=256; PRF=2000.; DR=2.5; WAVELENGTH=.032
CORE=slice(24,40); REF=np.r_[0:8,56:64]
F=np.fft.fftshift(np.fft.fftfreq(N,1/PRF)); BAND=(F>=-500)&(F<500)
FB=F[BAND]; HANN=np.hanning(N)[:,None]
ROOT=Path(__file__).resolve().parents[1]
def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,p)
def rd(z):
    return np.abs(np.fft.fftshift(np.fft.fft(z*HANN,axis=0),axes=0)[BAND]).T.astype("f4")
def unit(v):
    v=np.asarray(v,"f4").ravel()
    return v/max(float(np.linalg.norm(v)),1e-20)
def visual_features(z):
    a=rd(z);a/=max(float(a.max()),1e-20)
    a=gaussian_filter(a,(.45,.55))
    a=a.reshape(16,64,2).max(2)
    return unit(a),unit(np.maximum(a-.20,0))
def spectral_summary(z):
    p=np.mean(rd(z)**2,axis=0);p=gaussian_filter1d(p,1.)
    k=int(p.argmax());threshold=p[k]*.1
    lo=hi=k
    while lo>0 and p[lo-1]>=threshold:lo-=1
    while hi<len(p)-1 and p[hi+1]>=threshold:hi+=1
    cdf=np.cumsum(p);cdf/=max(float(cdf[-1]),1e-30)
    return dict(peak_hz=float(FB[k]),lobe_low_hz=float(FB[lo]),lobe_high_hz=float(FB[hi]),
                width10_hz=float(FB[hi]-FB[lo]),q05_hz=float(np.interp(.05,cdf,FB)),
                q95_hz=float(np.interp(.95,cdf,FB)))
def choose_frequency(rng,summary,zone):
    lo,hi=summary["lobe_low_hz"],summary["lobe_high_hz"];df=PRF/N
    if zone=="inside":options=[(max(-430.,lo),min(430.,hi))]
    elif zone=="edge":options=[(max(-430.,lo-2*df),min(430.,lo)),(max(-430.,hi),min(430.,hi+2*df))]
    else:options=[(-430.,min(430.,lo-3*df)),(max(-430.,hi+3*df),430.)]
    options=[(a,b) for a,b in options if b-a>2.]
    if not options:raise ValueError("No physically permitted frequency interval for requested zone")
    # Select side uniformly, then a continuous off-grid frequency.
    a,b=options[int(rng.integers(len(options)))]
    return float(rng.uniform(a,b))
def simulate(background,seed):
    """Retains legacy coherent scatterer motion/PSF equations; never modifies B."""
    rng=np.random.default_rng(seed)
    width=str(rng.choice(["compact","medium","extended"],p=[.25,.50,.25]))
    body=float(rng.uniform(*{"compact":(0.,2.),"medium":(5.,16.),"extended":(19.,34.)}[width]))
    sigma=float(rng.uniform(2.6,3.4))
    partial=bool(rng.random()<.12)
    # Peak may move within the core; full-target samples must subsequently pass
    # a retained-energy test instead of artificially truncating the waveform.
    center=float(rng.uniform(24.0,39.0) if partial else rng.uniform(27.,36.))
    summary=spectral_summary(background[:,CORE])
    zone=str(rng.choice(["inside","edge","outside"],p=[.20,.25,.55]))
    freq=choose_frequency(rng,summary,zone)
    velocity=freq*WAVELENGTH/2;acc=float(rng.uniform(-.4,.4))
    ns=1 if body<2.5 else int(rng.integers(4,9))
    offsets=np.zeros(1) if ns==1 else np.linspace(-body/2,body/2,ns)+rng.normal(0,.25,ns)
    amps=rng.uniform(.6,1.1,ns);phases=rng.uniform(-np.pi,np.pi,ns)
    micro_amp=rng.uniform(0,.003,ns);micro_hz=rng.uniform(.15,.7,ns);micro_phase=rng.uniform(-np.pi,np.pi,ns)
    t=(np.arange(N)-(N-1)/2)/PRF;r=np.arange(64)*DR
    target=np.zeros((N,64),dtype="c16")
    for j in range(ns):
        displacement=velocity*t+.5*acc*t*t+micro_amp[j]*np.sin(2*np.pi*micro_hz[j]*t+micro_phase[j])
        position=center*DR+offsets[j]+displacement
        psf=np.exp(-.5*((r[None,:]-position[:,None])/sigma)**2)
        phase=phases[j]+4*np.pi*displacement/WAVELENGTH
        target+=amps[j]*psf*np.exp(1j*phase[:,None])
    core_fraction=float(np.sum(np.abs(target[:,CORE])**2)/np.sum(np.abs(target)**2))
    if core_fraction<.35 or (not partial and core_fraction<.95):
        raise ValueError("Target not sufficiently contained for requested geometry")
    p=np.mean(np.abs(target[:,CORE])**2,axis=0)
    support=p>=p.max()*.01
    scr_bin=int(rng.choice(5,p=[.15,.25,.30,.20,.10]))
    scr=float(rng.uniform(*[(-18,-10),(-10,-4),(-4,2),(2,8),(8,16)][scr_bin]))
    clean=target[:,CORE][:,support];bg=background[:,CORE][:,support]
    gain=float(np.sqrt(10**(scr/10)*np.sum(np.abs(bg.astype("c16"))**2)/np.sum(np.abs(clean)**2)))
    target=(target*gain).astype("c8")
    leak=float(np.sum(np.abs(target[:,REF].astype("c16"))**2)/np.sum(np.abs(target.astype("c16"))**2))
    if leak>1e-6:raise ValueError("Target contaminates measured reference cells")
    tp=np.mean(np.abs(target[:,CORE].astype("c16"))**2,axis=0)
    p10=tp>=tp.max()*.1
    meta=dict(seed=int(seed),component_count=1,stress=False,width_class=width,body_extent_m=body,
        psf_sigma_m=sigma,psf_model="Legacy Gaussian amplitude PSF approximation; not a calibrated radar response",
        wavelength_m=WAVELENGTH,wavelength_status="Explicit inherited X-band assumption; carrier not documented",
        center_context_zero_based=center,center_core_zero_based=center-24,partial_target_requested=partial,
        core_target_energy_fraction=core_fraction,reference_target_energy_fraction=leak,
        frequency_hz=freq,frequency_zone=zone,background_spectrum=summary,scr_support_db=scr,
        scr_bin=scr_bin,scr_definition="Target/background total energy over core clean-target -20dB range-power support",
        scr_core_db=float(10*np.log10(np.sum(np.abs(target[:,CORE].astype("c16"))**2)/np.sum(np.abs(background[:,CORE].astype("c16"))**2))),
        target_range_support_core_zero_based=np.flatnonzero(support).tolist(),
        effective_width_10db_bins=int(np.flatnonzero(p10)[-1]-np.flatnonzero(p10)[0]+1),
        radial_velocity_m_s=velocity,acceleration_m_s2=acc,scatterer_offsets_m=offsets.tolist(),
        scatterer_amplitudes=amps.tolist(),scatterer_phase_rad=phases.tolist(),
        micro_motion_amplitude_m=micro_amp.tolist(),micro_motion_frequency_hz=micro_hz.tolist(),
        micro_motion_phase_rad=micro_phase.tolist(),common_gain=gain)
    return target,meta
def phase_labels(s):
    out={};cs=[]
    for lag in [1,4,16]:
        c=np.sum(s[lag:]*s[:-lag].conj(),axis=1,dtype="c16")
        norm=np.sqrt(np.sum(np.abs(s[lag:].astype("c16"))**2,axis=1)*np.sum(np.abs(s[:-lag].astype("c16"))**2,axis=1))
        valid=norm>max(float(norm.max())*1e-10,1e-30)
        q=np.ones(len(c),dtype="c8");good=valid&(abs(c)>1e-30);q[good]=(c[good]/abs(c[good])).astype("c8")
        w=np.zeros(len(c),dtype="f4");w[good]=np.clip(abs(c[good])/norm[good],0,1)
        out[f"phase_lag{lag}"]=q;out[f"phase_weight_lag{lag}"]=w
        if lag==1:cs=c
    f=float(np.angle(np.sum(cs))*PRF/(2*np.pi))
    k=int(np.argmin(abs(F-f)))
    out["relation_frequency_hz"]=np.float32(f);out["frequency_bin"]=np.int32(k)
    out["frequency_offset_hz"]=np.float32(f-F[k])
    return out
def digest(z):return hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
def render_rd(path,z,title,first_range,scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a=rd(z)/max(float(scale),1e-20)
    fig,ax=plt.subplots(figsize=(6.5,4.7),dpi=130)
    im=ax.imshow(a,origin="lower",aspect="auto",interpolation="nearest",cmap="jet",vmin=0,vmax=1,
        extent=[float(FB[0]-(PRF/N)/2),float(FB[-1]+(PRF/N)/2),first_range-.5,first_range+15.5])
    ax.set_xlabel("Doppler Frequency (Hz)");ax.set_ylabel("Range Bin");ax.set_title(title,fontsize=10)
    fig.colorbar(im,ax=ax,label="Amplitude / shared sample maximum")
    fig.tight_layout();fig.savefig(path);plt.close(fig)

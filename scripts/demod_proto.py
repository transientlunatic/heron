"""Demodulated-residual ExactGP prototype.
z = (h_D - h_XAS)*e^{+i Phi_ref};  Re(z),Im(z) smooth -> 2 GPs; reconstruct
h = h_XAS + [Re(z)cosPhi + Im(z)sinPhi] (LINEAR -> exact covariance)."""
import sys, time, numpy as np, torch
import astropy.units as u
from heron.models.gp.exact import ExactGPSurrogate
from heron.models.warping import get_warping
from heron.evaluation.mismatch import compute_mismatch
from heron.evaluation.psd import aligo_design_psd
from heron.train import _get_approximant

CK="checkpoints/phenomd_nonspinning_dense30_exact_xas_lsminq006.pt"; OS=1e27
ck=torch.load(CK, weights_only=False)
wc=dict(ck["warping"]); warp=get_warping(wc.pop("type"), **wc)
xas=ExactGPSurrogate.load(CK, device="cpu")          # for its XAS mean modules
train_x=ck["train_x"].to(torch.float64)
Dp=ck["train_y"]["plus"].to(torch.float64).numpy(); Dc=ck["train_y"]["cross"].to(torch.float64).numpy()

def xas_ref(q_col, t_col):
    """h_XAS_plus,cross (scaled) and cos/sin of the reference phase at raw (q,t)."""
    x=torch.stack([torch.as_tensor(q_col,dtype=torch.float64),
                   torch.as_tensor(t_col,dtype=torch.float64)],dim=1)
    xw=x.clone(); xw[:,1]=warp.warp(xw[:,1], mass_ratio=xw[:,0])
    with torch.no_grad():
        hXp=xas.models["plus"].mean_module(xw).cpu().numpy().astype(float)
        hXc=xas.models["cross"].mean_module(xw).cpu().numpy().astype(float)
    A=np.sqrt(hXp**2+hXc**2); afl=1e-3*(A.max() if A.max()>0 else 1.0); low=A<afl
    cosP=np.where(low,1.0,hXp/np.where(low,1.0,A)); sinP=np.where(low,0.0,hXc/np.where(low,1.0,A))
    return hXp,hXc,cosP,sinP

# ---- demodulated targets at training points ----
hXp,hXc,cosP,sinP=xas_ref(train_x[:,0].numpy(), train_x[:,1].numpy())
rp=Dp-hXp; rc=Dc-hXc
ReZ=rp*cosP+rc*sinP; ImZ=rp*sinP-rc*cosP
print(f"targets: |D-XAS| median={np.median(np.abs(rp)):.3e}  ReZ std={ReZ.std():.3e}  ImZ std={ImZ.std():.3e}",flush=True)

# ---- train two GPs on Re(z),Im(z) (scaled units, output_scale=1, ZeroMean) ----
t0=time.time()
demod=ExactGPSurrogate(
    train_x=train_x.to(torch.float32),
    train_y_plus=torch.tensor(ReZ,dtype=torch.float32),
    train_y_cross=torch.tensor(ImZ,dtype=torch.float32),
    warping=warp, nu=2.5, output_scale=1.0, device="cuda",
    total_mass=60.0, distance=100.0, training_iterations=200,
    ls_min_time=0.015, ls_min_q=0.06, noise_floor_rel=0.01, cholesky_size=6000,
    mean_module=None)
print(f"trained in {time.time()-t0:.0f}s",flush=True)
demod.save(sys.argv[1] if len(sys.argv)>1 else "/tmp/demod.pt")

def predict_demod(q, times):
    wf=demod.predict({"mass_ratio":float(q),"times":times})
    ReZp=wf["plus"].data; ImZp=wf["cross"].data
    vRe=np.diag(wf["plus"].covariance); vIm=np.diag(wf["cross"].covariance)
    hXp,hXc,cosP,sinP=xas_ref(np.full(len(times),q), times)
    hp=hXp+ReZp*cosP+ImZp*sinP; hc=hXc+ReZp*sinP-ImZp*cosP
    vp=cosP**2*vRe+sinP**2*vIm; vc=sinP**2*vRe+cosP**2*vIm
    return hp,hc,vp,vc

# ---- mismatch scan vs IMRPhenomD (same window/metric as before) ----
times=np.linspace(-0.5,0.02,512); dt=times[1]-times[0]
psd=aligo_design_psd(np.fft.rfftfreq(512,d=dt)); D=_get_approximant("IMRPhenomD")
qs=np.linspace(0.45,0.90,60); mm=np.empty(len(qs))
for i,q in enumerate(qs):
    hp,_,_,_=predict_demod(q,times)
    ref=D.time_domain({"mass_ratio":float(q),"total_mass":60*u.solMass,"luminosity_distance":100*u.Mpc},times=times)["plus"].data
    mm[i]=compute_mismatch(hp,ref,dt,psd)
np.savez(sys.argv[2] if len(sys.argv)>2 else "/tmp/demod_mm.npz", q=qs, mismatch=mm)
print(f"\nDEMOD mismatch vs D: median={np.median(mm):.3e}  worst={mm.max():.3e}@q={qs[mm.argmax()]:.3f}",flush=True)
print(" (compare: exact-XAS 3.19e-3, phase-amp 7.60e-5)")

# ---- K-calibration: var / true-squared-error, strain space, in-band ----
print("\nK-calibration (median var/err^2, plus channel; exact-XAS~0.3-33, phase-amp~1e3-1e8):")
band=(times>=-0.3)&(times<=0.0)
for q in (0.50,0.60,0.70,0.80):
    hp,_,vp,_=predict_demod(q,times)
    ref=D.time_domain({"mass_ratio":float(q),"total_mass":60*u.solMass,"luminosity_distance":100*u.Mpc},times=times)["plus"].data*OS
    err2=(hp-ref)**2; ratio=vp[band]/np.maximum(err2[band],1e-30*OS**2)
    print(f"  q={q}: var/err^2 median={np.median(ratio):.3e}   (rms err/rms signal={np.sqrt(np.mean(err2[band]))/np.sqrt(np.mean(ref[band]**2)):.3e})")

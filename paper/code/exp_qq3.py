import numpy as np, json
from scipy.optimize import minimize
from scipy.stats import chi2
exec(open('exp_qq.py').read().split("out=[]")[0])
def unit(t,p): return np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
def q3(x):
    psi=unit(x[0],x[1]); a=np.array([0,0,1.]); b=unit(x[2],x[3])
    PA=np.outer(a,a); PB=np.outer(b,b); I=np.eye(3)
    s=lambda P1,P2:np.linalg.norm(P2@P1@psi)**2
    return np.array([s(PA,PB),s(PA,I-PB),s(I-PA,PB),s(I-PA,I-PB)]),np.array([s(PB,PA),s(PB,I-PA),s(I-PB,PA),s(I-PB,I-PA)])
res=json.load(open('qq_results.json'))
rng=np.random.default_rng(0)
for o,(name,nab,pab,nba,pba) in zip(res,D):
    pab=np.array(pab)/sum(pab);pba=np.array(pba)/sum(pba);cab=pab*nab;cba=pba*nba
    llsat=ll(cab,pab)+ll(cba,pba)
    f=lambda x:-(ll(cab,q3(x)[0])+ll(cba,q3(x)[1]))
    best=min((minimize(f,rng.uniform(0,np.pi,4),method='Nelder-Mead',options=dict(maxiter=4000,xatol=1e-8,fatol=1e-10)) for _ in range(60)),key=lambda r:r.fun)
    llq3=-best.fun
    # QQ-constrained saturated (5 params): maximise with constraint pAB_diff==pBA_diff
    from scipy.optimize import minimize as mn
    def negll(v):
        a=np.exp(v[:4]);a/=a.sum(); b=np.exp(v[4:]);b/=b.sum(); return -(ll(cab,a)+ll(cba,b))
    cons={'type':'eq','fun':lambda v:(lambda a,b:(a[1]+a[2])-(b[1]+b[2]))(np.exp(v[:4])/np.exp(v[:4]).sum(),np.exp(v[4:])/np.exp(v[4:]).sum())}
    r=mn(negll,np.log(np.r_[pab,pba]),constraints=[cons],method='SLSQP')
    llqq=-r.fun
    o.update(G2q3=2*(llsat-llq3),dfq3=2,AICq3=2*4-2*llq3,G2qq=2*(llsat-llqq),AICqq=2*5-2*llqq,
             pq3=float(chi2.sf(2*(llsat-llq3),2)),pc=float(chi2.sf(o['G2c'],3)),pq2=float(chi2.sf(o['G2q'],4)),pqq=float(chi2.sf(2*(llsat-llqq),1)))
    print(name,{k:round(o[k],3) for k in ['q','z','G2q','G2c','pc','G2q3','pq3','G2qq','AICc','AICq3','AICqq','AICs']})
json.dump(res,open('qq_results.json','w'),indent=1)

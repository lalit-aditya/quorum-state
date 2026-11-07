import numpy as np, json
from scipy.optimize import minimize
from scipy.stats import norm
# Wang & Busemeyer (2013) Table 1: (name, NAB, [AyBy,AyBn,AnBy,AnBn], NBA, [ByAy,ByAn,BnAy,BnAn])
D=[("Clinton--Gore",447,[.4899,.0447,.1767,.2886],432,[.5625,.1991,.0255,.2130]),
   ("Gingrich--Dole",437,[.3936,.0824,.3204,.2037],377,[.3448,.3422,.0637,.2493]),
   ("White--Black",459,[.3987,.0174,.1612,.4227],486,[.4012,.0597,.1379,.4012]),
   ("Rose--Jackson",506,[.3379,.3241,.0178,.3202],462,[.4156,.0671,.1234,.3939]),
   ("Racial hostility (lab)",116,[.2500,.0603,.2931,.3966],108,[.3241,.0648,.2130,.3981]),
   ("Affirmative action (lab)",118,[.4831,.0424,.1610,.3136],106,[.5660,.1321,.0283,.2736])]
def qmodel(par):
    s,phi=par  # state angle, angle between A-yes and B-yes rays (2D real)
    psi=np.array([np.cos(s),np.sin(s)])
    a=np.array([1.,0.]); b=np.array([np.cos(phi),np.sin(phi)])
    P=lambda v:np.outer(v,v)
    I=np.eye(2); PA=P(a);PB=P(b)
    def seq(P1,P2): return np.linalg.norm(P2@P1@psi)**2
    ab=[seq(PA,PB),seq(PA,I-PB),seq(I-PA,PB),seq(I-PA,I-PB)]   # AyBy,AyBn,AnBy,AnBn
    ba=[seq(PB,PA),seq(PB,I-PA),seq(I-PB,PA),seq(I-PB,I-PA)]   # ByAy,ByAn,BnAy,BnAn   # ByAy,ByAn,BnAy,BnAn
    return np.array(ab),np.array(ba)
def ll(counts,probs): probs=np.clip(probs,1e-9,1); return np.sum(counts*np.log(probs))
out=[]
for name,nab,pab,nba,pba in D:
    pab=np.array(pab);pba=np.array(pba)
    pab/=pab.sum();pba/=pba.sum()
    cab=pab*nab; cba=pba*nba
    # q-test
    qAB=pab[1]+pab[2]; qBA=pba[1]+pba[2]; q=qAB-qBA
    p=(qAB*nab+qBA*nba)/(nab+nba); N=1/(1/nab+1/nba)
    z=q/np.sqrt(p*(1-p)/N)   # variance p(1-p)(1/nab+1/nba)
    # saturated
    llsat=ll(cab,pab)+ll(cba,pba)
    # classical order-invariant joint (3 params): same joint table, BA reordered: ByAy=AyBy, ByAn=AnBy, BnAy=AyBn, BnAn=AnBn
    pj=(cab+cba[[0,2,1,3]])/(nab+nba)
    llcl=ll(cab,pj)+ll(cba,pj[[0,2,1,3]])
    # quantum 2D, 2 params
    best=None
    for s0 in np.linspace(0,np.pi,13):
        for f0 in np.linspace(0,np.pi,13):
            r=minimize(lambda x:-(ll(cab,qmodel(x)[0])+ll(cba,qmodel(x)[1])),[s0,f0],method='Nelder-Mead')
            if best is None or r.fun<best.fun: best=r
    llq=-best.fun
    G2q=2*(llsat-llq); G2c=2*(llsat-llcl)
    aic=lambda L,k:2*k-2*L
    out.append(dict(name=name,nab=nab,nba=nba,q=q,z=z,
        G2q=G2q,dfq=4,G2c=G2c,dfc=3,AICq=aic(llq,2),AICc=aic(llcl,3),AICs=aic(llsat,6),
        phi=float(np.degrees(best.x[1])%180)))
for o in out: print({k:(round(v,4) if isinstance(v,float) else v) for k,v in o.items()})
json.dump(out,open('qq_results.json','w'),indent=1)

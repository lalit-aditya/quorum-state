import numpy as np, json, itertools
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_continuous_lyapunov
plt.rcParams.update({'font.size':9,'font.family':'serif','figure.dpi':150})
X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1.,-1]);I2=np.eye(2)
R=lambda n,t:np.cos(t/2)*I2-1j*np.sin(t/2)*(n[0]*X+n[1]*Y+n[2]*Z)
out={}
# --- reciprocity table
D=[("Clinton--Gore",[.4899,.0447,.1767,.2886],[.5625,.1991,.0255,.2130]),
   ("Gingrich--Dole",[.3936,.0824,.3204,.2037],[.3448,.3422,.0637,.2493]),
   ("White--Black",[.3987,.0174,.1612,.4227],[.4012,.0597,.1379,.4012]),
   ("Rose--Jackson",[.3379,.3241,.0178,.3202],[.4156,.0671,.1234,.3939]),
   ("Racial hostility (lab)",[.2500,.0603,.2931,.3966],[.3241,.0648,.2130,.3981]),
   ("Affirmative action (lab)",[.4831,.0424,.1610,.3136],[.5660,.1321,.0283,.2736])]
out['recip']=[(n,a[0]/(a[0]+a[1]),b[0]/(b[0]+b[1])) for n,a,b in D]
for r in out['recip']: print('recip',r)
# --- Exp B: solar example, corrected
psi=np.array([np.cos(0.7),np.sin(0.7)],complex)
ops={'Cost':R([0,1,0],-0.3),'Risk':R([1,0,0],-0.4),'Timing':R([0,0,1],0.6)}
sol=[]
for o in itertools.permutations(ops):
    s=psi.copy()
    for k in o: s=ops[k]@s
    sol.append((' -> '.join(o),abs(s[0])**2))
out['solar']=sol; print(sol)
s1=R([0,1,0],-0.3)@R([0,1,0],-0.4)@psi; s2=R([0,1,0],-0.4)@R([0,1,0],-0.3)@psi
out['sameaxis_diff']=float(np.abs(s1-s2).max())
# --- Exp B2: order sensitivity vs axis angle
rng=np.random.default_rng(1); phis=np.linspace(0,np.pi,37); S=[];Ssd=[]
th=1.0
for phi in phis:
    na=np.array([0,1.,0]); nb=np.array([np.sin(phi),np.cos(phi),0])
    Ua,Ub=R(na,th),R(nb,th); vals=[]
    for _ in range(2000):
        v=rng.normal(size=4); v=v[:2]+1j*v[2:]; v/=np.linalg.norm(v)
        vals.append(abs(abs((Ub@Ua@v)[0])**2-abs((Ua@Ub@v)[0])**2))
    S.append(np.mean(vals)); Ssd.append(np.std(vals)/np.sqrt(2000))
S=np.array(S); comm=2*np.abs(np.sin(phis))
out['orderS']=dict(corr=float(np.corrcoef(S,comm)[0,1]),Smax=float(S.max()))
fig,ax=plt.subplots(figsize=(3.3,2.3))
ax.plot(np.degrees(phis),S,'k-',label=r'mean $|\Delta P|$ (2000 random states)')
ax2=ax.twinx(); ax2.plot(np.degrees(phis),comm,'C0--',label=r'$\|[G_a,G_b]\|=2\sin\varphi$'); ax2.set_ylabel('commutator norm',color='C0')
ax.set_xlabel(r'angle between criterion axes $\varphi$ (deg)'); ax.set_ylabel('order sensitivity')

fig.tight_layout(); fig.savefig('figs/order_vs_angle.pdf')
# --- Exp C: schedule-dependent decay (Kofman-Kurizki overlap)
w=np.linspace(-3000,3000,3000001); dw=w[1]-w[0]
def F(tau): x=w*tau/2; s=np.sinc(x/np.pi); return tau*s**2/(2*np.pi)
def G_lor(wv,xi=1.0,w0=0.0,Gam=1.0): return Gam/(2*np.pi)*xi**2/((wv-w0)**2+xi**2)
taus=np.logspace(-1,1.7,80); curves={}
for lab,Gf,wa in [('resonant',G_lor,0.0),('detuned',G_lor,3.0),('flat',lambda v:np.full_like(v,1/(2*np.pi)),0.0)]:
    gr=2*np.pi*Gf(np.array([wa]))[0]
    Rt=[2*np.pi*np.sum(Gf(w+wa)*F(t))*dw/gr for t in taus]
    curves[lab]=np.array(Rt)
for k,v in curves.items(): print(k,v.min().round(3),v.max().round(3), taus[np.argmax(v)].round(2))
out['decay']={k:dict(min=float(v.min()),max=float(v.max()),tau_at_max=float(taus[np.argmax(v)])) for k,v in curves.items()}
# crossover tau for resonant/detuned where ratio crosses 1
fig,ax=plt.subplots(figsize=(3.3,2.3))
ax.semilogx(taus,curves['resonant'],'k-',label=r'resonant ($\omega_a=\omega_0$)')
ax.semilogx(taus,curves['detuned'],'C3-',label=r'detuned ($|\omega_a-\omega_0|=3\xi$)')
ax.semilogx(taus,curves['flat'],'C0--',label='flat (white-noise) environment')
ax.axhline(1,color='gray',lw=0.5); ax.set_xlabel(r'review interval $\tau$ (units of $\tau_c=1/\xi$)'); ax.set_ylabel(r'drift rate $R(\tau)/R_{\mathrm{GR}}$')
ax.set_ylim(0,2.9); ax.legend(fontsize=6.5,loc='upper right'); fig.tight_layout(); fig.savefig('figs/zeno_antizeno.pdf')
# --- Exp D: closed-loop coordination (Gaussian surrogate)
gam,om=0.5,1.0; Sg=np.array([[0,1],[-1,0]])
A1=-gam/2*np.eye(2)+om*Sg; A=np.block([[A1,np.zeros((2,2))],[np.zeros((2,2)),A1]])
Dn=gam*np.eye(4)
C=np.array([[ -1,0,1,0]])          # observe disagreement q2-q1
B=np.array([[1],[0],[-1],[0]])     # push q1 up, q2 down
def closed(g,r): return A+g*B@C, Dn+g**2*r*B@B.T
def stats(V):
    d=V[0,0]+V[2,2]-2*V[0,2]
    MI=0.5*np.log(np.linalg.det(V[:2,:2])*np.linalg.det(V[2:,2:])/np.linalg.det(V)); return d,MI
gs=np.linspace(0,3,121); res={}
for eta in [1.0,0.5,0.2]:
    r=0.25/eta; dis=[]
    for g in gs:
        Ac,Dc=closed(g,r)
        if np.max(np.linalg.eigvals(Ac).real)>=0: dis.append(np.nan); continue
        V=solve_continuous_lyapunov(Ac,-Dc); dis.append(stats(V)[0])
    res[eta]=np.array(dis)
    print('eta',eta,'g*',gs[np.nanargmin(dis)],'min dis',np.nanmin(dis),'dis(0)',dis[0])
out['feedback']={str(e):dict(gstar=float(gs[np.nanargmin(v)]),dmin=float(np.nanmin(v)),d0=float(v[0])) for e,v in res.items()}
# transient: start agreed, threshold crossing
V0=np.array([[1,0,0.95,0],[0,1,0,0.95],[0.95,0,1,0],[0,0.95,0,1]])
ts=np.linspace(0,12,601); thr=0.2; tr={}
for g in [0.0,0.6]:
    Ac,Dc=closed(g,0.25); M=[]
    for t in ts:
        E=expm(Ac*t); V=E@V0@E.T + solve_continuous_lyapunov(Ac,-Dc) - E@solve_continuous_lyapunov(Ac,-Dc)@E.T
        M.append(stats(V)[1])
    M=np.array(M); cross=ts[np.argmax(M<thr)] if (M<thr).any() else None; tr[g]=(M,cross)
    print('g',g,'cross',cross,'final MI',M[-1])
out['csd']={str(g):dict(cross=c,final=float(M[-1])) for g,(M,c) in tr.items()}
fig,axs=plt.subplots(1,2,figsize=(6.8,2.5))
for e,v in res.items(): axs[0].plot(gs,v,label=fr'$\eta={e}$')
axs[0].set_xlabel('feedback gain $g$'); axs[0].set_ylabel(r'disagreement $\mathrm{Var}(q_1-q_2)$'); axs[0].legend(fontsize=7)
for g,(M,c) in tr.items(): axs[1].plot(ts,M,label=f'$g={g}$')
axs[1].axhline(thr,color='gray',ls=':',lw=0.8); axs[1].set_xlabel('time (model units)'); axs[1].set_ylabel('mutual information $I_{ab}$ (nats)'); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig('figs/feedback.pdf')
json.dump(out,open('sim_results.json','w'),indent=1,default=str)

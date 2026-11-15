import json, urllib.request, concurrent.futures as cf, sys, numpy as np
B="https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
def get(p):
    with urllib.request.urlopen(B+p, timeout=180) as r: return json.load(r)
comps=[(43,106),(55,282),(55,43),(43,3)]
mids=[]
for c,s in comps: mids += [m["match_id"] for m in get(f"matches/{c}/{s}.json")]
NX,NY=16,12
def cell(x,y):
    # StatsBomb 120x80, attacking left->right for the acting team
    i=min(NX-1,max(0,int(x/120*NX))); j=min(NY-1,max(0,int(y/80*NY)))
    return j*NX+i
def proc(mid):
    shots=np.zeros(NX*NY); goals=np.zeros(NX*NY); moves=np.zeros(NX*NY)
    T=np.zeros((NX*NY,NX*NY))
    try: ev=get(f"events/{mid}.json")
    except Exception: return None
    for e in ev:
        t=e["type"]["name"]; loc=e.get("location")
        if not loc: continue
        c=cell(*loc[:2])
        if t=="Shot":
            if e["shot"]["type"]["name"]=="Penalty": continue
            shots[c]+=1; goals[c]+= e["shot"]["outcome"]["name"]=="Goal"
        elif t=="Pass":
            p=e["pass"]
            if p.get("type",{}).get("name") in ("Throw-in","Corner","Free Kick","Goal Kick","Kick Off"): 
                pass
            moves[c]+=1
            if "outcome" not in p: T[c,cell(*p["end_location"][:2])]+=1
        elif t=="Carry":
            moves[c]+=1; T[c,cell(*e["carry"]["end_location"][:2])]+=1
    return shots,goals,moves,T
S=np.zeros(NX*NY);G=np.zeros(NX*NY);M=np.zeros(NX*NY);TT=np.zeros((NX*NY,NX*NY));n=0
with cf.ThreadPoolExecutor(12) as ex:
    for r in ex.map(proc,mids):
        if r is None: continue
        s,g,m,t=r; S+=s;G+=g;M+=m;TT+=t;n+=1
tot=S+M
ps=np.divide(S,tot,out=np.zeros_like(S),where=tot>0)
pm=np.divide(M,tot,out=np.zeros_like(M),where=tot>0)
pg=np.divide(G,S,out=np.zeros_like(G),where=S>0)
Tm=np.divide(TT,M[:,None],out=np.zeros_like(TT),where=M[:,None]>0)
xt=np.zeros(NX*NY)
for it in range(60):
    new=ps*pg+pm*(Tm@xt)
    if np.abs(new-xt).max()<1e-7: xt=new;break
    xt=new
grid=xt.reshape(NY,NX)
# symmetrise top/bottom (left/right wing) to reduce noise
grid=(grid+grid[::-1,:])/2
print("matches",n,"iters",it,"shots",int(S.sum()),"moves",int(M.sum()), file=sys.stderr)
print("max",grid.max(),"min",grid.min(), file=sys.stderr)
json.dump({"nx":NX,"ny":NY,"grid":[[round(float(v),5) for v in row] for row in grid],
  "source":"StatsBomb open data: WC2018, WC2022, Euro2020, Euro2024; %d matches"%n},open("models/xt_grid.json","w"))
np.set_printoptions(precision=3,suppress=True,linewidth=200)
print(grid)

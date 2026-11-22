# Step 3: three-panel map for every folio (original | heat on a shared scale | cells above the manuscript-wide 99th percentile)
# Requires scores.npz from score.py (84 x 56 cell scores per folio).
import numpy as np, cv2, os, sys, csv, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from multiprocessing import Pool
from scipy.stats import rankdata
import argparse
_ap=argparse.ArgumentParser(description='Step 3: draw a three-panel map for every folio, an overview sheet and an index table.')
_ap.add_argument('--images',required=True); _ap.add_argument('--scores',default='scores.npz'); _ap.add_argument('--scores_dered',default=None,help='optional scores of the red-removed images')
_ap.add_argument('--out',default='maps')
_args=_ap.parse_args()
IMG=_args.images; OUT=_args.out
os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'font.size':10})
HAS_DERED=_args.scores_dered is not None
_s=np.load(_args.scores); _d=np.load(_args.scores_dered if HAS_DERED else _args.scores)
s={k:_s[k] for k in _s.files}; d={k:_d[k] for k in _d.files}; pages=sorted(int(k) for k in s)
R,C=84,56; mr,mc=5,3                       # 6% margin = 5 rows top/bottom, 3 columns left/right
grid=lambda z,p: z[str(p)].reshape(R,C)
inner=lambda z,p: grid(z,p)[mr:R-mr,mc:C-mc]
allin=np.concatenate([inner(s,p).ravel() for p in pages])
T99=np.percentile(allin,99); LO=np.percentile(allin,95); HI=np.percentile(allin,99.9)
score={p:np.percentile(inner(s,p),99) for p in pages}; dscore={p:np.percentile(inner(d,p),99) for p in pages}
rank=lambda sc:{p:i+1 for i,p in enumerate(sorted(pages,key=lambda q:-sc[q]))}
rk,drk=rank(score),rank(dscore)
def load(p):
    return cv2.imdecode(np.fromfile(os.path.join(IMG,f'ZD_{p}.png'),dtype=np.uint8),cv2.IMREAD_COLOR)
def ink_share(bgr):
    b=cv2.resize(bgr,(896,1344),interpolation=cv2.INTER_AREA); L=cv2.cvtColor(b,cv2.COLOR_BGR2LAB)[...,0]
    bg=cv2.medianBlur(cv2.morphologyEx(L,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))),21)
    m=((bg.astype(int)-L.astype(int))>25).astype(np.float32); return m.reshape(R,16,C,16).mean((1,3))[mr:R-mr,mc:C-mc].mean()
CJK=['Noto Sans CJK SC','Microsoft YaHei','SimHei','PingFang SC','DejaVu Sans']   # panel titles are bilingual; first installed font is used
cmap=plt.get_cmap('inferno_r')
BLUE=(31/255,95/255,191/255)
def render(p):
    bgr=load(p); H,W=bgr.shape[:2]; rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    g=grid(s,p); inn=np.zeros((R,C),bool); inn[mr:R-mr,mc:C-mc]=True
    top=(g>T99)&inn; ntop=int(top.sum())
    # heat overlay: interior cells only; cells below the manuscript-wide 95th percentile are transparent
    val=np.clip((g-LO)/(HI-LO),0,1); val[~inn]=0; val[g<LO]=0
    V=cv2.resize(val.astype(np.float32),(W,H),interpolation=cv2.INTER_NEAREST)
    col=cmap(V)[...,:3]; a=(V>0)*(0.35+0.45*V)
    heat=(rgb/255.)*(1-a[...,None])+col*a[...,None]
    M=cv2.resize(top.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
    cnts,_=cv2.findContours(M,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    outl=rgb.copy(); fill=outl.copy(); fill[M>0]=(31,95,191); outl=cv2.addWeighted(fill,0.22,outl,0.78,0)
    cv2.drawContours(outl,cnts,-1,(31,95,191),3)
    y0,x0=mr/R*H,mc/C*W
    ys,xs=np.nonzero(top[mr:R-mr,mc:C-mc]); h,w=R-2*mr,C-2*mc
    nb=float(np.mean(np.minimum.reduce([ys,h-1-ys,xs,w-1-xs])<5)) if ntop else float('nan')
    ink=ink_share(bgr)
    fig,axs=plt.subplots(1,3,figsize=(15,7.6),gridspec_kw={'wspace':0.03})
    for ax,im,t in zip(axs,[rgb,heat,outl],['原图 original','热力图（全书共用色标）heat, shared scale','高于全书第99百分位的单元格 cells > manuscript p99']):
        ax.imshow(im); ax.axis('off'); ax.set_title(t,fontsize=10,fontfamily=CJK)
        if im is not rgb: ax.add_patch(plt.Rectangle((x0,y0),W-2*x0,H-2*y0,fill=False,ls='--',lw=0.8,ec='0.25'))
    sm=plt.cm.ScalarMappable(norm=Normalize(LO,HI),cmap=cmap); cb=fig.colorbar(sm,ax=axs[1],orientation='horizontal',fraction=0.04,pad=0.02,shrink=0.85)
    cb.set_label('cell score (colour scale shared by all folios: manuscript p95 to p99.9; lighter cells not shown)',fontsize=8); cb.ax.tick_params(labelsize=7)
    dr=f'   de-red rank {drk[p]}/{len(pages)}' if HAS_DERED else ''
    fig.suptitle(f'ZD{p}   score {score[p]:.4f}   rank {rk[p]}/{len(pages)}{dr}   cells > p99: {ntop}   ink share {ink*100:.1f}%',fontsize=12,y=0.99)
    fig.savefig(os.path.join(OUT,f'ZD_{p:02d}.jpg'),dpi=250,bbox_inches='tight',pil_kwargs={'quality':85}); plt.close(fig)
    # thumbnail for the overview sheet
    tw=220; th=int(round(tw*H/W)); thumb=cv2.resize(outl,(tw,th),interpolation=cv2.INTER_AREA)
    return dict(folio=p,score=round(float(score[p]),4),rank=rk[p],dered_score=(round(float(dscore[p]),4) if HAS_DERED else ''),dered_rank=(drk[p] if HAS_DERED else ''),
                cells_above_p99=ntop,share_near_margin=(round(nb,2) if ntop else ''),ink_share_pct=round(ink*100,1),
                outer_nine=('yes' if (p<=9 or p>=86) else 'no')),thumb
if __name__=='__main__':
    with Pool(4) as pool: res=pool.map(render,pages)
    rows=[r for r,_ in res]; thumbs={r['folio']:t for r,t in res}
    with open(os.path.join(OUT,'folio_index.csv'),'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # overview sheet: folio order, 10 columns
    cols=10; tw=220; th=max(t.shape[0] for t in thumbs.values()); pad=34
    nrow=int(np.ceil(len(pages)/cols)); sheet=np.full((nrow*(th+pad)+10,cols*(tw+10)+10,3),255,np.uint8)
    for i,p in enumerate(pages):
        r,c=divmod(i,cols); t=thumbs[p]; y=r*(th+pad)+pad; x=c*(tw+10)+10
        sheet[y:y+t.shape[0],x:x+tw]=t
        col=(191,95,31) if rk[p]<=10 else (60,60,60)
        cv2.putText(sheet,f'ZD{p}  #{rk[p]}',(x,y-9),cv2.FONT_HERSHEY_SIMPLEX,0.6,col[::-1] if False else (col[2],col[1],col[0]),2,cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT,'overview_all_folios.jpg'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])
    print('T99',T99,'LO',LO,'HI',HI,'done',len(rows))

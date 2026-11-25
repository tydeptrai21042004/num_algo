#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(root,out):
    root=Path(root);out=Path(out)
    def save(fig,d,name):
        d.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(d/(name+'.pdf'));fig.savefig(d/(name+'.png'),dpi=220);plt.close(fig)
    for path in root.rglob('quality.csv'):
        q=pd.read_csv(path);a=pd.read_csv(path.with_name('attacks.csv'));d=out/path.parent.relative_to(root);a=a[(a.status=='ok')&(a.qualified==1)]
        if not len(a) or 'ber' not in a:continue
        g=a.groupby(['method','variant','condition_id']).ber.mean().unstack();fig,ax=plt.subplots(figsize=(13,max(2.5,.3*len(g)+1.5)));im=ax.imshow(g,cmap='viridis',vmin=0,vmax=.5,aspect='auto');ax.set_yticks(range(len(g)),[f'{m}/{v}' for m,v in g.index],fontsize=7);ax.set_xticks(range(0,len(g.columns),3),g.columns[::3],rotation=90,fontsize=7);ax.set_xlabel('Frozen attack condition');fig.colorbar(im,ax=ax,label='BER');save(fig,d,'attack60_heatmap')
        mean=a.groupby(['method','variant']).ber.mean().rename('attack_ber').reset_index();g=q[q.qualified==1].groupby(['method','variant']).agg(psnr=('psnr_db','median'),latency=('embed_ms','median')).reset_index().merge(mean);g['psnr']=pd.to_numeric(g.psnr,errors='coerce');g=g[np.isfinite(g.psnr)];fig,ax=plt.subplots(figsize=(6,4));
        for m,z in g.groupby('method'):ax.scatter(z.psnr,z.attack_ber,label=m,s=22)
        ax.set(xlabel='Median PSNR (dB)',ylabel='Mean attacked BER');ax.legend(fontsize=7);save(fig,d,'psnr_ber')
        if 'hyperparameters' in str(path):
            q['family']=q.variant.str.split('_').str[0]
            for family,col in [('B01','psnr'),('B02','steps'),('B03','offsets'),('B04','color'),('B05','carrier')]:
                z=q[q.family==family];
                if not len(z):continue
                zz=z.groupby('variant').agg(value=(col,'first'),qualification=('qualified','mean'),latency=('embed_ms','median')).reset_index().merge(mean[['variant','attack_ber']],on='variant',how='left');fig,ax=plt.subplots(figsize=(8,4));ax.plot(range(len(zz)),zz.qualification,'o-',label='Qualification');ax.plot(range(len(zz)),zz.attack_ber,'s-',label='BER (qualified)');ax.set_xticks(range(len(zz)),zz.value.astype(str),rotation=35,ha='right',fontsize=7);ax.legend();ax.set_ylabel('Rate');save(fig,d,f'{family}_sensitivity')
    for path in root.rglob('raw_results.csv'):
        df=pd.read_csv(path);d=out/path.parent.relative_to(root)
        if 'sample' not in df or 'embed_ms' not in df:continue
        z=df[(df.status=='ok')&(df.experiment=='C01')];fig,ax=plt.subplots(figsize=(6,4));
        for name,g in z.assign(regime=np.where(z.payload_bits==z.pixels/64,'full capacity','fixed 4096')).groupby('regime'):
            zz=g.groupby('pixels').embed_ms.median();ax.loglog(zz.index,zz.values,'o-',label=name)
        ax.set(xlabel='Pixels',ylabel='Median embedding latency (ms)');ax.legend();save(fig,d,'resolution_scaling')
        z=df[(df.status=='ok')&(df.experiment=='C02')];g=z.groupby('payload_bits').embed_ms.median();fig,ax=plt.subplots(figsize=(6,4));ax.semilogx(g.index,g.values,'o-');ax.set(xlabel='Payload (bits)',ylabel='Median embedding latency (ms)');save(fig,d,'payload_scaling')
        z=df[(df.experiment=='C03')&df.embed_ms.notna()];groups=list(z.groupby('method'))
        if groups:
            fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([g.embed_ms for _,g in groups],labels=[m for m,_ in groups],showfliers=False);ax.tick_params(axis='x',rotation=25);ax.set_ylabel('Embedding latency (ms)');save(fig,d,'latency_distribution')
    for path in root.rglob('video_codec_summary.csv'):
        df=pd.read_csv(path);d=out/path.parent.relative_to(root);
        if not {'width','generation','mean_qualified_ber'}<=set(df.columns):continue
        z=df[(df.status=='ok')&(df.width==512)&(df.generation==1)];
        if len(z) and 'mean_qualified_ber' in z:
            g=z.pivot_table(index='codec',columns='method',values='mean_qualified_ber');fig,ax=plt.subplots(figsize=(11,4));g.plot.bar(ax=ax);ax.set_ylabel('Post-codec BER (qualified frames)');save(fig,d,'video_lossless_ber')
    for path in root.rglob('stream_summary.csv'):
        df=pd.read_csv(path);d=out/path.parent.relative_to(root);z=df[df.status=='ok'];fig,ax=plt.subplots(figsize=(7,4));
        if len(z) and 'successful_fps' in z:
            for m,g in z.groupby('method'):
                a=g.groupby('target_fps').successful_fps.median();ax.plot(a.index,a.values,'o-',label=m)
            ax.set(xlabel='Target FPS',ylabel='Successful processed FPS');ax.legend(fontsize=7);save(fig,d,'video_throughput')
        else:plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/paper');p.add_argument('--out',default='results/paper_ready/figures');a=p.parse_args();plot(a.root,a.out)

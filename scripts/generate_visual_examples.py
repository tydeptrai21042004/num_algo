#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experiments.common import CONDITIONS

def render(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for folder in root.glob('robustness60/visual_examples/*/*'):
        if not (folder/'marked.png').exists():continue
        with PdfPages(out/f'fig_attacks60_{folder.name}.pdf') as pdf:
            for i in range(1,61,6):
                fig,ax=plt.subplots(6,3,figsize=(8,13))
                for row,c in enumerate(CONDITIONS[i:i+6]):
                    aid=c['condition_id']
                    for col,(file,label) in enumerate([(f'{aid}_attacked.png',f"{aid} {c['attack']} {c['parameter']}"),(f'{aid}_recovered.png','Recovered'),(f'{aid}_diff_x4.png','Difference ×4')]):
                        a=ax[row,col];p=folder/file
                        if p.exists():a.imshow(Image.open(p),cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        else:a.text(.5,.5,'Missing',ha='center')
                        a.set_title(label,fontsize=7);a.axis('off')
                fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,5,figsize=(12,3))
        for ax,f in zip(axes,['host.png','marked.png','watermark.png','clean_recovered.png','embedding_diff_x16.png']):ax.imshow(Image.open(folder/f),cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_title(f.removesuffix('.png'),fontsize=8);ax.axis('off')
        fig.tight_layout();fig.savefig(out/f'fig_embedding_{folder.name}.pdf');plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();render(a.root,a.out)

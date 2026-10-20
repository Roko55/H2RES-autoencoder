"""Plot frozen annual wind diversity without fitting or model selection."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/wind_comparison'

def main():
    summary = json.loads((DATA/'annual_wind_summary.json').read_text())
    rows = list(csv.DictReader((DATA/'annual_wind_paths.csv').open()))
    labels = ['Historical\nreference','Autoencoder','Current\npipeline']
    colors = ['#59636c','#cb783b','#187f83']
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.7))
    for i,((name,record),color) in enumerate(zip(summary.items(),colors)):
        if name != 'Historical reference':
            v=np.array([float(r['annual_mean_wind_CF']) for r in rows if r['model']==name])
        else:
            v=np.array(record['annual_means'])
        axes[0].scatter(np.linspace(i-.16,i+.16,len(v)),np.sort(v),s=23,color=color)
        axes[0].plot([i-.22,i+.22],[v.mean()]*2,color=color,lw=2)
        axes[1].bar(i,v.std(),color=color,width=.6)
        axes[1].text(i,v.std()+.00025,f'{v.std():.5f}',ha='center',fontsize=10)
    for ax in axes:
        ax.set_xticks(range(3),labels,fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Annual mean wind capacity factor')
    axes[0].set_title('Each dot represents one annual profile')
    axes[1].set_ylabel('Standard deviation of annual means')
    axes[1].set_title('Variation between annual wind yields')
    axes[1].set_ylim(0,.0097)
    fig.text(.5,.02,'Corrected training-calibrated corpus. Historical: 5 years; each generator: 15 individual paths.\nSame sampling scale (1), seeds 101-103; compact checkpoint selected on validation only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,1))
    fig.savefig(ROOT/'figures/annual_wind_comparison.png',dpi=220,facecolor='white')
    plt.close(fig)
    records=list(csv.DictReader((DATA/'wind_week.csv').open()))
    names=list(dict.fromkeys(r['model'] for r in records))
    fig,axes=plt.subplots(3,1,figsize=(11,6),sharex=True,sharey=True)
    for ax,name,color in zip(axes,names,colors):
        rs=[r for r in records if r['model']==name]
        ax.plot([int(r['hour']) for r in rs],[float(r['wind_CF']) for r in rs],color=color,lw=1.7)
        ax.set_title(name,loc='left',fontsize=10,fontweight='bold')
        ax.set_ylim(0,.45);ax.set_ylabel('Wind CF');ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Hour within the first calendar week')
    fig.text(.5,.018,'Individual paths, not ensemble averages. Fixed first week and sampling seed 101.\nSynthetic paths are alternative weather sequences, not forecasts of the historical week.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,1))
    fig.savefig(ROOT/'figures/wind_week.png',dpi=200,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    main()

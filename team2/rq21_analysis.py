"""Read saved RQ2.1 results; requires numpy, matplotlib, PyYAML. No GPU."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(p):return json.loads(p.read_text()) if p.exists() else {}
def save(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model-size',choices=['100M','350M','both'],default='both');a=p.parse_args()
 root=a.root.resolve();out=a.out.resolve()
 if out==root or root/'runs'==out or root/'runs' in out.parents or root/'eval'==out or root/'eval' in out.parents:p.error('Use a separate analysis output directory')
 out.mkdir(parents=True,exist_ok=True);rows=[];lat=[];metrics=[];inventory=[]
 for config in sorted((root/'runs').glob('*/config.yaml')):
  name=config.parent.name;c=yaml.safe_load(config.read_text());s=load(config.with_name('summary.json'));n=s.get('n_params')
  if not n:continue
  size='100M' if 90e6<n<110e6 else '350M' if 330e6<n<370e6 else None
  if size is None or a.model_size not in ('both',size):continue
  ep=root/'eval/eval_output'/name;l=load(ep/'loss.json');d=load(ep/'downstream.json');m=c['model'];tokens=s['tokens'];budget=round(tokens/n)
  inventory.append(dict(run=name,size=size,finished=s.get('finished'),loss_available=bool(l),downstream_available=bool(d),latency_files=len(list(ep.glob('latency_*.json')))))
  r=dict(run=name,size=size,layers=m['n_layers'],width=m['d_model'],params=n,tokens=tokens,tokens_per_param=tokens/n,budget=budget,embedding_share=m['vocab_size']*m['d_model']/n,gpu_kind=s.get('gpu_kind'),gpu_hours=s.get('gpu_hours'),train_seconds=s.get('train_seconds'))
  if l:r.update(loss=l['loss'],perplexity=l['perplexity'],windows=l['windows_scored'])
  if d:
   for task,vals in d['scores'].items():
    for metric,value in vals.items():
     metrics.append(dict(run=name,task=task,metric=metric,score=value,n_examples=d['n_examples'][task]))
     if not task.startswith('mmlu_'):r[task+'_'+metric]=value
   mm=[(v['acc'],d['n_examples'][t]) for t,v in d['scores'].items() if t.startswith('mmlu_') and 'acc' in v]
   if mm:r.update(mmlu_subjects=len(mm),mmlu_macro=sum(v for v,n in mm)/len(mm),mmlu_micro=sum(v*n for v,n in mm)/sum(n for v,n in mm))
  rows.append(r)
  for path in sorted(ep.glob('latency_*.json')):
   z=load(path)
   for mode,caches in z['timings_ms'].items():
    for cache,reps in caches.items():
     for i,x in enumerate(reps):lat.append(dict(run=name,job=z['job_id'],device=z['device_kind'],step=z.get('step'),mode=mode,cache_capacity=int(cache),repeat=i,n_steps=len(x),median_ms=float(np.median(x)),mean_ms=float(np.mean(x)),p95_ms=float(np.quantile(x,.95))))
 save(out/'results.csv',rows);save(out/'inventory.csv',inventory);save(out/'downstream_all_metrics.csv',metrics);save(out/'latency_repeats.csv',lat)
 for size in ['100M','350M']:
  rr=[r for r in rows if r['size']==size and 'loss' in r]
  if not rr:continue
  fig,ax=plt.subplots(figsize=(8,5))
  for budget in sorted(set(r['budget'] for r in rr)):
   sub=sorted([r for r in rr if r['budget']==budget],key=lambda r:r['layers']);ax.plot([r['layers'] for r in sub],[r['loss'] for r in sub],'o-',label='t'+str(budget))
  ax.set(xlabel='Transformer layers',ylabel='Held-out loss (lower is better)',title=size+' depth versus quality');ax.legend();fig.tight_layout();fig.savefig(out/(size+'_loss.png'),dpi=160);plt.close(fig)
  fig,axes=plt.subplots(2,2,figsize=(13,10))
  for ax,(mode,cache) in zip(axes.flat,[(m,c) for m in ['as_shipped','compiled_only'] for c in [512,2048]]):
   for r in rr:
    ll=[x for x in lat if x['run']==r['run'] and x['mode']==mode and x['cache_capacity']==cache]
    if not ll:continue
    if len(set(x['device'] for x in ll))!=1:raise ValueError('Mixed devices: '+r['run'])
    v=[x['median_ms'] for x in ll];mid=np.median(v)
    ax.errorbar(mid,r['loss'],xerr=[[mid-min(v)],[max(v)-mid]],fmt='o' if r['budget']==20 else 's',color='#2563eb' if r['budget']==20 else '#c66b12',alpha=.8)
    ax.annotate('{}L/t{}'.format(r['layers'],r['budget']),(mid,r['loss']),xytext=(4,4),textcoords='offset points',fontsize=8)
   ax.set(title=mode+' | cache capacity '+str(cache),xlabel='Decode ms/token: median of repeat medians; range',ylabel='Held-out loss')
  fig.suptitle(size+' quality versus latency | circles t20; squares t60');fig.tight_layout();fig.savefig(out/(size+'_quality_latency.png'),dpi=160);plt.close(fig)
 print('Runs:',len(rows),'evaluated:',sum('loss' in r for r in rows),'output:',out)
if __name__=='__main__':main()

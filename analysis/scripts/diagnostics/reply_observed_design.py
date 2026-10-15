"""Actual candidate-stream parametric sensitivity with fixed-scale M1 Laplace fits.

This is conditional on the realized interaction stream and estimated scale
parameters, not prospective power or a full hierarchical NUTS power analysis.
Only aggregate diagnostics and synthetic-fit summaries are exported.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.linalg import helmert, cho_factor, cho_solve
from scipy.optimize import minimize, brentq
from scipy.stats import norm


def save(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=Path('analysis.yaml'))
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(args.config.read_text())['reply_observed_design_sensitivity']
    out=args.output; out.mkdir(parents=True, exist_ok=False)
    src=args.root/cfg['source_directory']
    frame_path=src/'tables/model_frame.parquet'; posterior_path=src/'models/M1.nc'
    f=pd.read_parquet(frame_path)
    f['_choice']=pd.factorize(f.choice_set_id,sort=False)[0]
    f=f.sort_values('_choice',kind='stable').reset_index(drop=True)
    choice=f._choice.to_numpy(); starts=np.flatnonzero(np.r_[True,np.diff(choice)!=0])
    sizes=np.diff(np.r_[starts,len(f)]); n=len(starts)
    assert n==2143 and len(f)==17489
    assert np.all(np.add.reduceat(f.chosen.to_numpy(),starts)==1)
    post=xr.open_dataset(posterior_path,group='posterior')
    terms=[str(x) for x in post.term.values]
    x=f[terms].to_numpy(float); ai=f.candidate_target_is_ai.to_numpy(float)
    mean=lambda name: post[name].mean(('chain','draw')).values
    # The available posterior supplies variance scales only. Its archive hash
    # differs from the admitted posterior; actual nuisance coefficients are
    # re-estimated here and the provenance distinction is retained explicitly.
    scales={'receiver_h':float(mean('sigma_receiver_effect')[0]),
            'receiver_ai':float(mean('sigma_receiver_effect')[1]),
            'actor':float(mean('sigma_actor_ai_effect')),
            'topic':float(mean('sigma_topic_ai_effect'))}
    blocks=[x]; precisions=[np.array([1/(cfg['prior_focal_sigma'] if t in ('candidate_target_is_ai','g2_x_candidate_ai') else cfg['prior_adjustment_sigma'])**2 for t in terms])]
    def add_centered(labels, selected, sigma, multiplier):
        levels=pd.unique(labels[selected])
        if len(levels)<2: return
        lookup={v:i for i,v in enumerate(levels)}
        b=np.zeros((len(f),len(levels)-1))
        idx=np.flatnonzero(selected)
        basis=helmert(len(levels),full=False).T
        b[idx]=basis[np.array([lookup[v] for v in labels[idx]])]*multiplier[idx,None]
        blocks.append(b); precisions.append(np.full(len(levels)-1,1/sigma**2))
    group=f.group.to_numpy(); all_rows=np.ones(len(f),bool)
    receiver=f.candidate_author_id.astype(str).to_numpy()
    actor=f.actor_id.astype(str).to_numpy()
    topic=f.primary_root_topic.fillna('UNTAGGED').astype(str).to_numpy()
    for g in ('G1','G2'):
        for a in (0,1):
            add_centered(receiver,(group==g)&(ai==a),scales['receiver_ai' if a else 'receiver_h'],np.ones(len(f)))
        add_centered(actor,group==g,scales['actor'],ai)
    add_centered(topic,all_rows,scales['topic'],ai)
    design=np.column_stack(blocks); prior=np.concatenate(precisions)
    cf=[]
    for arm in (0,1):
        d=design.copy()
        for j,name in enumerate(terms):
            if name=='g2_x_candidate_ai': d[:,j]=arm*ai
            elif name.startswith('g2_x_'): d[:,j]=arm*f[name[5:]].to_numpy(float)
        cf.append(d)
    def probs(theta,d=design):
        u=d@theta; maximum=np.maximum.reduceat(u,starts)
        ex=np.exp(u-maximum[choice]); z=np.add.reduceat(ex,starts)
        return ex/z[choice],maximum+np.log(z)
    def fitted(theta,chosen):
        p,logz=probs(theta)
        return float(logz.sum()-(design[chosen]@theta).sum()+.5*np.dot(prior*theta,theta)), design.T@p-design[chosen].sum(axis=0)+prior*theta
    def curvature(theta):
        p,_=probs(theta)
        means=np.add.reduceat(p[:,None]*design,starts,axis=0)
        return design.T@(p[:,None]*design)-means.T@means+np.diag(prior)
    def projection(theta,with_gradient=False):
        rates=[]; gradients=[]
        for d in cf:
            p,_=probs(theta,d); q=np.add.reduceat(p*ai,starts)
            rates.append(q.mean())
            if with_gradient: gradients.append(d.T@(p*(ai-q[choice]))/n)
        return (rates,gradients[1]-gradients[0]) if with_gradient else rates
    def fit(chosen,start):
        result=minimize(fitted,start,args=(chosen,),jac=True,method='L-BFGS-B',options={'maxiter':cfg['max_iterations'],'ftol':1e-12,'gtol':1e-6,'maxcor':30})
        theta=result.x
        # Newton polishing both checks stationarity and provides Laplace Hessian.
        for _ in range(6):
            value,gradient=fitted(theta,chosen); h=curvature(theta)
            if np.max(np.abs(gradient))<1e-5: break
            step=cho_solve(cho_factor(h),gradient)
            rate=1.
            while fitted(theta-rate*step,chosen)[0]>value and rate>1e-5: rate*=.5
            theta=theta-rate*step
        gradient=fitted(theta,chosen)[1]
        h=curvature(theta)
        rates,g=projection(theta,True)
        se=math.sqrt(float(g@cho_solve(cho_factor(h),g)))
        return theta,rates[1]-rates[0],se,float(np.max(np.abs(gradient)))
    chosen=np.flatnonzero(f.chosen.to_numpy()==1)
    base,baseline_delta,baseline_se,baseline_grad=fit(chosen,np.zeros(design.shape[1]))
    assert baseline_grad<1e-4
    # Independent finite-difference check of the pooled-contrast gradient.
    _,g=projection(base,True)
    for j in (0,len(terms)-1,design.shape[1]-1):
        direction=np.zeros(len(base));direction[j]=1e-5
        plus=projection(base+direction);minus=projection(base-direction)
        numerical=((plus[1]-plus[0])-(minus[1]-minus[0]))/2e-5
        assert abs(numerical-g[j])<1e-7
    counts=[]
    for arm in ('G1','G2'):
        a=f[f.group==arm]; c=a.drop_duplicates('choice_set_id'); by=c.groupby('actor_id').size()
        s=a.groupby('choice_set_id').candidate_target_is_ai.agg(['min','max'])
        counts.append(dict(group=arm,participants=len(by),human_reply_events=len(c),candidate_rows=len(a),ai_candidate_rows=int(a.candidate_target_is_ai.sum()),mixed_sets=int(((s['min']==0)&(s['max']==1)).sum()),minimum_replies=int(by.min()),median_replies=float(by.median()),maximum_replies=int(by.max())))
    save(out/'observed_design.csv',counts)
    metadata=dict(config=cfg,scales=scales,parameters=design.shape[1],observed_fit_delta=baseline_delta,observed_fit_se=baseline_se,observed_fit_gradient=baseline_grad,
        observed_fit_pooled_probabilities=projection(base),
        source_hashes={str(p.relative_to(args.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (frame_path,posterior_path)},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='M1 fixed-scale Laplace approximation, conditional on observed participants, accounts, topics and candidate stream; not full Bayesian hierarchical power',
        posterior_provenance='Available valid21 M1 posterior supplies scales only; hash differs from admitted archived posterior. All utility coefficients re-estimated from observed choices.',
        limitations=['scale uncertainty not propagated','endogenous candidate stream held fixed','no new communities','effect size chosen retrospectively','Gaussian delta-method intervals'])
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rng=np.random.default_rng(cfg['seed']); rows=[]; calibrations=[]
    j0=terms.index('candidate_target_is_ai'); j1=terms.index('g2_x_candidate_ai')
    started=time.time(); cutoff=norm.ppf(1-cfg['alpha']/2)
    for delta in cfg['differences']:
        truth=base.copy()
        def baseline_root(v):
            b=truth.copy();b[j0]=v
            return projection(b)[0]-cfg['baseline_probability']
        truth[j0]=brentq(baseline_root,-30,30)
        def contrast_root(v):
            b=truth.copy();b[j1]=v; rates=projection(b)
            return rates[1]-rates[0]-delta
        truth[j1]=brentq(contrast_root,-30,30)
        rates=projection(truth);assert abs(rates[1]-rates[0]-delta)<1e-10
        calibrations.append(dict(target_difference=delta,pooled_g1=rates[0],pooled_g2=rates[1]))
        p,_=probs(truth)
        for rep in range(cfg['repetitions']):
            # Gumbel-max is an exact categorical draw within each observed set.
            u=np.log(np.maximum(p,1e-300))+rng.gumbel(size=len(p))
            maxima=np.maximum.reduceat(u,starts)
            synthetic=np.flatnonzero(u==maxima[choice]);assert len(synthetic)==n
            estimate,contrast,se,grad=fit(synthetic,truth)
            rows.append(dict(target_difference=delta,replicate=rep,estimated_difference=contrast,se=se,max_gradient=grad,valid=int(grad<1e-4),excludes_zero=int(abs(contrast)>cutoff*se),positive_detection=int(contrast>cutoff*se),covers_truth=int(abs(contrast-delta)<=cutoff*se)))
            if (rep+1)%25==0:
                save(out/'simulations.csv',rows)
                print(json.dumps(dict(target=delta,completed=rep+1,elapsed_seconds=round(time.time()-started))),flush=True)
    save(out/'calibration.csv',calibrations); save(out/'simulations.csv',rows)
    summary=[]
    for delta in cfg['differences']:
        r=[v for v in rows if v['target_difference']==delta]; valid=[v for v in r if v['valid']]
        rate=float(np.mean([v['excludes_zero'] for v in valid])) if valid else float('nan')
        summary.append(dict(target_difference=delta,attempted=len(r),valid=len(valid),detection_rate=rate,mcse=math.sqrt(rate*(1-rate)/len(valid)) if valid else float('nan'),positive_detection_rate=float(np.mean([v['positive_detection'] for v in valid])),coverage=float(np.mean([v['covers_truth'] for v in valid])),mean_estimate=float(np.mean([v['estimated_difference'] for v in valid])),mean_se=float(np.mean([v['se'] for v in valid]))))
    save(out/'summary.csv',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()

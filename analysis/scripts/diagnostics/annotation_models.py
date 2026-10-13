"""Prepare, inspect, fit and report two bounded annotation analyses.

`prepare`, `inspect`, and `check` never sample. Each explicit `fit` runs one
model variant. No command updates the manuscript or promotes fitted results.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Set before any numerical import, including imports in the preparation module.
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_name]='1'

import numpy as np
import pandas as pd
from scripts.diagnostics.annotation_panels import (
    ROOT, DEFAULT_OUTPUT, CELLS, prepare, read_panels, select_panel, cell_summary,
    reference_rows, file_hash, save_json,
)

VARIANTS=('primary','doubled_prior','equal_participant','unclear_negative','unclear_positive',
          '48h','mixed_requests','any_human','other_human','benchmark')


def runtime(local_temp):
    local_temp=Path(local_temp).resolve()
    local_temp.mkdir(parents=True,exist_ok=True)
    from scripts.diagnostics.rq2_admission import configure_runtime
    configure_runtime(local_temp)
    os.environ.update(XDG_CACHE_HOME=str(local_temp/'cache'),MPLCONFIGDIR=str(local_temp/'mpl'))
    return local_temp


def fitted_contrasts(probabilities):
    probabilities=np.asarray(probabilities)
    return {
        'G1_AI_minus_H':probabilities[...,0]-probabilities[...,1],
        'G2_AI_minus_H':probabilities[...,2]-probabilities[...,3],
        'AI_visible_minus_hidden':probabilities[...,0]-probabilities[...,2],
        'H_visible_minus_hidden':probabilities[...,1]-probabilities[...,3],
        'interaction_visible_minus_hidden_AI_gap':
            (probabilities[...,0]-probabilities[...,1])-(probabilities[...,2]-probabilities[...,3]),
    }


def build_model(frame, analysis, sampling, *, multiplier=1., equal_participant=False):
    """Crossed intercepts with common-reference, account-marginal probabilities.

    Uptake reference: one row per request within community. Criticism reference:
    observed commenter/thread rows. Both account types use the SAME reference
    rows; target-account effects are integrated over N(0, sigma_target), not
    carried between a real human and an imagined AI version of that account.
    """
    import pymc as pm
    if frame.empty or set(frame.cell)!=set(CELLS) or not frame.y.isin([0,1]).all():
        raise ValueError('A binary outcome and observations in all four cells are required')
    frame=frame.reset_index(drop=True)
    codes={}
    coords={'cell':list(CELLS),'observation':np.arange(len(frame))}
    for key in ('participant','context','target'):
        if frame[key].isna().any():
            raise ValueError('Missing clustering keys')
        codes[key],levels=pd.factorize(frame[key],sort=True)
        coords[key]=np.arange(len(levels))
    refs,weights=reference_rows(frame,analysis,equal_participant)
    visible=frame.group.eq('G1').to_numpy(dtype=float)
    ai=frame.target_type.eq('AI').to_numpy(dtype=float)
    nodes,gh_weights=np.polynomial.hermite.hermgauss(20)
    gh_weights=gh_weights/np.sqrt(np.pi)
    with pm.Model(coords=coords) as model:
        alpha=pm.Normal('alpha',mu=0 if analysis=='uptake' else -2,sigma=1.5*multiplier)
        coefficients={key:pm.Normal('beta_'+key,0,sampling['prior_focal_sigma']*multiplier)
                      for key in ('visible','ai','interaction')}
        effects={}
        sigmas={}
        for key in codes:
            sigmas[key]=pm.HalfNormal('sigma_'+key,sampling['prior_scale_sigma']*multiplier)
            z=pm.Normal('z_'+key,0,1,dims=key)
            effects[key]=sigmas[key]*z[codes[key]]
        base=alpha+coefficients['visible']*visible+effects['participant']+effects['context']
        eta=base+coefficients['ai']*ai+coefficients['interaction']*visible*ai+effects['target']
        pm.Bernoulli('outcome',p=pm.math.sigmoid(eta),observed=frame.y.to_numpy(),dims='observation')
        standardized=[]
        for cell in CELLS:
            group,typ=cell.split('_')
            keep=frame.iloc[refs].group.eq(group).to_numpy()
            focal=base[refs[keep]]+(coefficients['ai']+coefficients['interaction']*int(group=='G1'))*int(typ=='AI')
            probability=pm.math.sigmoid(focal[:,None]+np.sqrt(2)*sigmas['target']*nodes[None,:])
            integrated=pm.math.sum(probability*gh_weights[None,:],axis=1)
            standardized.append(pm.math.sum(integrated*weights[keep]))
        p=pm.Deterministic('cell_probability',pm.math.stack(standardized),dims='cell')
        pm.Deterministic('probability_interaction',(p[0]-p[1])-(p[2]-p[3]))
    return model


def _configuration():
    import yaml
    config=yaml.safe_load((ROOT/'analysis.yaml').read_text())['analysis']
    if config['sampling']['chains']!=4:
        raise ValueError('These analyses require four chains')
    return config


def inputs(args):
    panels,meta=read_panels(args.output)
    benchmark=pd.read_csv(args.benchmark) if args.benchmark else None
    if args.variant!='benchmark' and (benchmark is not None or args.benchmark_draw is not None):
        raise ValueError('Benchmark options require the benchmark variant')
    config=_configuration()
    seed=config['seed']+7000+(100 if args.analysis=='criticism' else 0)
    if args.benchmark_draw is not None:
        if args.benchmark_draw<0:
            raise ValueError('Benchmark draw must be nonnegative')
        seed+=args.benchmark_draw
    frame,audit=select_panel(panels,args.analysis,args.variant,benchmark=benchmark,
                             benchmark_draw=args.benchmark_draw,seed=seed)
    return frame,audit,meta,config,seed


def fit_name(args):
    name=args.analysis+'_'+args.variant
    if args.variant=='benchmark':
        if args.benchmark_draw is None:
            raise ValueError('Benchmark draw must be specified')
        name+='_'+str(args.benchmark_draw)
    return name


def fit_identity(args):
    return dict(prepared_sha256=file_hash(args.output/'prepared.json'),analysis=args.analysis,variant=args.variant,
                benchmark_sha256=file_hash(args.benchmark) if args.benchmark else None,
                benchmark_path=str(args.benchmark) if args.benchmark else None,
                benchmark_draw=args.benchmark_draw)


def _summarize(values):
    return dict(estimate=float(np.mean(values)),ci_low=float(np.quantile(values,.025)),
                ci_high=float(np.quantile(values,.975)),posterior_probability_gt_zero=float(np.mean(values>0)))


def verify_fit(directory, expected_identity=None):
    meta=json.loads((directory/'fit.json').read_text())
    if expected_identity is not None and meta['identity']!=expected_identity:
        raise ValueError('Fit belongs to different inputs or analysis settings')
    if meta['status']!='complete' or not meta.get('diagnostics',{}).get('passed'):
        raise ValueError('A complete fit with passing sampler diagnostics is required')
    required={'posterior.nc','cells.csv','attrition.csv','parameter_diagnostics.csv','diagnostics.json',
              'estimands.csv','prior_probabilities.csv','prior_predictive_checks.csv','posterior_predictive_checks.csv'}
    if not required.issubset(meta.get('output_hashes',{})):
        raise ValueError('Fit output authentication is incomplete')
    for name,expected in meta['output_hashes'].items():
        if file_hash(directory/name)!=expected:
            raise ValueError('Fit output changed')
    benchmark=meta['identity'].get('benchmark_path')
    if benchmark and file_hash(benchmark)!=meta['identity']['benchmark_sha256']:
        raise ValueError('Benchmark calibration changed')
    return meta


def worker(args):
    runtime(args.local_temp)
    frame,audit,meta,config,seed=inputs(args)
    dest=args.output/'fits'/fit_name(args)
    dest.mkdir(parents=True,exist_ok=False)
    identity=fit_identity(args)
    metadata=dict(identity=identity,status='running',seed=seed,sampling=config['sampling'],workers=2,
                  prior_multiplier=2 if args.variant=='doubled_prior' else 1,
                  label_source='machine' if args.variant!='benchmark' else 'benchmark_informed_probabilistic_bias_scenario',
                  measurement_validation='pending_separate_benchmark_report',
                  estimand='within-community common reference rows; target account effect integrated over its population distribution',
                  weighting='equal participants' if args.variant=='equal_participant' else ('equal requests' if args.analysis=='uptake' else 'equal human replies'),
                  inference_scope='exploratory ecological associations; no causal disclosure effect or observed acceptance of advice')
    save_json(dest/'fit.json',metadata)
    try:
        import arviz as az
        import pymc as pm
        from scripts.diagnostics.rq3_interaction import predictive_checks, sampler_diagnostics
        metadata.update(pymc=pm.__version__,arviz=az.__version__)
        cell_summary(frame).to_csv(dest/'cells.csv',index=False)
        audit.to_csv(dest/'attrition.csv',index=False)
        model=build_model(frame,args.analysis,config['sampling'],multiplier=metadata['prior_multiplier'],
                          equal_participant=args.variant=='equal_participant')
        with model:
            prior=pm.sample_prior_predictive(draws=1000,random_seed=seed+1,var_names=['cell_probability','outcome'])
            p=prior.prior.cell_probability.values.reshape(-1,4)
            pd.DataFrame([dict(cell=cell,**_summarize(p[:,i])) for i,cell in enumerate(CELLS)]).to_csv(dest/'prior_probabilities.csv',index=False)
            checks_frame=frame.assign(target_key=frame.context)
            predictive_checks(checks_frame,prior.prior_predictive.outcome.values.reshape(-1,len(frame))).to_csv(dest/'prior_predictive_checks.csv',index=False)
            if args.variant=='equal_participant':
                primary=dest.with_name(args.analysis+'_primary')
                verify_fit(primary,{**identity,'variant':'primary'})
                idata=az.from_netcdf(primary/'posterior.nc')
                deterministic=pm.compute_deterministics(idata.posterior,model=model,
                    var_names=['cell_probability','probability_interaction'],progressbar=False)
                for name in deterministic:
                    idata.posterior[name]=deterministic[name]
                metadata.update(sampling_performed=False,posterior_source_sha256=file_hash(primary/'posterior.nc'))
            else:
                idata=pm.sample(draws=config['sampling']['draws'],tune=config['sampling']['tune'],chains=4,cores=2,
                                blas_cores=2,target_accept=config['sampling']['target_accept'],random_seed=seed,
                                progressbar=False)
                metadata['sampling_performed']=True
            predictive=pm.sample_posterior_predictive(idata,var_names=['outcome'],random_seed=seed+2,progressbar=False)
        diag=sampler_diagnostics(idata,dest)
        parameter=pd.read_csv(dest/'parameter_diagnostics.csv')
        diag['finite_diagnostics']=bool(np.isfinite(parameter[['r_hat','ess_bulk','ess_tail']].to_numpy()).all())
        diag['passed']=bool(diag['passed'] and diag['finite_diagnostics'])
        save_json(dest/'diagnostics.json',diag)
        idata.to_netcdf(dest/'posterior.nc')
        p=idata.posterior.cell_probability.transpose('chain','draw','cell').values.reshape(-1,4)
        estimates={cell:p[:,i] for i,cell in enumerate(CELLS)}
        estimates.update(fitted_contrasts(p))
        pd.DataFrame([dict(estimand=name,unit='probability',**_summarize(values)) for name,values in estimates.items()]).to_csv(dest/'estimands.csv',index=False)
        predictive_checks(checks_frame,predictive.posterior_predictive.outcome.values.reshape(-1,len(frame))).to_csv(dest/'posterior_predictive_checks.csv',index=False)
        # Re-authenticate prepared sources, including machine labels, before declaring completion.
        read_panels(args.output)
        if fit_identity(args)!=identity:
            raise ValueError('Fit inputs changed during execution')
        metadata.update(status='complete' if diag['passed'] else 'diagnostics_failed',diagnostics=diag,
                        output_hashes={p.name:file_hash(p) for p in dest.iterdir() if p.is_file() and p.name!='fit.json'})
        save_json(dest/'fit.json',metadata)
        print(json.dumps({'fit':fit_name(args),'status':metadata['status'],'diagnostics':diag}),flush=True)
        return 0 if diag['passed'] else 1
    except BaseException as exc:
        metadata.update(status='failed',error=str(exc))
        save_json(dest/'fit.json',metadata)
        raise


def launch_fit(args):
    """One explicit fit, two CPUs, low priority, RSS/host watchdog and 24 h limit."""
    from scripts.diagnostics.rq2_contact_runner import host,host_ready,group_rss,stop_group,GIB
    local=runtime(args.local_temp)
    lock=(local.parent/'ysocial-annotation-fit.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rq2=ROOT/'runs/rq2_contact_20260907/run.json'
    if rq2.exists() and not args.allow_rq2_overlap:
        state=json.loads(rq2.read_text())
        if state.get('status') in ('running','waiting_for_host','resuming_after_progress'):
            try:
                os.kill(int(state['runner_pid']),0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('RQ2 runner is active; do not overlap heavy analyses')
    frame,*_=inputs(args)
    if set(frame.cell)!=set(CELLS):
        raise ValueError('Insufficient support: all four cells must contain observations')
    if args.variant=='equal_participant':
        verify_fit(args.output/'fits'/(args.analysis+'_primary'),{**fit_identity(args),'variant':'primary'})
    capacity=host(local,ROOT)
    if not host_ready(capacity):
        raise RuntimeError(f'Host capacity gate failed: {capacity}')
    destination=args.output/'fits'/fit_name(args)
    if destination.exists():
        raise FileExistsError('Fit output already exists; inspect it rather than overwrite it')
    permitted=set(os.sched_getaffinity(0))
    if args.allow_rq2_overlap and rq2.exists():
        rq2_state=json.loads(rq2.read_text())
        permitted-=set(rq2_state.get('cpu_affinity',[]))
    cpus=sorted(permitted)[-2:]
    if len(cpus)!=2:
        raise RuntimeError('Two permitted CPUs are required')
    command=['taskset','-c',','.join(map(str,cpus)),'nice','-n','10',sys.executable,'-u','-m',
             'scripts.diagnostics.annotation_models','_worker','--output',str(args.output),
             '--analysis',args.analysis,'--variant',args.variant,'--local-temp',str(local)]
    if args.benchmark:
        command+=['--benchmark',str(args.benchmark),'--benchmark-draw',str(args.benchmark_draw)]
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    cancelled=False
    def stop_requested(*_):
        nonlocal cancelled
        cancelled=True
    previous={s:signal.signal(s,stop_requested) for s in (signal.SIGTERM,signal.SIGINT)}
    proc=None
    peak=0
    rss_limit_gib=18 if args.allow_rq2_overlap else 30
    reason=None
    try:
        with (args.output/(fit_name(args)+'.log')).open('a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print(f'Fit worker PID {proc.pid}; log {log.name}',flush=True)
            while proc.poll() is None:
                rss,_=group_rss(proc.pid)
                peak=max(peak,rss)
                capacity=host(local,ROOT)
                if cancelled:
                    reason='cancelled'
                elif time.monotonic()-started>=86400:
                    reason='24_hour_timeout'
                elif rss>=rss_limit_gib*GIB:
                    reason=f'{rss_limit_gib}_GiB_RSS_watchdog'
                elif capacity['available_memory']<32*GIB:
                    reason='host_memory_pressure'
                elif min(capacity['local_free'],capacity['repository_free'])<10*GIB:
                    reason='disk_pressure'
                if reason:
                    stop_group(proc)
                    break
                time.sleep(5)
            code=proc.wait()
    finally:
        if proc is not None and group_rss(proc.pid)[1]:
            stop_group(proc)
        for sig,handler in previous.items():
            signal.signal(sig,handler)
    save_json(args.output/(fit_name(args)+'.execution.json'),dict(exit_code=code,reason=reason,peak_rss_gib=peak/GIB,
               elapsed_seconds=time.monotonic()-started,cpu_affinity=cpus))
    return code if code else (1 if reason else 0)


def inspect(output):
    panels,_=read_panels(output)
    cells=[]
    audits=[]
    for analysis in ('uptake','criticism'):
        for variant in VARIANTS:
            if variant=='benchmark' or (analysis=='criticism' and variant in ('48h','mixed_requests','any_human','other_human')):
                continue
            frame,audit=select_panel(panels,analysis,variant)
            cells.append(cell_summary(frame).assign(analysis=analysis,variant=variant))
            audits.append(audit.assign(analysis=analysis,variant=variant))
    summary=pd.concat(cells,ignore_index=True)
    summary.to_csv(output/'planned_cells.csv',index=False)
    pd.concat(audits,ignore_index=True).to_csv(output/'planned_attrition.csv',index=False)
    print(summary.loc[summary.variant.isin(['primary','48h','mixed_requests']),
          ['analysis','variant','cell','observations','positive','participants','contexts']].to_string(index=False))


def report(output):
    """Collect verified fits; incomplete/failed fits are status rows, never admitted."""
    read_panels(output)
    rows=[]
    statuses=[]
    fits=output/'fits'
    for directory in sorted(fits.iterdir()) if fits.exists() else []:
        if not directory.is_dir():
            continue
        path=directory/'fit.json'
        if not path.exists():
            statuses.append(dict(fit=directory.name,status='missing_fit_metadata'))
            continue
        meta=json.loads(path.read_text())
        if meta['identity']['prepared_sha256']!=file_hash(output/'prepared.json'):
            raise ValueError('Fit belongs to different prepared inputs')
        status=meta['status']
        execution=output/(directory.name+'.execution.json')
        if execution.exists():
            execution_status=json.loads(execution.read_text())
            if execution_status['exit_code'] or execution_status['reason']:
                status='runner_failed'
        elif status=='complete':
            status='awaiting_runner_exit'
        statuses.append(dict(fit=directory.name,status=status,measurement_validation=meta.get('measurement_validation')))
        if status!='complete':
            continue
        verify_fit(directory)
        row=pd.read_csv(directory/'estimands.csv').assign(fit=directory.name,analysis=meta['identity']['analysis'],
                variant=meta['identity']['variant'],measurement_validation=meta['measurement_validation'])
        rows.append(row)
    pd.DataFrame(statuses,columns=['fit','status','measurement_validation']).to_csv(output/'fit_status.csv',index=False)
    if rows:
        pd.concat(rows,ignore_index=True).to_csv(output/'fit_estimands.csv',index=False)
    elif (output/'fit_estimands.csv').exists():
        raise ValueError('No current complete fits; stale aggregate exists and requires inspection')
    print(f'{len(rows)} verified complete fits; {len(statuses)} fit directories. No results promoted.')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','inspect','check','fit','report','_worker'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--analysis',choices=('uptake','criticism'),default='uptake')
    p.add_argument('--variant',choices=VARIANTS,default='primary')
    p.add_argument('--local-temp',type=Path)
    p.add_argument('--allow-rq2-overlap',action='store_true')
    p.add_argument('--benchmark',type=Path)
    p.add_argument('--benchmark-draw',type=int)
    return p


def main():
    args=parser().parse_args()
    args.output=args.output.resolve()
    if args.benchmark:
        args.benchmark=args.benchmark.resolve()
    if args.stage=='prepare':
        result=prepare(args.output)
        print(json.dumps(result['rows']),flush=True)
        inspect(args.output)
    elif args.stage=='inspect':
        inspect(args.output)
    elif args.stage=='report':
        report(args.output)
    else:
        if args.local_temp is None:
            raise ValueError('A local --local-temp is required for model compilation or fitting')
        if args.stage=='fit':
            return launch_fit(args)
        if args.stage=='_worker':
            return worker(args)
        runtime(args.local_temp)
        frame,_,_,config,_=inputs(args)
        model=build_model(frame,args.analysis,config['sampling'],multiplier=2 if args.variant=='doubled_prior' else 1,
                          equal_participant=args.variant=='equal_participant')
        logp=float(model.compile_logp()(model.initial_point()))
        if not np.isfinite(logp):
            raise ValueError('Model initial log probability is nonfinite')
        print(json.dumps({'analysis':args.analysis,'variant':args.variant,'rows':len(frame),'initial_logp':logp,'sampling_started':False}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

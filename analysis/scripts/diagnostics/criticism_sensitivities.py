"""T22: run two authenticated Sol criticism sensitivities sequentially.

The existing runner reweights primary posterior draws without sampling, then
fits one doubled-prior model. Its two-CPU/30-GiB/24-hour-per-stage watchdog and
source authentication remain in force. No result promotion or manuscript edit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.diagnostics.annotation_models import (
    ROOT, inputs, fit_identity, launch_fit, verify_fit, runtime,
)
from scripts.diagnostics.annotation_panels import reference_rows, file_hash, save_json

OUTPUT=ROOT/'runs/thread_criticism_sol_review_20260910/outputs/analysis'
VARIANTS=('equal_participant','doubled_prior')


def arguments(variant,local_temp):
    return SimpleNamespace(output=OUTPUT,analysis='criticism',variant=variant,
                           local_temp=local_temp,allow_rq2_overlap=False,
                           benchmark=None,benchmark_draw=None)


def preflight(local_temp):
    import numpy as np
    import pandas as pd
    args=arguments('equal_participant',local_temp)
    frame,_,_,config,_=inputs(args)
    primary=verify_fit(OUTPUT/'fits/criticism_primary',{**fit_identity(args),'variant':'primary'})
    if primary['sampling']!=config['sampling']:
        raise ValueError('Sampling/prior settings differ from the saved primary fit')
    if len(frame)!=2547 or frame.y.sum()!=299:
        raise ValueError('Criticism cohort changed')
    other,*_=inputs(arguments('doubled_prior',local_temp))
    pd.testing.assert_frame_equal(frame,other)
    refs,weights=reference_rows(frame,'criticism',True)
    weighted=frame.iloc[refs][['group','participant']].copy()
    weighted['weight']=weights
    mass=weighted.groupby(['group','participant']).weight.sum()
    for group,n in [('G1',20),('G2',16)]:
        assert len(mass.loc[group])==n
        np.testing.assert_allclose(mass.loc[group].to_numpy(),np.full(n,1/n),rtol=1e-12,atol=1e-12)
    for variant in VARIANTS:
        path=OUTPUT/'fits'/f'criticism_{variant}'
        if path.exists():
            verify_fit(path,fit_identity(arguments(variant,local_temp)))
    print('Preflight passed: identical 2547-comment / 299-positive panel; authenticated primary fit; '
          'each participant has weight 1/20 in G1 or 1/16 in G2.',flush=True)


def run(local_temp):
    import pandas as pd
    started=time.monotonic()
    execution=dict(status='running',sampling_variants=['doubled_prior'],
                   saved_draw_variants=['equal_participant'],completed=[],
                   script_sha256=file_hash(Path(__file__)))
    status=OUTPUT/'criticism_sensitivities_execution.json'
    primary_hash=file_hash(OUTPUT/'fits/criticism_primary/posterior.nc')
    save_json(status,execution)
    try:
        preflight(local_temp)
        for variant in VARIANTS:
            args=arguments(variant,local_temp)
            path=OUTPUT/'fits'/f'criticism_{variant}'
            if path.exists():
                print(f'Reusing authenticated completed {variant}',flush=True)
                verify_fit(path,fit_identity(args))
            else:
                print(f'START {variant}',flush=True)
                code=launch_fit(args)
                if code:raise RuntimeError(f'{variant} exited {code}; inspect its stage log')
            meta=verify_fit(path,fit_identity(args))
            expected_sampling=variant=='doubled_prior'
            if meta['sampling_performed']!=expected_sampling:
                raise ValueError(f'Unexpected sampling status for {variant}')
            execution['completed'].append(variant)
            save_json(status,execution)
        if file_hash(OUTPUT/'fits/criticism_primary/posterior.nc')!=primary_hash:
            raise ValueError('Primary posterior changed')
        tables=[]
        for variant in ('primary',*VARIANTS):
            tables.append(pd.read_csv(OUTPUT/'fits'/f'criticism_{variant}'/'estimands.csv').assign(variant=variant))
        combined=pd.concat(tables,ignore_index=True)
        combined.to_csv(OUTPUT/'criticism_sensitivity_comparison.csv',index=False)
        print(combined[combined.estimand.eq('interaction_visible_minus_hidden_AI_gap')].to_string(index=False),flush=True)
        execution.update(status='complete',primary_posterior_unchanged=True)
        print('COMPLETE: both sensitivities passed authentication and convergence checks; no manuscript edits.',flush=True)
    except BaseException as error:
        execution.update(status='failed',error=str(error))
        raise
    finally:
        execution['elapsed_seconds']=time.monotonic()-started
        save_json(status,execution)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-temp',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    runtime(args.local_temp)
    if args.check_only:preflight(args.local_temp)
    else:run(args.local_temp)

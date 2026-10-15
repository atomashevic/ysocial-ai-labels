"""Prepare machine-annotation panels without model fitting or new annotation.

Human benchmark data never replace the frozen machine labels. See README.md
for estimands, follow-up boundaries, and optional calibration scenarios.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnostics.annotate_interlocutors import FIELDS, build_records, digest
from scripts.diagnostics.annotation_descriptives import response_links
from scripts.publication_analysis.config import load_config
from scripts.publication_analysis.data import _valid_contents
from scripts.publication_analysis.rq3 import observation_end_by_group
from scripts.publication_analysis.study import EXPERIMENT_DAY_ORIGIN, EXPERIMENT_TZ, PRIMARY_VALID_DAYS

ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / 'runs/interlocutor_annotations_20260908'
REQUESTS = ROOT / 'runs/request_classification_20260907'
DEFAULT_OUTPUT = ANNOTATIONS / 'outputs/analysis'
LABELS = ('YES', 'NO', 'UNCLEAR')
CELLS = ('G1_AI', 'G1_H', 'G2_AI', 'G2_H')


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def implementation_files():
    return [Path(__file__), Path(__file__).with_name('annotation_models.py'),
            Path(__file__).with_name('annotation_descriptives.py'),
            Path(__file__).with_name('annotate_interlocutors.py'),
            Path(__file__).with_name('classify_requests.py'),
            Path(__file__).with_name('rq3_interaction.py'),
            Path(__file__).with_name('rq2_contact_runner.py'),
            ROOT / 'analysis.yaml',
            *[ROOT / 'scripts/publication_analysis' / name
              for name in ('config.py', 'data.py', 'study.py', 'io.py', 'rq3.py')]]


def source_files():
    config = load_config(ROOT / 'analysis.yaml', profile='release', run_id='rq2_m2_directional_contact_release_20260903')
    contents = config.processed_dir / 'snapshot_contents.parquet'
    if not contents.exists():
        contents = config.processed_dir / 'snapshot_contents.csv'
    return [*implementation_files(), contents, ANNOTATIONS / 'run.json',
            ANNOTATIONS / 'outputs/annotations.csv', REQUESTS / 'run.json',
            REQUESTS / 'outputs/contribution_labels.csv',
            ROOT / 'runs/rq3_downvote_roundts_20260902/outputs/tables/rq3_event_history.parquet']


def valid_window_reason(start, group, hours, observation_end, *, valid_days=PRIMARY_VALID_DAYS,
                        origin=EXPERIMENT_DAY_ORIGIN, timezone=EXPERIMENT_TZ):
    """Require (t, t+h] to remain inside consecutive valid calendar days.

    The last exact recorded reply gives a conservative right endpoint. This
    establishes coverage eligibility, not uninterrupted participant attention.
    """
    start = pd.Timestamp(start)
    if pd.isna(start) or start.tzinfo is None:
        raise ValueError('Window starts require timezone-aware exact timestamps')
    end = start + pd.Timedelta(hours=hours)
    if hours <= 0:
        raise ValueError('Follow-up horizon must be positive')
    days = pd.date_range(start.tz_convert(timezone).normalize(), end.tz_convert(timezone).normalize(), freq='D')
    base = pd.Timestamp(origin).date()
    if any((day.date() - base).days not in valid_days for day in days):
        return 'excluded_day_in_window'
    latest = observation_end.get(str(group))
    if latest is None or pd.isna(latest) or end > latest:
        return 'right_censored'
    return 'eligible'


def direct_parent_label(label, target):
    """Positive criticism elsewhere does not count as criticism of the parent."""
    if label not in LABELS:
        raise ValueError('Unknown machine label')
    if label == 'NO':
        if target != 'NONE':
            raise ValueError('Negative label has an inconsistent target')
        return 'NO'
    if label == 'UNCLEAR':
        if target != 'UNCLEAR':
            raise ValueError('Unclear label has an inconsistent target')
        return 'UNCLEAR'
    if target not in ('DIRECT_PARENT', 'OTHER_INTERLOCUTOR'):
        raise ValueError('Positive label has an unresolved target')
    return 'YES' if target == 'DIRECT_PARENT' else 'NO'


def _codes(frame, column):
    """Group-scoped internal integer codes; never write source account IDs."""
    keys = pd.MultiIndex.from_arrays([frame.group.astype(str), frame[column]])
    return pd.factorize(keys, sort=True)[0]


def load_inputs():
    config = load_config(ROOT / 'analysis.yaml', profile='release', run_id='rq2_m2_directional_contact_release_20260903')
    contents = _valid_contents(config)
    identity = json.loads((ANNOTATIONS / 'run.json').read_text())
    frame_hash = hashlib.sha256(contents.to_json(orient='split', date_format='iso').encode()).hexdigest()
    if frame_hash != identity['source_frame_sha256'] or digest(build_records(contents, 'human')) != identity['records_sha256']:
        raise ValueError('Annotation source frame or input records changed')
    contents = contents.reset_index(drop=True).rename_axis('row_id').reset_index()
    ann = pd.read_csv(ANNOTATIONS / 'outputs/annotations.csv').rename(columns={'contribution_index': 'row_id'})
    req = pd.read_csv(REQUESTS / 'outputs/contribution_labels.csv').rename(columns={'contribution_index': 'row_id'})
    if len(req) != len(contents) or len(ann) != contents.author_type.eq('H').sum():
        raise ValueError('Incomplete annotation coverage')
    joined = contents[['row_id','tweet']].merge(req[['row_id','tweet','label']], on='row_id', validate='one_to_one')
    if len(joined) != len(contents) or not joined.tweet_x.eq(joined.tweet_y).all():
        raise ValueError('Request annotations do not match contribution text')
    if not set(req.label).issubset(LABELS):
        raise ValueError('Invalid request labels')
    contents = contents.merge(req[['row_id','label']].rename(columns={'label':'request_label'}), on='row_id', validate='one_to_one')
    for field in FIELDS:
        if not set(ann[field + '_label']).issubset(LABELS):
            raise ValueError('Invalid interpersonal labels')
    # Avoid loading any annotation quotation into a generated analytical panel.
    columns = ['row_id','group','author_type'] + [field + suffix for field in FIELDS for suffix in ('_label','_target')]
    human = contents.loc[contents.author_type.eq('H')].merge(ann[columns], on=['row_id','group','author_type'], validate='one_to_one')
    if len(human) != len(ann):
        raise ValueError('Interpersonal annotations do not match authoritative group/author types')
    events = pd.read_parquet(ROOT / 'runs/rq3_downvote_roundts_20260902/outputs/tables/rq3_event_history.parquet')
    if events.duplicated(['group','event_id']).any() or not set(events.experiment_day).issubset(PRIMARY_VALID_DAYS):
        raise ValueError('Invalid source event history')
    return contents, human, events


def build_panels(contents, human, events):
    """Return anonymous model inputs and descriptive outcomes in one pass."""
    f = contents.copy()
    if f[['group','content_id','author_id','root_id']].isna().any().any():
        raise ValueError('Missing authoritative contribution/account/thread identity')
    f['account_code'] = _codes(f, 'author_id')
    f['thread_code'] = _codes(f, 'root_id')
    if f.duplicated(['group','content_id']).any() or f.row_id.duplicated().any():
        raise ValueError('Duplicate contribution identities')
    links = response_links(f)
    parent_metadata = f[['group','content_id','row_id','account_code','request_label']].rename(columns={
        'content_id':'parent_content_id','row_id':'parent_row_id','account_code':'parent_account_code',
        'request_label':'parent_request_label'})
    links = links.merge(parent_metadata, on=['group','parent_content_id'], validate='many_to_one')
    end = observation_end_by_group(events)
    followups = {(g, c): part for (g, c), part in links.loc[links.author_type.eq('H')].groupby(['group','parent_content_id'])}
    votes = events.loc[events.actor_type.eq('H') & events.event_type.eq('downvote')]
    vote_groups = {(g, c): part for (g, c), part in votes.groupby(['group','target_content_id'])}
    empty_replies = links.iloc[:0]
    empty_votes = votes.iloc[:0]
    timings = []
    requester_by_response = links.set_index('row_id').parent_author_id.to_dict()
    for row in f.itertuples():
        follow = followups.get((row.group,row.content_id), empty_replies)
        down = vote_groups.get((row.group,row.content_id), empty_votes)
        down = down.loc[down.actor_id.ne(row.author_id)]
        if (down.event_timestamp < row.created_at.floor('h')).any():
            raise ValueError('Downvote precedes the contribution creation hour')
        result = {'row_id':row.row_id}
        for hours in (24,48):
            end_time = row.created_at + pd.Timedelta(hours=hours)
            hits = follow.loc[follow.created_at.le(end_time)]
            requester = requester_by_response.get(row.row_id)
            own = hits.author_id.eq(requester)
            voted = down.loc[down.event_timestamp.ge(row.created_at.floor('h')) & down.event_timestamp.le(end_time)]
            result.update({f'window_{hours}h':valid_window_reason(row.created_at,row.group,hours,end),
                           f'requester_{hours}h':int(own.any()), f'any_human_{hours}h':int(len(hits)>0),
                           f'other_human_{hours}h':int((~own).any()), f'downvote_{hours}h':int(len(voted)>0),
                           f'later_hour_downvote_{hours}h':int(voted.event_timestamp.gt(row.created_at).any())})
        timings.append(result)
    timing = pd.DataFrame(timings)
    # Every direct response to a human contribution remains available for label-error sensitivity.
    uptake = links.loc[links.parent_author_type.eq('H')].copy()
    uptake = uptake.rename(columns={'parent_row_id':'item_id','parent_account_code':'participant',
                                    'account_code':'target','parent_request_label':'machine_label',
                                    'author_type':'target_type'})
    uptake['context'] = uptake.item_id
    cols = ['row_id','item_id','group','target_type','participant','context','target','machine_label']
    uptake = uptake[cols].merge(timing, on='row_id', validate='one_to_one')
    critique = links.loc[links.author_type.eq('H')].merge(human[['row_id','content_criticism_label','content_criticism_target']],
                                                       on='row_id', validate='one_to_one')
    critique['machine_label'] = [direct_parent_label(a,b) for a,b in zip(critique.content_criticism_label,critique.content_criticism_target)]
    critique['item_id'] = critique.row_id
    critique = critique.rename(columns={'account_code':'participant','thread_code':'context',
                                         'parent_account_code':'target','parent_author_type':'target_type'})
    critique = critique[cols]
    rare = human[['row_id','group','comment_to',*[field + suffix for field in FIELDS for suffix in ('_label','_target')]]].copy()
    rare['is_comment'] = rare.comment_to.notna() & rare.comment_to.ne(-1)
    rare = rare.drop(columns='comment_to').merge(f[['row_id','account_code']], on='row_id', validate='one_to_one')
    rare = rare.merge(timing, on='row_id', validate='one_to_one')
    targets = links[['row_id','parent_author_type']].rename(columns={'parent_author_type':'resolved_parent_type'})
    rare = rare.merge(targets, on='row_id', how='left', validate='one_to_one')
    return {'uptake':uptake, 'criticism':critique, 'rare':rare}, end


def rare_summary(frame):
    rows = []
    for category in FIELDS:
        for group in ('G1','G2'):
            arm = frame.loc[frame.group.eq(group)]
            for label in LABELS:
                part = arm.loc[arm[category+'_label'].eq(label)]
                base = dict(category=category,group=group,label=label,n=len(part),denominator=len(arm),
                            authors=part.account_code.nunique(), annotation_proportion=len(part)/len(arm) if len(arm) else np.nan)
                rows.append(dict(**base,metric='prevalence',eligible_n=len(part),positive_n=len(part)))
                for target in ('AI','H','OTHER_INTERLOCUTOR'):
                    mask = (part[category+'_target'].eq('OTHER_INTERLOCUTOR') if target=='OTHER_INTERLOCUTOR' else
                            part[category+'_target'].eq('DIRECT_PARENT') & part.resolved_parent_type.eq(target))
                    rows.append(dict(**base,metric='addressed_'+target,eligible_n=len(part),positive_n=int(mask.sum())))
                for hours in (24,48):
                    eligible = part.loc[part[f'window_{hours}h'].eq('eligible')]
                    for outcome in ('any_human','downvote','later_hour_downvote'):
                        rows.append(dict(**base,metric=f'{outcome}_{hours}h',eligible_n=len(eligible),
                                         positive_n=int(eligible[f'{outcome}_{hours}h'].sum())))
    result = pd.DataFrame(rows)
    result['metric_proportion'] = result.positive_n / result.eligible_n.replace(0, np.nan)
    return result


def prepare(output):
    output = Path(output)
    sources = {str(p.relative_to(ROOT)):file_hash(p) for p in source_files()}
    if (output/'prepared.json').exists():
        meta = json.loads((output/'prepared.json').read_text())
        if meta['sources'] != sources:
            raise ValueError('Prepared inputs differ from live sources; choose a new explicit output directory')
        for name, expected in meta['panels'].items():
            if file_hash(output/name) != expected:
                raise ValueError('Prepared output changed')
        print('Prepared inputs already match; no files changed', flush=True)
        return meta
    if output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output without preparation metadata; inspect it before retrying')
    f, h, events = load_inputs()
    panels, end = build_panels(f,h,events)
    output.mkdir(parents=True,exist_ok=True)
    panel_hashes = {}
    for name, table in panels.items():
        path = output/f'{name}.parquet'
        table.to_parquet(path,index=False)
        panel_hashes[path.name] = file_hash(path)
    rare_summary(panels['rare']).to_csv(output/'rare_descriptives.csv',index=False)
    panel_hashes['rare_descriptives.csv'] = file_hash(output/'rare_descriptives.csv')
    if {str(p.relative_to(ROOT)):file_hash(p) for p in source_files()} != sources:
        raise ValueError('Sources changed during preparation')
    meta = dict(sources=sources,panels=panel_hashes,rows={name:len(p) for name,p in panels.items()},
                request_population=f.loc[f.author_type.eq('H')].groupby(['group','request_label']).size()
                    .rename('contributions').reset_index().to_dict('records'),
                observation_end={g:str(t) for g,t in end.items()},
                measurements='frozen machine annotations; human benchmark used only for validation and optional sensitivity',
                observation_rule='(response time, response time + horizon] within consecutive valid calendar days; conservative last-exact-reply endpoint',
                advice_content='Direct responses to explicit machine-labelled requests; actual provision or acceptance of advice is not measured',
                interpretation='exploratory; one community per condition; no exposure or causal claim',
                paper='https://arxiv.org/abs/2608.07297')
    save_json(output/'prepared.json',meta)
    return meta


def read_panels(output):
    output = Path(output)
    meta = json.loads((output/'prepared.json').read_text())
    for name,expected in meta['sources'].items():
        if file_hash(ROOT/name) != expected:
            raise ValueError(f'Prepared source changed: {name}')
    for name,expected in meta['panels'].items():
        if file_hash(output/name) != expected:
            raise ValueError(f'Prepared output changed: {name}')
    return {name:pd.read_parquet(output/f'{name}.parquet') for name in ('uptake','criticism','rare')},meta


def benchmark_labels(frame, table, measurement, draw, seed):
    """Optional probabilistic bias scenario; primary machine labels stay intact.

    p_positive must come from a separately justified, sampling-aware benchmark
    calibration. This is sensitivity analysis, not a joint measurement model.
    One draw is shared by every response to the same request.
    """
    required={'draw','measurement','group','target_type','machine_label','p_positive'}
    if set(table) != required or table[list(required)].isna().any().any():
        raise ValueError('Benchmark probability columns must match the documented schema')
    if not set(table.machine_label).issubset(LABELS) or not table.p_positive.between(0,1).all():
        raise ValueError('Invalid benchmark labels or probabilities')
    keys=['draw','measurement','group','target_type','machine_label']
    if table.duplicated(keys).any():
        raise ValueError('Duplicate benchmark strata')
    calibration=table.loc[table.measurement.eq(measurement) & table.draw.eq(draw)]
    items=frame[['item_id','group','machine_label']].drop_duplicates().copy()
    # Request label concerns the human request author, not the response author.
    if measurement=='recommendation_request':
        items['target_type']='H'
    else:
        items=items.merge(frame[['item_id','target_type']].drop_duplicates(),on='item_id',validate='one_to_one')
    if items.item_id.duplicated().any():
        raise ValueError('Inconsistent machine labels for shared items')
    items=items.sort_values('item_id').merge(calibration.drop(columns=['draw','measurement']),
        on=['group','target_type','machine_label'],how='left',validate='many_to_one')
    if items.p_positive.isna().any():
        raise ValueError('Benchmark calibration missing a required group/type/label stratum')
    items['sensitivity_positive']=np.random.default_rng(seed).random(len(items)) < items.p_positive
    return frame.item_id.map(items.set_index('item_id').sensitivity_positive).to_numpy(dtype=bool)


def select_panel(panels, analysis, variant='primary', *, benchmark=None, benchmark_draw=None, seed=0):
    common={'primary','doubled_prior','equal_participant','unclear_negative','unclear_positive','benchmark'}
    allowed=common | ({'48h','mixed_requests','any_human','other_human'} if analysis=='uptake' else set())
    if analysis not in ('uptake','criticism') or variant not in allowed:
        raise ValueError('Unsupported analysis/variant')
    frame=panels[analysis].copy()
    audits=[dict(stage='all_candidates',n=len(frame))]
    machine=frame.machine_label
    if variant=='benchmark':
        if benchmark is None or benchmark_draw is None:
            raise ValueError('Benchmark sensitivity requires explicit calibration data and draw')
        positive=benchmark_labels(frame,benchmark,'recommendation_request' if analysis=='uptake' else 'criticism_direct_parent',benchmark_draw,seed)
        include=np.ones(len(frame),dtype=bool)
    else:
        positive=machine.eq('YES').to_numpy()
        include=machine.ne('UNCLEAR').to_numpy()
        if variant in ('unclear_positive','unclear_negative'):
            include[:]=True
            if variant=='unclear_positive':
                positive |= machine.eq('UNCLEAR').to_numpy()
    if analysis=='uptake':
        frame=frame.loc[positive].copy()
        audits.append(dict(stage='request_label_selected',n=len(frame)))
        hours=48 if variant=='48h' else 24
        for reason,n in frame[f'window_{hours}h'].value_counts().items():
            audits.append(dict(stage=f'window_{hours}h_{reason}',n=int(n)))
        frame=frame.loc[frame[f'window_{hours}h'].eq('eligible')].copy()
        if variant=='mixed_requests':
            mixed=frame.groupby('item_id').target_type.nunique().eq(2)
            frame=frame.loc[frame.item_id.isin(mixed.index[mixed])].copy()
        outcome='any_human' if variant=='any_human' else ('other_human' if variant=='other_human' else 'requester')
        frame['y']=frame[f'{outcome}_{hours}h'].astype(int)
    else:
        frame['y']=positive.astype(int)
        frame=frame.loc[include].copy()
    frame['cell']=frame.group+'_'+frame.target_type
    audits.append(dict(stage='selected',n=len(frame)))
    return frame.reset_index(drop=True),pd.DataFrame(audits)


def cell_summary(frame):
    rows=[]
    for cell in CELLS:
        p=frame.loc[frame.cell.eq(cell)]
        row=dict(cell=cell,observations=len(p),positive=int(p.y.sum()),participants=p.participant.nunique(),
                 contexts=p.context.nunique(),target_accounts=p.target.nunique(),
                 raw_proportion=float(p.y.mean()) if len(p) else np.nan)
        for c in ('downvote_24h','later_hour_downvote_24h','downvote_48h'):
            if c in p:
                horizon=c.rsplit('_',1)[-1]
                eligible=p.loc[p['window_'+horizon].eq('eligible')]
                row[c]=int(eligible[c].sum())
                row[c+'_eligible_n']=len(eligible)
        rows.append(row)
    return pd.DataFrame(rows)


def reference_rows(frame, analysis, equal_participant=False):
    """Same within-group reference rows for both account types."""
    reference=frame.drop_duplicates('item_id') if analysis=='uptake' else frame
    rows=reference.index.to_numpy()
    weights=np.ones(len(reference),dtype=float)
    if equal_participant:
        weights=1/reference.groupby(['group','participant']).participant.transform('size').to_numpy()
    for group in ('G1','G2'):
        mask=reference.group.eq(group).to_numpy()
        if not mask.any():
            raise ValueError('Both communities are required')
        weights[mask]/=weights[mask].sum()
    return rows,weights

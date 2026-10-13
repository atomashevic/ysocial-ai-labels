"""Lightweight exploratory annotation proportions and observed responses.

No fits, resampling, model calls, or changes to authenticated analysis code.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd
from scripts.diagnostics.annotate_interlocutors import FIELDS, build_records, digest
from scripts.publication_analysis.config import load_config
from scripts.publication_analysis.data import _valid_contents

ROOT = Path(__file__).resolve().parents[2]
RUN = 'interlocutor_annotations_20260908'
REQUEST_RUN = 'request_classification_20260907'


def response_links(contents):
    """Resolve direct, strictly later, same-thread, non-self content replies."""
    parents = contents[['group', 'content_id', 'author_id', 'author_type', 'created_at', 'root_id']].rename(
        columns={c: 'parent_' + c for c in ['content_id', 'author_id', 'author_type', 'created_at', 'root_id']})
    linked = contents.merge(parents, left_on=['group', 'comment_to'],
                            right_on=['group', 'parent_content_id'], validate='many_to_one')
    return linked.loc[(linked.created_at > linked.parent_created_at)
                      & linked.root_id.eq(linked.parent_root_id)
                      & linked.author_id.ne(linked.parent_author_id)].copy()


def main():
    config = load_config(ROOT / 'analysis.yaml', profile='release', run_id='rq2_m2_directional_contact_release_20260903')
    f = _valid_contents(config)
    identity = json.loads((ROOT / 'runs' / RUN / 'run.json').read_text())
    assert hashlib.sha256(f.to_json(orient='split', date_format='iso').encode()).hexdigest() == identity['source_frame_sha256']
    assert digest(build_records(f, 'human')) == identity['records_sha256']
    f = f.reset_index(drop=True).rename_axis('contribution_index').reset_index()
    assert not f.duplicated(['group', 'content_id']).any()
    f['is_reply'] = f.comment_to.notna() & f.comment_to.ne(-1)
    annotations = pd.read_csv(ROOT / 'runs' / RUN / 'outputs/annotations.csv')
    assert len(annotations) == 2902 and not annotations.contribution_index.duplicated().any()
    requests = pd.read_csv(ROOT / 'runs' / REQUEST_RUN / 'outputs/contribution_labels.csv')
    assert len(requests) == len(f) and not requests.contribution_index.duplicated().any()
    check = f.merge(requests[['contribution_index', 'tweet']], on='contribution_index', validate='one_to_one')
    assert check.tweet_x.eq(check.tweet_y).all()
    f = f.merge(requests[['contribution_index', 'label']].rename(columns={'label': 'recommendation_request_label'}),
                on='contribution_index', validate='one_to_one')
    h = f.loc[f.author_type.eq('H')].merge(annotations, on=['contribution_index', 'group', 'author_type', 'is_reply'],
                                          validate='one_to_one', suffixes=('', '_annotation'))
    assert len(h) == 2902
    categories = ('recommendation_request',) + FIELDS
    links = response_links(f)
    for typ in ('H', 'AI'):
        counts = links.loc[links.author_type.eq(typ)].groupby(['group', 'parent_content_id']).size()
        h['received_' + typ + '_replies'] = [int(counts.get((r.group, r.content_id), 0)) for r in h.itertuples()]
    events = pd.read_parquet(ROOT / 'runs/rq3_downvote_roundts_20260902/outputs/tables/rq3_event_history.parquet')
    assert events.event_id.notna().all() and not events.duplicated(['group', 'event_id']).any()
    down = events.loc[events.actor_type.eq('H') & events.event_type.eq('downvote')].merge(
        f[['group', 'content_id', 'author_id', 'created_at']], left_on=['group', 'target_content_id'],
        right_on=['group', 'content_id'], validate='many_to_one')
    down = down.loc[down.actor_id.ne(down.author_id)].copy()
    # Reactions have hourly timestamps. Retain same-hour votes explicitly.
    down['same_hour'] = down.event_timestamp.eq(down.created_at.dt.floor('h'))
    assert (down.event_timestamp >= down.created_at.dt.floor('h')).all()
    for name, subset in [('human_downvotes', down), ('later_hour_human_downvotes', down.loc[~down.same_hour])]:
        counts = subset.groupby(['group', 'content_id']).size()
        h[name] = [int(counts.get((r.group, r.content_id), 0)) for r in h.itertuples()]
    rows, reception, addressed = [], [], []
    for scope in ('all_contributions', 'comments'):
        cohort = h if scope == 'all_contributions' else h.loc[h.is_reply]
        for category in categories:
            for group in ('G1', 'G2'):
                arm = cohort.loc[cohort.group.eq(group)]
                for label in ('YES', 'NO', 'UNCLEAR'):
                    part = arm.loc[arm[category + '_label'].eq(label)]
                    rows.append(dict(scope=scope, category=category, group=group, label=label,
                                     n=len(part), denominator=len(arm), proportion=len(part)/len(arm),
                                     authors=part.author_id.nunique()))
                    row = dict(scope=scope, category=category, group=group, label=label, contributions=len(part),
                               authors=part.author_id.nunique())
                    for outcome in ('received_H_replies', 'received_AI_replies', 'human_downvotes', 'later_hour_human_downvotes'):
                        row[outcome + '_total'] = int(part[outcome].sum())
                        row[outcome + '_any_n'] = int(part[outcome].gt(0).sum())
                        row[outcome + '_any_proportion'] = float(part[outcome].gt(0).mean()) if len(part) else None
                    reception.append(row)
                if category in FIELDS:
                    pos = arm.loc[arm[category + '_label'].eq('YES')]
                    for target in ('AI', 'H', 'OTHER_INTERLOCUTOR'):
                        selected = (pos[category + '_target'].eq('DIRECT_PARENT') & pos.parent_author_type.eq(target)
                                    if target != 'OTHER_INTERLOCUTOR' else pos[category + '_target'].eq(target))
                        addressed.append(dict(scope=scope, category=category, group=group, target=target,
                                              n=int(selected.sum()), positive_denominator=len(pos),
                                              eligible_parent_denominator=(int((arm.parent_author_type.eq(target)
                                                  & ~arm.self_reply & arm.parent_context_status.eq('available')).sum())
                                                  if target != 'OTHER_INTERLOCUTOR' else None)))
    proportions = pd.DataFrame(rows)
    differences = []
    for (scope, category), part in proportions.loc[proportions.label.eq('YES')].groupby(['scope', 'category']):
        arms = part.set_index('group')
        differences.append(dict(scope=scope, category=category,
                                G1_n=int(arms.loc['G1','n']), G1_denominator=int(arms.loc['G1','denominator']),
                                G2_n=int(arms.loc['G2','n']), G2_denominator=int(arms.loc['G2','denominator']),
                                G2_minus_G1_pp=100*(arms.loc['G2','proportion']-arms.loc['G1','proportion'])))
    # AI uptake around requests: replies to requests, and human returns to responses.
    chain_rows = []
    for group in ('G1', 'G2'):
        for label in ('YES', 'NO', 'UNCLEAR'):
            prompts = h.loc[h.group.eq(group) & h.recommendation_request_label.eq(label)]
            answers = links.loc[links.group.eq(group) & links.parent_content_id.isin(prompts.content_id)]
            for typ in ('AI', 'H'):
                a = answers.loc[answers.author_type.eq(typ)]
                follow = links.loc[links.group.eq(group) & links.author_type.eq('H') & links.parent_content_id.isin(a.content_id)]
                # The follow-up's parent is the answer, whose parent is the request.
                follow = follow.merge(a[['content_id','parent_author_id']].rename(
                    columns={'content_id':'answer_id','parent_author_id':'requester_id'}),
                    left_on='parent_content_id', right_on='answer_id', validate='many_to_one')
                back = follow.loc[follow.author_id.eq(follow.requester_id)]
                chain_rows.append(dict(group=group, request_label=label, response_author_type=typ,
                    human_contributions=len(prompts), contributions_with_response=a.parent_content_id.nunique(),
                    responses=len(a), responses_with_human_followup=follow.parent_content_id.nunique(),
                    responses_with_requester_followup=back.parent_content_id.nunique(),
                    responses_with_human_downvote=down.loc[down.group.eq(group) & down.content_id.isin(a.content_id), 'content_id'].nunique(),
                    human_followups=len(follow), requester_followups=len(back)))
    # Requests can themselves be AI-authored: distinguish their reception.
    request_rows = []
    for (group, typ, label), part in f.groupby(['group','author_type','recommendation_request_label']):
        follow = links.loc[links.group.eq(group) & links.author_type.eq('H') & links.parent_content_id.isin(part.content_id)]
        request_rows.append(dict(group=group, author_type=typ, label=label, contributions=len(part),
            authors=part.author_id.nunique(), contributions_with_human_reply=follow.parent_content_id.nunique(), human_replies=len(follow),
            contributions_with_human_downvote=down.loc[down.group.eq(group) & down.content_id.isin(part.content_id), 'content_id'].nunique()))
    tables = {'proportions': proportions, 'differences': pd.DataFrame(differences),
              'reception': pd.DataFrame(reception), 'addressed_targets': pd.DataFrame(addressed),
              'request_chains': pd.DataFrame(chain_rows), 'request_reception': pd.DataFrame(request_rows)}
    for name, table in tables.items():
        table.insert(0, 'run_id', RUN)
        table.to_csv(ROOT / f'results/tables/annotation_exploratory_{name}.csv', index=False)
    print(pd.DataFrame(differences).to_string(index=False))
    print(pd.DataFrame(chain_rows).query('request_label == "YES"').to_string(index=False))
    print('Checks passed: source fingerprint, annotation input identity, text-to-request mapping, unique group/content and event keys.')


if __name__ == '__main__':
    main()

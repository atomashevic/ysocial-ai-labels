"""Prepare Sol direct-parent criticism for the existing Bayesian model runner."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnostics.annotation_models import ROOT
from scripts.diagnostics.annotate_recommendations import redact
from scripts.diagnostics.annotation_panels import (
    build_panels, cell_summary, file_hash, load_inputs, read_panels,
    save_json, select_panel, source_files,
)
import pandas as pd

RUN = ROOT / 'runs/thread_criticism_sol_review_20260910'
OUTPUT = RUN / 'outputs/analysis'
CATEGORIES = {'CONTENT', 'STYLE_CONDUCT', 'PARTICIPATION'}
TARGETS = {'DIRECT_PARENT', 'OTHER_PREVIOUS', 'GENERAL', 'UNRESOLVED'}


def parent_outcome(label, categories, targets):
    """Count any category directed at the parent; never double-count overlap."""
    categories = [] if pd.isna(categories) or categories == '' else categories.split(';')
    targets = json.loads(targets)
    if not isinstance(targets, dict) or set(categories) != set(targets):
        raise ValueError('Category and target keys differ')
    if len(categories) != len(set(categories)) or not set(categories) <= CATEGORIES:
        raise ValueError('Invalid criticism categories')
    if not set(targets.values()) <= TARGETS:
        raise ValueError('Invalid criticism target')
    if label not in ('YES', 'NO') or (label == 'YES') != bool(categories):
        raise ValueError('Inconsistent binary criticism label')
    return 'YES' if 'DIRECT_PARENT' in targets.values() else 'NO'


def attach_labels(panel, contents, annotations):
    human_comments = contents.loc[contents.author_type.eq('H') &
                                  contents.comment_to.notna() & contents.comment_to.ne(-1)]
    if annotations.contribution_index.duplicated().any():
        raise ValueError('Duplicate Sol contribution IDs')
    if set(annotations.contribution_index) != set(human_comments.row_id):
        raise ValueError('Sol labels must cover the complete human-comment cohort')
    check = human_comments[['row_id', 'group', 'tweet']].merge(
        annotations, left_on='row_id', right_on='contribution_index',
        validate='one_to_one', suffixes=('_source', '_sol'))
    if not check.group_source.eq(check.group_sol).all() or not check.tweet.map(redact).eq(check.focal_text).all():
        raise ValueError('Sol text or authoritative community does not match source')
    labels = annotations.copy()
    labels['machine_label'] = [parent_outcome(r.criticism_label, r.criticism_categories, r.targets)
                               for r in labels.itertuples()]
    result = panel.drop(columns='machine_label').merge(
        labels[['contribution_index', 'machine_label']], left_on='row_id',
        right_on='contribution_index', validate='one_to_one', how='left')
    if result.machine_label.isna().any():
        raise ValueError('An eligible reply lacks a Sol label')
    return result.drop(columns='contribution_index')


def main():
    paths = [*source_files(), Path(__file__),
             Path(__file__).with_name('annotate_recommendations.py'),
             RUN/'annotations.csv', RUN/'annotations.json']
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    if (OUTPUT/'prepared.json').exists():
        _, meta = read_panels(OUTPUT)
        if meta['sources'] != sources:
            raise ValueError('Prepared source set changed')
        print(json.dumps(meta['rows']), flush=True)
        return
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError('Refusing to overwrite nonempty unprepared output')
    contents, human, events = load_inputs()
    panels, _ = build_panels(contents, human, events)
    ann = pd.read_csv(RUN/'annotations.csv', encoding='utf-8-sig')
    panels['criticism'] = attach_labels(panels['criticism'], contents, ann)
    # The existing reader expects all three panel files. Only criticism is used.
    panels['uptake'] = panels['uptake'].iloc[:0]
    panels['rare'] = panels['rare'].iloc[:0]
    frame, audit = select_panel(panels, 'criticism', 'primary')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, table in panels.items():
        path = OUTPUT/f'{name}.parquet'
        table.to_parquet(path, index=False)
        hashes[path.name] = file_hash(path)
    cell_summary(frame).to_csv(OUTPUT/'planned_cells.csv', index=False)
    audit.to_csv(OUTPUT/'planned_attrition.csv', index=False)
    if sources != {str(p.relative_to(ROOT)): file_hash(p) for p in paths}:
        raise ValueError('Sources changed during preparation')
    meta = dict(sources=sources, panels=hashes, rows={k: len(v) for k,v in panels.items()},
                annotation_comments=len(ann), analysis='criticism',
                outcome='Any CONTENT, STYLE_CONDUCT or PARTICIPATION category directed at the direct parent',
                other_targets='Criticism solely of other, general or unresolved targets counts NO for the direct-parent outcome',
                measurements='Sol machine labels; independent human validation pending',
                interpretation='Exploratory ecological associations; one community per condition',
                unused_panels='uptake and rare are empty; this run supports criticism only')
    save_json(OUTPUT/'prepared.json', meta)
    read_panels(OUTPUT)
    print(cell_summary(frame).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

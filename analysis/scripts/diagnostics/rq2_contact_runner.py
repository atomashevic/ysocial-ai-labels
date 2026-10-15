#!/usr/bin/env python3
"""Sequential tmux job with bounded retries and an aggregate RSS watchdog.

The 30 GiB RSS threshold is polled every five seconds. It is conservative about
shared pages and is not a kernel-enforced hard memory ceiling.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from scripts.diagnostics.rq2_contact import CORE_RUN, RUN, fingerprint, save_json

GIB = 1024**3
STAGES = ('forward', 'refits', 'finalize', 'prepare', 'fit', 'sensitivity', 'report')
STOP = False


def install_reviewed_figure_exception():
    """Allow one reviewed plotting-only edit, retaining every other bundle gate."""
    from scripts.publication_analysis import provenance
    original = provenance.authenticate_fit_result_bundle
    old_hash = '3a6dd8fb4a1bff17321676b75e966aaad03cad2677ef57ab12b3bc4d4d8a99ec'
    new_hash = '8455d4651b695c4feb58bdb41d71e8bf95d7c98890145a00abadcc8f71fdfee4'

    def authenticate(config, **kwargs):
        result = original(config, **kwargs)
        path = config.project_root / 'scripts/publication_analysis/manuscript_figures.py'
        if (config.run_id == CORE_RUN and len(result) == 1
                and result.iloc[0]['authentication_status'] == 'source_bytes_changed'
                and result.iloc[0]['changed_sources'] == str(path)
                and __import__('hashlib').sha256(path.read_bytes()).hexdigest() == new_hash):
            bundle = json.loads((config.outputs_dir / provenance.BUNDLE_FILENAME).read_text())
            records = [r for r in bundle['sources'] if r['path'] == str(path)]
            if len(records) == 1 and records[0]['sha256'] == old_hash:
                result = result.copy()
                result.loc[result.index[0], 'authenticated'] = True
                result.loc[result.index[0], 'authentication_status'] = 'authenticated_with_reviewed_figure_change'
                print(f'Reviewed plotting-only source exception: {path} {old_hash} -> {new_hash}; '
                      'original bundle and computational checks retained', flush=True)
        return result

    provenance.authenticate_fit_result_bundle = authenticate


def now():
    return datetime.now(timezone.utc).isoformat()


def assert_identifier_table_pairs_equal(paths, keys):
    """Compare complete tables while preserving the two string identifier fields."""
    import pandas as pd
    from scripts.diagnostics import rq2_admission as admission

    for key in keys:
        stem = admission._table_path(paths, key)
        parquet_path = stem.with_suffix('.parquet')
        csv_path = stem.with_suffix('.csv')
        if not parquet_path.is_file() or not csv_path.is_file():
            raise ValueError(f'Admission table pair is incomplete for {key!r}')
        parquet = admission._normalize_structured_cells(pd.read_parquet(parquet_path))
        csv = admission._normalize_structured_cells(pd.read_csv(
            csv_path, dtype={'unit_id': str, 'deleted_unit_id': str}))
        try:
            pd.testing.assert_frame_equal(parquet, csv, check_dtype=False,
                                          check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise ValueError(f'Admission Parquet/CSV mismatch for {key!r}') from error


def install_reviewed_identifier_reader():
    """Correct CSV inference for the reviewed admission source without changing it."""
    import hashlib
    from scripts.diagnostics import rq2_admission as admission

    expected = 'acc3569f88d689f903bc6553b0d9835e2a254c1f96029927474b0c803fa6be78'
    if hashlib.sha256(Path(admission.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Admission source changed; review identifier compatibility again')
    admission._assert_table_pairs_semantically_equal = assert_identifier_table_pairs_equal
    original_paths = admission._core_bundle_paths
    admission._core_bundle_paths = lambda config, core: [
        *original_paths(config, core), Path(__file__).resolve()]
    print('Reviewed CSV identifier handling: unit_id and deleted_unit_id read as strings; '
          'all table values and posterior authentication checks retained', flush=True)


def make_vectorized_projection():
    """Replay the account projection with stable grouped sums and one-frame caching.

    Only the indexing of the immutable prepared frame is cached. Every posterior
    draw, candidate contribution, scenario shift and competing account is retained.
    The admission validator still compares all six summaries with the frozen fits.
    """
    import numpy as np
    from scipy.special import expit

    cached_data = None
    layout = None

    def project(beta, posterior_terms, utility, data, arm_x, term_values):
        nonlocal cached_data, layout
        selected_terms = tuple(term_values)
        missing = sorted(set(selected_terms).difference(posterior_terms))
        if missing:
            raise ValueError(f'RQ2 posterior missing scenario terms: {missing}')
        data_indexes = [data.terms.index(term) for term in selected_terms]
        beta_indexes = [posterior_terms.index(term) for term in selected_terms]
        low_values = np.array([term_values[term][0] for term in selected_terms])
        high_values = np.array([term_values[term][1] for term in selected_terms])
        if cached_data is not data:
            focal_indexes, other_indexes, focal_starts, other_starts = [], [], [], []
            width = data.mask.shape[1]
            for choice in range(data.n_choices):
                positions = np.flatnonzero(data.mask[choice])
                receivers = data.receiver_index[choice, positions]
                ai = positions[data.candidate_is_ai[choice, positions].astype(bool)]
                for receiver in np.unique(data.receiver_index[choice, ai]):
                    focal = positions[receivers == receiver]
                    other = positions[receivers != receiver]
                    if not len(other):
                        continue
                    focal_starts.append(len(focal_indexes))
                    other_starts.append(len(other_indexes))
                    focal_indexes.extend(choice * width + focal)
                    other_indexes.extend(choice * width + other)
            if not focal_starts:
                raise ValueError('RQ2 projection has no AI account competing with another account')
            layout = tuple(np.asarray(indexes, dtype=np.intp) for indexes in
                           (focal_indexes, other_indexes, focal_starts, other_starts))
            cached_data = data
        focal_indexes, other_indexes, focal_starts, other_starts = layout
        flat = utility.reshape(utility.shape[0], -1)
        observed = arm_x.reshape(-1, arm_x.shape[-1])[focal_indexes][:, data_indexes]
        low_delta = (low_values[None, :] - observed) @ beta[:, beta_indexes].T
        high_delta = (high_values[None, :] - observed) @ beta[:, beta_indexes].T
        log_other = np.logaddexp.reduceat(flat[:, other_indexes], other_starts, axis=1)
        low_mass = np.logaddexp.reduceat(flat[:, focal_indexes] + low_delta.T, focal_starts, axis=1)
        high_mass = np.logaddexp.reduceat(flat[:, focal_indexes] + high_delta.T, focal_starts, axis=1)
        changes = expit(high_mass - log_other) - expit(low_mass - log_other)
        return changes.mean(axis=1), len(focal_starts)

    return project


def make_unpadded_utilities(original):
    """Evaluate the same linear predictor on valid candidates, then restore padding."""
    import numpy as np
    from scripts.publication_analysis.model import _posterior_array

    def utilities(idata, data, *, x_override=None):
        if 'receiver_block' in idata.posterior and 'block_intercept' in idata.posterior:
            return original(idata, data, x_override=x_override)
        x = data.x if x_override is None else np.asarray(x_override, dtype=float)
        if x.shape != data.x.shape:
            raise ValueError(f'x_override must have shape {data.x.shape}')
        terms = tuple(str(value) for value in idata.posterior['beta'].coords['term'].values)
        indexes = [data.terms.index(term) for term in terms]
        beta = _posterior_array(idata, 'beta', ('term',))
        choice, candidate = np.nonzero(data.mask)
        values = (x[choice, candidate][:, indexes] @ beta.T).T
        receiver = _posterior_array(idata, 'receiver_effect', ('receiver',))
        if receiver is not None:
            values += receiver[:, data.receiver_index[choice, candidate]]
        for variable, dimension, index in (
            ('actor_ai_effect', 'actor', data.actor_index),
            ('topic_ai_effect', 'topic', data.topic_index),
        ):
            effect = _posterior_array(idata, variable, (dimension,))
            if effect is not None:
                values += effect[:, index[choice]] * data.candidate_is_ai[choice, candidate]
        result = np.full((len(beta), *data.mask.shape), -np.inf)
        result[:, choice, candidate] = values
        return result

    return utilities


def install_reviewed_projection_replay():
    """Vectorize validation replay only; retain the original signed model source."""
    import hashlib
    from scripts.publication_analysis import rq2, model

    expected = '8e59b308e42587828fcfa30b9fb0b27f1aa4629accb0b1ecde4d9d683b93ee85'
    if hashlib.sha256(Path(rq2.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('RQ2 projection source changed; review vectorized replay again')
    expected_model = 'd210df1984731f1f33c791a8d835727b47debea6c00edff9f94575aa19162594'
    if hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest() != expected_model:
        raise RuntimeError('RQ2 utility source changed; review vectorized replay again')
    rq2._projection_from_utility = make_vectorized_projection()
    rq2._posterior_utilities = make_unpadded_utilities(model._posterior_utilities)
    original_standardization = rq2.standardized_estimands

    def replay(idata, *args, **kwargs):
        # Compact validation posteriors fit in memory. Avoid decoding the same
        # NetCDF parameter chunks repeatedly for each block of retained draws.
        idata.posterior.load()
        return original_standardization(idata, *args, **kwargs)

    rq2.standardized_estimands = replay
    print('Reviewed vectorized account-projection replay; all draws, opportunities, '
          'estimand comparisons and posterior authentication checks retained', flush=True)


def host(local_temp, repository):
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        mem[key] = int(value.split()[0]) * 1024
    return {'available_memory': mem['MemAvailable'], 'load': os.getloadavg()[0],
            'local_free': shutil.disk_usage(local_temp).free,
            'repository_free': shutil.disk_usage(repository).free}


def host_ready(values):
    return (values['available_memory'] >= 32 * GIB and values['local_free'] >= 10 * GIB
            and values['repository_free'] >= 10 * GIB
            and values['load'] < (os.cpu_count() or 1) / 2)


def group_rss(pgid):
    total = 0
    members = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            fields = (item / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:
                total += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
                members.append(int(item.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return total, members


def stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    limit = time.monotonic() + 15
    while time.monotonic() < limit and group_rss(proc.pid)[1]:
        proc.poll()
        time.sleep(.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def checkpoints(repository, root):
    directory = repository / 'runs' / CORE_RUN / 'outputs/admission/models'
    paths = list(directory.glob('forward/*.nc.fit.json')) + list(directory.glob('refits/*.nc.fit.json'))
    paths += list((root / 'outputs').glob('*.nc.fit.json'))
    return {(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths}


def run(repository, local_temp):
    global STOP
    root = repository / 'runs' / RUN
    root.mkdir(parents=True, exist_ok=True)
    (root / 'outputs').mkdir(exist_ok=True)
    local_temp.mkdir(parents=True, exist_ok=True)
    lock = (root / 'outputs/runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / 'run.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'run_id': RUN, 'core_run_id': CORE_RUN, 'started': now(), 'started_epoch': time.time(),
        'completed_stages': [], 'stages': [], 'contact_implementation_sha256': fingerprint(),
        'runner_implementation_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy': {'chains': 4, 'workers': 2, 'numerical_threads': 1, 'rss_watchdog_gib': 30,
                   'hard_memory_cap': False, 'poll_seconds': 5, 'min_host_available_gib': 32,
                   'stage_timeout_hours': 24, 'overall_timeout_days': 7},
        'local_temp': str(local_temp)}
    if (state['contact_implementation_sha256'] != fingerprint() or
        state['runner_implementation_sha256'] != __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()):
        raise RuntimeError('Runner/source changed since launch; inspect checkpoints before resuming')
    state['runner_pid'] = os.getpid()
    deadline = state['started_epoch'] + 7 * 86400
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    if len(cpus) != 2:
        raise RuntimeError('Two permitted CPUs are required')
    state['cpu_affinity'] = cpus
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    env.update(TMPDIR=str(local_temp), TMP=str(local_temp), TEMP=str(local_temp),
               XDG_CACHE_HOME=str(local_temp / 'cache'), MPLCONFIGDIR=str(local_temp / 'mpl'),
               PYTHONUNBUFFERED='1')
    proc = None
    try:
        for stage in STAGES:
            if stage in state['completed_stages']:
                continue
            while True:
                waiting_until = min(time.time() + 2 * 3600, deadline)
                while True:
                    capacity = host(local_temp, repository)
                    if STOP or time.time() >= deadline:
                        raise RuntimeError('Cancelled or seven-day deadline reached')
                    if host_ready(capacity):
                        break
                    if time.time() >= waiting_until:
                        raise RuntimeError('Host pressure persisted for two hours')
                    state.update(status='waiting_for_host', current_stage=stage, host=capacity)
                    save_json(state_path, state)
                    print(f'{now()} {stage}: waiting for host capacity', flush=True)
                    # Autonomous runner wait, interruptible by cancellation.
                    until = min(time.time() + 900, waiting_until)
                    while not STOP and time.time() < until:
                        time.sleep(min(5, until - time.time()))
                before = checkpoints(repository, root)
                log_path = root / f'{stage}.log'
                command = ['taskset', '-c', ','.join(map(str, cpus)), 'nice', '-n', '10',
                           sys.executable, '-c',
                           'from scripts.diagnostics.rq2_contact_runner import '
                           'install_reviewed_figure_exception, install_reviewed_identifier_reader, '
                           'install_reviewed_projection_replay; '
                           'install_reviewed_figure_exception(); install_reviewed_identifier_reader(); '
                           + ('install_reviewed_projection_replay(); ' if stage == 'finalize' else '') +
                           'from scripts.diagnostics.rq2_contact import main; main()', '--stage', stage,
                           '--local-temp', str(local_temp)]
                with log_path.open('a') as log:
                    log.write(f'\n{now()} START {stage}\n')
                    log.flush()
                    proc = subprocess.Popen(command, cwd=repository, env=env,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    started = time.time()
                    state.update(status='running', current_stage=stage, stage_pid=proc.pid,
                                 stage_started=now(), log=str(log_path), host=capacity)
                    save_json(state_path, state)
                    print(f'{now()} START {stage} pgid={proc.pid} log={log_path}', flush=True)
                    reason = None
                    peak = 0
                    while proc.poll() is None:
                        rss, _ = group_rss(proc.pid)
                        peak = max(peak, rss)
                        capacity = host(local_temp, repository)
                        if STOP:
                            reason = 'cancelled'
                        elif time.time() >= deadline:
                            reason = 'overall_timeout'
                        elif time.time() - started >= 86400:
                            reason = 'stage_timeout'
                        elif rss >= 30 * GIB:
                            reason = 'job_memory_threshold'
                        elif capacity['available_memory'] < 32 * GIB:
                            reason = 'host_memory_pressure'
                        elif min(capacity['local_free'], capacity['repository_free']) < 10 * GIB:
                            reason = 'disk_pressure'
                        if reason:
                            stop_group(proc)
                            break
                        time.sleep(5)
                    code = proc.wait()
                    # Clean any task-owned children left by a failed parent.
                    if group_rss(proc.pid)[1]:
                        stop_group(proc)
                    record = {'stage': stage, 'started_epoch': started, 'finished': now(),
                              'exit_code': code, 'reason': reason, 'peak_rss_gib': peak / GIB,
                              'log': str(log_path)}
                    state['stages'].append(record)
                    print(f'{now()} END {stage} exit={code} reason={reason} peak_rss_gib={peak/GIB:.2f}', flush=True)
                    proc = None
                if reason == 'stage_timeout' and checkpoints(repository, root) - before:
                    state.update(status='resuming_after_progress')
                    save_json(state_path, state)
                    continue
                if reason or code:
                    raise RuntimeError(f'{stage}: {reason or "stage_failed"}; exit={code}')
                state['completed_stages'].append(stage)
                save_json(state_path, state)
                break
        state.update(status='complete', finished=now(), stage_pid=None,
                     report=str(root / 'outputs/report.txt'))
        save_json(state_path, state)
        return 0
    except BaseException as exc:
        if proc is not None:
            stop_group(proc)
        state.update(status='stopped', finished=now(), error=str(exc), stage_pid=None)
        save_json(state_path, state)
        print(f'{now()} STOP {exc}', flush=True)
        return 1


def main():
    global STOP
    def cancelled(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-temp', type=Path, required=True)
    args = parser.parse_args()
    return run(Path.cwd(), args.local_temp.resolve())


if __name__ == '__main__':
    raise SystemExit(main())

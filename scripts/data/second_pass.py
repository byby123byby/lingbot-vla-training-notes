"""Local numerical review; statistical flags do not establish failed demonstrations."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
original = json.loads((ROOT / 'episode_decisions.json').read_text())
lookup = {(r['task'], r['episode_index']): r for r in original}
mapping = {Path(k).name: v for k, v in json.loads((ROOT / 'episode_id_mapping.json').read_text()).items()}
JOINTS = list(range(6)) + list(range(7, 13))
results, contexts, source_hashes = [], [], {}

def read_table(paths):
    return pa.concat_tables([pq.read_table(p) for p in sorted(paths)])

def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(lengths.max()) if len(lengths) else 0

for folder in sorted((ROOT / 'filtered_lerobot').iterdir()):
    if not folder.is_dir():
        continue
    task = folder.name.removesuffix('_joint_v30')
    info = json.loads((folder / 'meta/info.json').read_text())
    data = read_table(folder.glob('data/**/*.parquet'))
    meta = {r['episode_index']: r for r in read_table(folder.glob('meta/episodes/**/*.parquet')).to_pylist()}
    labels_list = pq.read_table(folder / 'meta/tasks.parquet').to_pylist()
    labels = {r['task_index']: r['task'] for r in labels_list}
    assert len(labels) == len(labels_list)
    for f in sorted(folder.glob('**/*.parquet')):
        source_hashes[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    x_all = np.array(data['observation.state'].to_pylist(), dtype=np.float64)
    a_all = np.array(data['action'].to_pylist(), dtype=np.float64)
    ep_all = data['episode_index'].to_numpy()
    index_all = data['index'].to_numpy()
    assert np.array_equal(index_all, np.arange(len(data)))
    assert len(data) == info['total_frames']
    assert len(meta) == len(mapping[folder.name]) == info['total_episodes']
    for ep in sorted(meta):
        mask = ep_all == ep
        x, a = x_all[mask], a_all[mask]
        n = len(a)
        old_id = mapping[folder.name][str(ep)]
        prior = lookup[task, old_id]
        m = meta[ep]
        errors = []
        if n != m['length'] or n != prior['frames']: errors.append('length_mismatch')
        if a.shape != (n, 14) or x.shape != (n, 14): errors.append('invalid_shape')
        if not np.isfinite(a).all() or not np.isfinite(x).all(): errors.append('nonfinite')
        if not np.array_equal(data['frame_index'].to_numpy()[mask], np.arange(n)): errors.append('frame_indices')
        if not np.array_equal(index_all[mask], np.arange(m['dataset_from_index'], m['dataset_to_index'])): errors.append('metadata_indices')
        if not np.allclose(data['timestamp'].to_numpy()[mask], np.arange(n) / info['fps'], atol=1e-4): errors.append('timestamps')
        tids = set(map(int, data['task_index'].to_numpy()[mask]))
        if any(i not in labels or labels[i] not in m['tasks'] for i in tids): errors.append('instruction_mapping')
        if any(not isinstance(s, str) or not s.strip() for s in m['tasks']): errors.append('empty_instruction')
        digest = hashlib.sha256(x.tobytes() + a.tobytes()).hexdigest()
        if digest != prior['state_action_sha256']: errors.append('content_hash_mismatch')
        dx, da = np.diff(x, axis=0), np.diff(a, axis=0)
        if np.max(np.abs(dx)) == 0: errors.append('entire_episode_static')
        step = np.abs(da[:, JOINTS])
        t, k = np.unravel_index(np.argmax(step), step.shape)
        joint = JOINTS[k]
        jerk = np.diff(a[:, JOINTS], n=3, axis=0) * info['fps'] ** 3
        energy = np.sum(jerk ** 2, axis=1)
        count = max(1, int(np.ceil(len(energy) * .05)))
        concentration = np.sort(energy)[-count:].sum() / max(energy.sum(), 1e-30)
        lag_errors = {str(lag): float(np.sqrt(np.mean((a[:n-lag or None, JOINTS] - x[lag:, JOINTS]) ** 2))) for lag in range(4)}
        r = dict(task=task, original_episode=old_id, filtered_episode=int(ep), frames=n,
                 prior_review=prior['review_only'], errors=errors,
                 max_joint_step=float(step[t, k]), peak_frame=int(t + 1), peak_joint=joint,
                 peak_fraction=float((t + 1) / (n - 1)),
                 action_jerk_ratio=prior['action_jerk_ratio_to_task_median'],
                 jerk_energy_top_5pct_fraction=float(concentration),
                 static_run_seconds=longest_run(np.max(np.abs(dx), axis=1) < 1e-7) / info['fps'],
                 lag_rms=lag_errors, best_lag=int(min(lag_errors, key=lag_errors.get)),
                 gripper_action_min=float(a[:, [6, 13]].min()), gripper_action_max=float(a[:, [6, 13]].max()),
                 gripper_state_min=float(x[:, [6, 13]].min()), gripper_state_max=float(x[:, [6, 13]].max()),
                 sha256=digest)
        results.append(r)
        if r['prior_review'] or r['max_joint_step'] > .2:
            lo, hi = max(0, t - 3), min(n, t + 6)
            contexts.append(dict(task=task, original_episode=old_id, joint=joint,
                                 frames=list(range(lo, hi)), action=a[lo:hi, joint].tolist(),
                                 state=x[lo:hi, joint].tolist(), instructions=m['tasks']))

by_task = defaultdict(list)
for r in results: by_task[r['task']].append(r)
for task, rows in by_task.items():
    modal_lag = Counter(r['best_lag'] for r in rows).most_common(1)[0][0]
    for key in ['frames', 'max_joint_step', 'static_run_seconds']:
        values = np.log(np.maximum([r[key] for r in rows], 1e-9))
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        for r, v in zip(rows, values):
            r[key + '_robust_z'] = float(.67448975 * (v - median) / mad) if mad > 1e-12 else None
    for r in rows:
        # Review priorities only: no unverified biomechanical or success labels.
        r['flags'] = []
        if r['max_joint_step'] > .2: r['flags'].append('joint_step_above_0.2_rad_review')
        if r['static_run_seconds'] >= 1: r['flags'].append('static_run_at_least_1s_review')
        if r['best_lag'] != modal_lag: r['flags'].append('task_modal_lag_differs_review')
        if abs(r['frames_robust_z'] or 0) > 4.5: r['flags'].append('duration_outlier_review')
        r['decision'] = 'exclude_integrity_error' if r['errors'] else 'retain'

assert len(results) == 2479 and sum(r['frames'] for r in results) == 544167
assert sum(r['prior_review'] for r in results) == 33
hash_groups = defaultdict(list)
for r in original: hash_groups[r['state_action_sha256']].append([r['task'], r['episode_index']])
duplicates = [v for v in hash_groups.values() if len(v) > 1]
summary = dict(episodes_checked=len(results), frames_checked=sum(r['frames'] for r in results), tasks=len(by_task),
               prior_review_checked=sum(r['prior_review'] for r in results),
               hard_error_episodes=sum(bool(r['errors']) for r in results),
               additional_exclusions=sum(r['decision'] != 'retain' for r in results),
               original_2500_duplicate_groups=len(duplicates),
               best_lag_counts=dict(Counter(r['best_lag'] for r in results)),
               review_flag_counts=dict(Counter(f for r in results for f in r['flags'])),
               flagged_episodes=sum(bool(r['flags']) for r in results),
               prior_review_flagged=sum(bool(r['flags']) and r['prior_review'] for r in results),
               maximum_static_run_seconds=max(r['static_run_seconds'] for r in results),
               maximum_joint_step=max(r['max_joint_step'] for r in results),
               new_cloud_credits_used=0,
               caveat='Numerical and metadata review only; videos not present locally. Review flags are not rejection labels.')
for name, obj in [('summary', summary), ('episode_review', results), ('peak_contexts', contexts),
                  ('source_sha256', source_hashes), ('duplicate_groups', duplicates)]:
    (OUT / (name + '.json')).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))

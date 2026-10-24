"""Reproduce local peak-context classifications; never edits the training selection."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
rows = json.loads((root / 'episode_review.json').read_text())
contexts = {(r['task'], r['original_episode']): r
            for r in json.loads((root / 'peak_contexts.json').read_text())}
isolated, continuous = [], []
for r in rows:
    if r['max_joint_step'] <= .2:
        continue
    c = contexts[r['task'], r['original_episode']]
    action = c['action']
    steps = [b - a for a, b in zip(action, action[1:])]
    peak = max(range(len(steps)), key=lambda i: abs(steps[i]))
    prior_hold = peak >= 2 and max(abs(v) for v in steps[peak - 2:peak]) < 1e-7
    needs_video = prior_hold and r['jerk_energy_top_5pct_fraction'] > .95
    record = {k: r[k] for k in ['task', 'original_episode', 'filtered_episode', 'frames',
              'peak_frame', 'peak_joint', 'max_joint_step', 'action_jerk_ratio',
              'jerk_energy_top_5pct_fraction']}
    record.update(status='retain_pending_video_review' if needs_video else 'retain_continuous_motion',
                  preceding_two_steps_static=prior_hold, context=c)
    (isolated if needs_video else continuous).append(record)
for name, records in [('priority_video_review', isolated), ('continuous_motion_review', continuous)]:
    assert records == json.loads((root / (name + '.json')).read_text())
print(json.dumps({'priority_video_review': len(isolated), 'continuous_motion': len(continuous),
                  'saved_classifications_match': True}))

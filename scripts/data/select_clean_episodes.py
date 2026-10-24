"""Conservative experimental filter; thresholds are not official quality labels."""
from pathlib import Path
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT=Path('/workspace/clean_screen_20261024')
keys=['action_jerk_rms','state_jerk_rms','action_velocity_peak']
records=[]; tasks=[]; selection={}; contexts=[]
for f in sorted(OUT.glob('*.json')):
    data=json.loads(f.read_text())
    if 'episodes' not in data or 'summary' not in data: continue
    rows=data['episodes']; summary=data['summary']; task=summary['task']
    for key in keys:
        values=np.log(np.maximum([r[key] for r in rows],1e-12))
        median=np.median(values); mad=np.median(np.abs(values-median))
        for i,r in enumerate(rows):
            r[key+'_robust_z']=float(.67448975*(values[i]-median)/mad) if mad>1e-12 else 0.
    jerk_median=float(np.median([r['action_jerk_rms'] for r in rows]))
    for r in rows:
        r['action_jerk_ratio_to_task_median']=r['action_jerk_rms']/jerk_median
        severe=(all(r[k+'_robust_z']>4.5 for k in keys) and r['action_jerk_ratio_to_task_median']>3)
        r['keep']=not r['errors'] and not severe
        r['reason']='integrity_error' if r['errors'] else ('joint_motion_outlier' if severe else 'keep')
        r['review_only']=not severe and r['action_jerk_rms_robust_z']>4.5
    kept=[r['episode_index'] for r in rows if r['keep']]
    assert len(kept)>=40, ('too many excluded; review before training',task)
    selection[rows[0]['dataset']]=kept
    summary=dict(summary,kept=len(kept),excluded=50-len(kept),kept_frames=sum(r['frames'] for r in rows if r['keep']))
    tasks.append(summary); records.extend(rows)
    excluded=[r for r in rows if not r['keep']]
    if excluded:
        p=Path(rows[0]['dataset'])
        df=pa.concat_tables([pq.read_table(x,columns=['episode_index','frame_index','action','observation.state']) for x in sorted(p.glob('data/**/*.parquet'))]).to_pandas()
        for r in excluded:
            ep=df[df.episode_index==r['episode_index']]
            a=np.stack(ep.action); x=np.stack(ep['observation.state']); j=np.r_[0:6,7:13]
            diff=np.abs(np.diff(a[:,j],axis=0)); t,k=np.unravel_index(np.argmax(diff),diff.shape); joint=int(j[k])
            contexts.append({'task':task,'episode_index':r['episode_index'],'peak_step':int(t+1),'joint':joint,
              'max_position_step':float(diff[t,k]),'action_context':a[max(0,t-2):t+4,joint].tolist(),
              'state_context':x[max(0,t-2):t+4,joint].tolist()})
assert len(tasks)==50 and len(records)==2500
policy={'version':'conservative_v1','data':'official Aloha-AgileX clean 50 tasks x 50 episodes',
 'hard_exclusion':'invalid state/action/index/timestamp/metadata or sampled camera decoding failure',
 'statistical_exclusion':'within each task, log-metric robust Z > 4.5 for action jerk RMS, state jerk RMS and peak joint velocity, AND action jerk RMS > 3 times task median',
 'derivatives':'finite differences at metadata fps; joints 0:6 and 7:13, grippers excluded; no environment calls',
 'robust_z':'0.67448975 * (log(value) - median(log(values))) / MAD(log(values)); zero if MAD <= 1e-12',
 'limitation':'heuristic experimental filtering; improvement in task success is not yet established; camera checks sample 3 frames per camera per episode'}
for name,value in [('selection.json',selection),('selection_policy.json',policy),('episode_decisions.json',records),('jump_contexts.json',contexts),('task_summary.json',tasks)]:
 (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))
(OUT/'train_paths_corrected.txt').write_text(''.join('robotwin '+p+'\n' for p in selection))
summary={'total_episodes':len(records),'kept':sum(r['keep'] for r in records),'excluded':sum(not r['keep'] for r in records),
 'review_only_kept':sum(r['review_only'] for r in records),'total_frames':sum(r['frames'] for r in records),
 'kept_frames':sum(r['frames'] for r in records if r['keep']),'tasks':len(tasks),'minimum_episodes_per_task':min(t['kept'] for t in tasks)}
(OUT/'selection_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)

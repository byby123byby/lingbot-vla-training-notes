"""Conservative arm/instruction review on the existing local 2478-episode selection."""
from pathlib import Path
from collections import Counter
import hashlib, json, re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path('outputs/data-screening/video-review/clean_screen_final_20261024')
OUT=Path('outputs/data-screening/semantic-pass')
mapping={Path(k).name:v for k,v in json.loads((ROOT/'episode_id_mapping.json').read_text()).items()}
rows=[]; hashes={}
arm_pattern=re.compile(r'\b(left|right)[ -]+(?:arm|hand)\b',re.I)
for folder in sorted((ROOT/'filtered_lerobot').iterdir()):
    if not folder.is_dir(): continue
    paths=sorted(folder.glob('data/**/*.parquet'))
    table=pa.concat_tables([pq.read_table(p) for p in paths])
    meta=pq.read_table(folder/'meta/tasks.parquet').to_pylist()
    labels={r['task_index']:r['task'] for r in meta}
    for p in paths+[folder/'meta/tasks.parquet',folder/'meta/info.json']:
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    a=np.array(table['action'].to_pylist()); s=np.array(table['observation.state'].to_pylist())
    eps=table['episode_index'].to_numpy(); tids=table['task_index'].to_numpy()
    for ep in sorted(set(eps)):
        mask=eps==ep; act=a[mask]; state=s[mask]
        instructions=[labels[int(i)] for i in sorted(set(tids[mask]))]
        named=sorted(set(x.lower() for text in instructions for x in arm_pattern.findall(text)))
        span=np.ptp(state,axis=0)
        activity={side:float(span[js].max()) for side,js in [('left',slice(0,6)),('right',slice(7,13))]}
        active=[side for side in activity if activity[side]>.1]
        inactive=[side for side in activity if activity[side]<1e-5]
        row=dict(task=folder.name.removesuffix('_joint_v30'),original_episode=mapping[folder.name][str(ep)],filtered_episode=int(ep),frames=len(act),instructions=instructions,named_arms=named,joint_range_max_rad=activity,decision='retain',review_flags=[])
        if len(named)==1 and named[0] in inactive and len(active)==1:
            row['review_flags'].append('named_arm_static_opposite_moves')
        if len(named)>1: row['review_flags'].append('multi_arm_instruction_requires_semantic_review')
        if row['task']=='place_bread_skillet':
            grip=state[:,[6,13]]; ag=act[:,[6,13]]
            # Task source: both grip, skillet stays held, opposite arm releases bread.
            releases=[i for i in range(2) if grip[-1,i]>.95 and ag[-1,i]>.95 and grip[:,i].min()<.05 and ag[:,i].min()<.05]
            holds=[i for i in range(2) if grip[-1,i]<.05 and ag[-1,i]<.05]
            resolved=(len(releases)==len(holds)==1 and releases[0]!=holds[0] and np.all(grip[0]>.95))
            row['bread_arm_from_release_and_task_source']=['left','right'][releases[0]] if resolved else None
            row['initial_grippers_lr']=grip[0].tolist();row['final_state_grippers_lr']=grip[-1].tolist();row['final_action_grippers_lr']=ag[-1].tolist()
            row['release_start_frame']=int(np.flatnonzero(grip[:,releases[0]]<.05)[-1]+1) if resolved else None
            if len(named)==1 and resolved and named[0]!=row['bread_arm_from_release_and_task_source']:
                row['decision']='exclude_bread_arm_instruction_conflict'
            elif named and not resolved: row['review_flags'].append('bread_arm_unresolved')
        rows.append(row)
assert len(rows)==2478 and sum(r['frames'] for r in rows)==543987
excluded=[r for r in rows if r['decision'].startswith('exclude')]
flags=[r for r in rows if 'named_arm_static_opposite_moves' in r['review_flags']]
summary=dict(episodes_checked=len(rows),tasks=len(set(r['task'] for r in rows)),episodes_with_explicit_arm=sum(bool(r['named_arms']) for r in rows),single_named_arm=sum(len(r['named_arms'])==1 for r in rows),multi_named_arm=sum(len(r['named_arms'])>1 for r in rows),bread_checked=sum(r['task']=='place_bread_skillet' for r in rows),additional_exclusions=len(excluded),additional_excluded_frames=sum(r['frames'] for r in excluded),static_named_arm_review_count=len(flags),new_cloud_credits=0)
for name,obj in [('episode_audit',rows),('additional_exclusions',excluded),('static_arm_review',flags),('source_sha256',hashes),('summary',summary)]:
    (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('BREAD EXCLUSIONS',[(r['original_episode'],r['frames']) for r in excluded])
for r in flags: print('STATIC_NAMED_ARM',r['task'],r['original_episode'],r['instructions'],r['joint_range_max_rad'])

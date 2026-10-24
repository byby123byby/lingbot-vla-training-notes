from pathlib import Path
import json,re,hashlib
import numpy as np
import pyarrow.parquet as pq
root=Path('outputs/data-screening/semantic-pass/clean_screen_semantic_20261024')
out=Path('outputs/data-screening/multi-arm-pass')
prior=json.load(open('outputs/data-screening/semantic-pass/episode_audit.json'))
target={(x['task'],x['original_episode']):x for x in prior if len(x['named_arms'])>1}
mapping={Path(k).name:v for k,v in json.loads((root/'episode_id_mapping.json').read_text()).items()};rows=[];sources=[]
def events(a,lo=.2,hi=.8):
    ev=[]
    for side,col in [('left',6),('right',13)]:
        g=a[:,col]; last='open' if g[0]>hi else 'closed' if g[0]<lo else 'unknown'
        for i in range(1,len(g)-2):
            typ='closed' if np.all(g[i:i+3]<lo) else 'open' if np.all(g[i:i+3]>hi) else None
            if typ and typ!=last:
                ev.append({'frame':i,'arm':side,'event':typ});last=typ
    return sorted(ev,key=lambda x:(x['frame'],x['arm']))
for task in sorted({x[0] for x in target}):
    folder=root/'filtered_lerobot'/(task+'_joint_v30');p=folder/'data/chunk-000/file-000.parquet'
    t=pq.read_table(p);s=np.array(t['observation.state'].to_pylist());a=np.array(t['action'].to_pylist());eps=t['episode_index'].to_numpy()
    sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    mp=mapping.get('/RoboTwin/data/lerobot/'+task+'_joint_v30',mapping.get(task+'_joint_v30'))
    for ep in sorted(set(eps)):
        old=int(mp[str(ep)]);key=(task,old)
        if key not in target:continue
        mask=eps==ep;st=s[mask];act=a[mask];ev=events(st);ae=events(act)
        r={'task':task,'original_episode':old,'filtered_episode':int(ep),'frames':len(st),'instructions':target[key]['instructions'],'named_arm_sequence':re.findall(r'\b(left|right)[ -]+(?:arm|hand)\b',' '.join(target[key]['instructions']),re.I),'state_events':ev,'action_events':ae,'strict_state_events':events(st,.05,.95),'initial_grip':st[0,[6,13]].tolist(),'final_grip':st[-1,[6,13]].tolist()}
        rows.append(r)
        print(task,old,'named',','.join(r['named_arm_sequence']),'events',' '.join(f"{e['arm'][0]}{e['event'][0]}@{e['frame']}" for e in ev))
assert len(rows)==122
(out/'events.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(out/'input_hashes.json').write_text(json.dumps(sources,indent=2))

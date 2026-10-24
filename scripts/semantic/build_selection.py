"""Build a new local version, preserving original video links and other task files."""
from pathlib import Path
import hashlib, json, shutil, tarfile
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path('outputs/data-screening/video-review/clean_screen_final_20261024')
OUT=Path('outputs/data-screening/semantic-pass/clean_screen_semantic_20261024')
CLOUD='/workspace/clean_screen_semantic_20261024'
AUDIT=OUT.parent
exclusions=json.loads((AUDIT/'additional_exclusions.json').read_text())
exclude_ids={r['original_episode'] for r in exclusions}
assert len(exclude_ids)==23 and all(r['task']=='place_bread_skillet' for r in exclusions)
OUT.mkdir(exist_ok=False)
selection=json.loads((ROOT/'selection.json').read_text())
key='/RoboTwin/data/lerobot/place_bread_skillet_joint_v30'
selection[key]=[i for i in selection[key] if i not in exclude_ids]
assert len(selection[key])==25 and sum(map(len,selection.values()))==2455
(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
shutil.copytree(ROOT/'filtered_lerobot',OUT/'filtered_lerobot')
# Preserve original absolute read-only video links from the prior cloud archive.
with tarfile.open(ROOT.parent/'clean_screen_final_20261024.tar.gz') as tar:
    for m in tar.getmembers():
        if m.issym():
            rel=Path(m.name).relative_to('clean_screen_final_20261024')
            assert rel.parts[0]=='filtered_lerobot' and rel.name=='videos'
            assert Path(m.linkname).is_absolute()
            (OUT/rel).symlink_to(m.linkname,target_is_directory=True)

mapping={Path(k).name:v for k,v in json.loads((ROOT/'episode_id_mapping.json').read_text()).items()}
name='place_bread_skillet_joint_v30'; src=ROOT/'filtered_lerobot'/name;dst=OUT/'filtered_lerobot'/name
oldmap=mapping[name]
keep=[int(k) for k,v in oldmap.items() if v not in exclude_ids]
idmap={old:new for new,old in enumerate(keep)}
table=pq.read_table(src/'data/chunk-000/file-000.parquet')
kept=table.filter(pc.is_in(table['episode_index'],value_set=pa.array(keep,type=pa.int64())))
kept=kept.set_column(kept.schema.get_field_index('episode_index'),table.schema.field('episode_index'),pa.array([idmap[i] for i in kept['episode_index'].to_pylist()],type=pa.int64()))
kept=kept.set_column(kept.schema.get_field_index('index'),table.schema.field('index'),pa.array(np.arange(len(kept))))
pq.write_table(kept,dst/'data/chunk-000/file-000.parquet')
meta=pq.read_table(src/'meta/episodes/chunk-000/file-000.parquet')

def aggregate(records):
    # Match the installed cloud version: counts-weighted episode quantiles are
    # approximations, not global quantiles. Reproduce ALL old statistics first.
    result={}
    features={k[6:].rsplit('/',1)[0] for k in records[0] if k.startswith('stats/')}
    for f in features:
        values={k[6:].rsplit('/',1)[1]:np.stack([np.asarray(m[k]) for m in records]) for k in records[0] if k.startswith('stats/'+f+'/')}
        means=values['mean']; counts=values['count']; total=counts.sum(0)
        while counts.ndim<means.ndim:counts=np.expand_dims(counts,-1)
        mean=(means*counts).sum(0)/total
        r={'min':values['min'].min(0),'max':values['max'].max(0),'mean':mean,
           'std':np.sqrt(((values['std']**2+(means-mean)**2)*counts).sum(0)/total),'count':total}
        for k,v in values.items():
            if k.startswith('q'):r[k]=(v*counts).sum(0)/total
        result[f]=r
    return result

oldstats=json.loads((src/'meta/stats.json').read_text())
for f,stats in aggregate(meta.to_pylist()).items():
    for k,v in stats.items():np.testing.assert_allclose(v,oldstats[f][k],rtol=1e-12,atol=1e-12)
newmeta=[];offset=0
for m in meta.to_pylist():
    old=m['episode_index']
    if old not in idmap:continue
    new=idmap[old];shift=offset-m['dataset_from_index']
    for k,v in list(m.items()):
        if k.startswith('stats/index/') and not k.endswith(('/count','/std')):m[k]=(np.asarray(v)+shift).tolist()
        if k.startswith('stats/episode_index/') and not k.endswith(('/count','/std')):m[k]=np.full_like(np.asarray(v),new).tolist()
    m['episode_index']=new;m['dataset_from_index']=offset;offset+=m['length'];m['dataset_to_index']=offset
    newmeta.append(m)
pq.write_table(pa.Table.from_pylist(newmeta,schema=meta.schema),dst/'meta/episodes/chunk-000/file-000.parquet')
info=json.loads((src/'meta/info.json').read_text());info.update(total_episodes=25,total_frames=len(kept),splits={'train':'0:25'})
(dst/'meta/info.json').write_text(json.dumps(info,indent=2))
(dst/'meta/stats.json').write_text(json.dumps(aggregate(newmeta),default=lambda x:x.tolist(),indent=2,allow_nan=False))
mapping[name]={str(new):oldmap[str(old)] for old,new in idmap.items()}
(OUT/'episode_id_mapping.json').write_text(json.dumps({CLOUD+'/filtered_lerobot/'+k:v for k,v in mapping.items()},indent=2))
(OUT/'filtered_train_paths.txt').write_text(''.join('robotwin '+CLOUD+'/filtered_lerobot/'+Path(k).name+'\n' for k in selection))
records=json.loads((ROOT/'episode_decisions.json').read_text())
for r in records:
    if r['task']=='place_bread_skillet' and r['episode_index'] in exclude_ids:
        assert r['keep'];r.update(keep=False,reason='instruction_arm_conflict_from_gripper_sequence_and_task_source',review_only=False)
(OUT/'episode_decisions.json').write_text(json.dumps(records,indent=2))
summary=dict(total_episodes=2500,kept=sum(r['keep'] for r in records),excluded=sum(not r['keep'] for r in records),kept_frames=sum(r['frames'] for r in records if r['keep']),tasks=50,minimum_episodes_per_task=25,new_exclusions=23,new_cloud_credits=0,bread_remaining=25)
assert summary['kept']==2455 and summary['kept_frames']==540182
(OUT/'selection_summary.json').write_text(json.dumps(summary,indent=2))
for f in ['official_place_bread_skillet.py']:shutil.copy2(ROOT/f,OUT/f)
for f in ['additional_exclusions.json','episode_audit.json','source_sha256.json']:shutil.copy2(AUDIT/f,OUT/f)
# This verification is for later execution with the original mount and training env.
code=(ROOT/'verify_filtered_data.py').read_text().replace('/workspace/clean_screen_final_20261024',CLOUD)
(OUT/'verify_on_cloud.py').write_text(code)
checks=[]
for folder in sorted((OUT/'filtered_lerobot').iterdir()):
    ds=pq.read_table(folder/'data/chunk-000/file-000.parquet')
    original=pq.read_table(ROOT/'filtered_lerobot'/folder.name/'data/chunk-000/file-000.parquet')
    if folder.name==name:original=original.filter(pc.is_in(original['episode_index'],value_set=pa.array(keep,type=pa.int64())))
    cols=['action','observation.state','frame_index','timestamp','task_index']
    assert ds.select(cols).equals(original.select(cols))
    assert ds['index'].to_pylist()==list(range(len(ds)))
    em=pq.read_table(folder/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert len(em)==len(selection['/RoboTwin/data/lerobot/'+folder.name])
    for new,m in enumerate(em):
        assert m['episode_index']==new and m['dataset_to_index']-m['dataset_from_index']==m['length']
        segment=ds.slice(m['dataset_from_index'],m['length'])
        assert set(segment['episode_index'].to_pylist())=={new}
        assert segment['frame_index'].to_pylist()==list(range(m['length']))
    if folder.name!=name:
        for f in (ROOT/'filtered_lerobot'/folder.name).rglob('*'):
            if f.is_file():assert f.read_bytes()==(folder/f.relative_to(ROOT/'filtered_lerobot'/folder.name)).read_bytes()
    checks.append(dict(task=folder.name,episodes=len(em),frames=len(ds),preserved_content=True,indices_valid=True))
assert sum(x['episodes'] for x in checks)==2455 and sum(x['frames'] for x in checks)==540182
(OUT/'local_verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(summary,indent=2))
print('Verified all retained values and indices; other 49 tasks byte-identical. Cloud video loader not run.')

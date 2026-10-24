"""Create a LeRobot v3 view of selected episodes; original videos remain read-only."""
from pathlib import Path
import json, shutil
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

OUT=Path('/workspace/clean_screen_20261024')
selection=json.loads((OUT/'selection.json').read_text())
viewroot=OUT/'filtered_lerobot'
viewroot.mkdir(exist_ok=False)
checks=[]; manifest=[]; mappings={}
for source,keep in selection.items():
    src=Path(source); dst=viewroot/src.name
    (dst/'data/chunk-000').mkdir(parents=True)
    (dst/'meta/episodes/chunk-000').mkdir(parents=True)
    (dst/'videos').symlink_to((src/'videos').resolve(),target_is_directory=True)
    for p in (src/'meta').iterdir():
        if p.is_file() and p.name not in ('info.json','stats.json'):
            shutil.copyfile(p,dst/'meta'/p.name)
    table=pa.concat_tables([pq.read_table(p) for p in sorted(src.glob('data/**/*.parquet'))])
    kept=table.filter(pc.is_in(table['episode_index'],value_set=pa.array(keep,type=pa.int64())))
    idmap={old:new for new,old in enumerate(keep)}
    ids=pa.array([idmap[x] for x in kept['episode_index'].to_pylist()],type=pa.int64())
    kept=kept.set_column(kept.schema.get_field_index('episode_index'),table.schema.field('episode_index'),ids)
    kept=kept.set_column(kept.schema.get_field_index('index'),table.schema.field('index'),pa.array(np.arange(len(kept))))
    pq.write_table(kept,dst/'data/chunk-000/file-000.parquet')
    meta=pa.concat_tables([pq.read_table(p) for p in sorted(src.glob('meta/episodes/**/*.parquet'))])
    newmeta=[]; offset=0; per_episode_stats=[]
    for m in meta.to_pylist():
        old=m['episode_index']
        if old not in idmap: continue
        new=idmap[old]; shift=offset-m['dataset_from_index']
        for k,v in list(m.items()):
            if k.startswith('stats/index/') and not k.endswith(('/count','/std')):
                m[k]=(np.asarray(v)+shift).tolist()
            if k.startswith('stats/episode_index/') and not k.endswith(('/count','/std')):
                m[k]=np.full_like(np.asarray(v),new).tolist()
        m['episode_index']=new
        m['dataset_from_index']=offset; offset+=m['length']; m['dataset_to_index']=offset
        for k in ['data/chunk_index','data/file_index','meta/episodes/chunk_index','meta/episodes/file_index']: m[k]=0
        stats={}
        for k,v in m.items():
            if k.startswith('stats/'):
                feature,stat=k[6:].rsplit('/',1)
                stats.setdefault(feature,{})[stat]=np.asarray(v)
        per_episode_stats.append(stats); newmeta.append(m)
    assert offset==len(kept)
    pq.write_table(pa.Table.from_pylist(newmeta,schema=meta.schema),dst/'meta/episodes/chunk-000/file-000.parquet')
    info=json.loads((src/'meta/info.json').read_text())
    info['total_episodes']=len(keep); info['total_frames']=len(kept); info['splits']={'train':f'0:{len(keep)}'}
    (dst/'meta/info.json').write_text(json.dumps(info,indent=2))
    stats=aggregate_stats(per_episode_stats)
    (dst/'meta/stats.json').write_text(json.dumps(stats,default=lambda x:x.tolist(),indent=2,allow_nan=False))
    ds=LeRobotDataset(repo_id=dst.name,root=dst,video_backend='pyav')
    assert len(ds)==len(kept)
    for idx in [0,len(ds)-1]:
        sample=ds[idx]
        assert sample['action'].shape==(14,) and sample['observation.state'].shape==(14,)
        for key,value in info['features'].items():
            if value['dtype']=='video': assert tuple(sample[key].shape)==tuple(value['shape'])
    checks.append({'dataset':str(dst),'episodes':len(keep),'frames':len(ds),'first_last_samples':'passed'})
    mappings[str(dst)]={str(new):old for old,new in idmap.items()}
    manifest.append('robotwin '+str(dst))
    print('FILTERED_DATASET_VERIFIED',dst.name,len(keep),len(ds),flush=True)
(OUT/'filtered_train_paths.txt').write_text('\n'.join(manifest)+'\n')
(OUT/'filtered_loader_checks.json').write_text(json.dumps(checks,indent=2))
(OUT/'episode_id_mapping.json').write_text(json.dumps(mappings,indent=2))
print('FILTERED_DATA_COMPLETE',len(checks),'datasets',flush=True)

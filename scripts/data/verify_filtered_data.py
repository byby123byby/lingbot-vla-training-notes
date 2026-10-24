from pathlib import Path
import json
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
out=Path('/workspace/clean_screen_20261024')
selection=json.loads((out/'selection.json').read_text()); checks=[]
columns=['action','observation.state','frame_index','timestamp','task_index']
for source,ids in selection.items():
    src=Path(source); dst=out/'filtered_lerobot'/src.name
    a=pa.concat_tables([pq.read_table(p) for p in sorted(src.glob('data/**/*.parquet'))])
    a=a.filter(pc.is_in(a['episode_index'],value_set=pa.array(ids,type=pa.int64())))
    b=pq.read_table(dst/'data/chunk-000/file-000.parquet')
    assert a.select(columns).equals(b.select(columns)),src
    assert (dst/'videos').is_symlink() and (dst/'videos').resolve()==(src/'videos').resolve()
    ds=LeRobotDataset(repo_id=dst.name,root=dst,video_backend='pyav')
    for i in (0,len(ds)-1):
        sample=ds[i]
        assert sample['action'].shape==(14,)
    checks.append({'dataset':str(dst),'frames':len(b),'content_preserved':True,'training_environment_loader':'passed'})
(out/'content_verification.json').write_text(json.dumps(checks,indent=2))
print('CONTENT_AND_TRAINING_ENV_VERIFIED',len(checks),'datasets',sum(x['frames'] for x in checks),'frames')

"""Read-only, CPU-only audit of the mounted competition LeRobot v3 clean data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import av

ROOT = Path('/RoboTwin/data/lerobot')
OUT = Path('/workspace/clean_screen_20261024')
JOINTS = list(range(6)) + list(range(7,13))  # exclude both grippers
COLS = ['observation.state','action','timestamp','frame_index','episode_index','index','task_index']
OUT.mkdir(exist_ok=True)

def audit_task(p):
    info = json.loads((p/'meta/info.json').read_text())
    files = sorted(p.glob('data/**/*.parquet'))
    table = pa.concat_tables([pq.read_table(f,columns=COLS) for f in files])
    meta = pa.concat_tables([pq.read_table(f) for f in sorted(p.glob('meta/episodes/**/*.parquet'))]).to_pylist()
    df = table.to_pandas()
    fps = info['fps']
    cameras = [k for k,v in info['features'].items() if v['dtype']=='video']
    task = p.name.removesuffix('_joint_v30')
    result, containers = [], {}
    for ep, rows in df.groupby('episode_index',sort=True):
        r = {'task':task,'dataset':str(p),'episode_index':int(ep),'frames':len(rows),'fps':fps,'errors':[]}
        errors = r['errors']
        records = [x for x in meta if x['episode_index']==ep]
        if len(records)!=1:
            errors.append('episode_metadata_count'); result.append(r); continue
        m = records[0]
        r['instructions'] = m['tasks']
        if not m['tasks'] or any(not str(x).strip() for x in m['tasks']): errors.append('empty_instruction')
        if m['length']!=len(rows): errors.append('metadata_length')
        if not np.array_equal(rows.frame_index.to_numpy(),np.arange(len(rows))): errors.append('frame_index_gap_or_duplicate')
        if not np.array_equal(rows['index'].to_numpy(),np.arange(m['dataset_from_index'],m['dataset_to_index'])): errors.append('global_index_mismatch')
        ts=rows.timestamp.to_numpy().astype(float)
        if not np.isfinite(ts).all() or not np.allclose(ts,np.arange(len(rows))/fps,atol=1e-4): errors.append('timestamp_mismatch')
        x=np.stack(rows['observation.state']).astype(np.float64)
        a=np.stack(rows['action']).astype(np.float64)
        if x.shape!=(len(rows),14) or a.shape!=x.shape: errors.append('shape_mismatch')
        if not np.isfinite(x).all() or not np.isfinite(a).all(): errors.append('nonfinite_state_or_action')
        if len(rows)<4: errors.append('too_short_for_dynamics')
        r['state_action_sha256']=hashlib.sha256(x.tobytes()+a.tobytes()).hexdigest()
        if not errors:
            for name,arr in [('action',a),('state',x)]:
                q=arr[:,JOINTS]
                for order,label in [(1,'velocity'),(2,'acceleration'),(3,'jerk')]:
                    d=np.diff(q,n=order,axis=0)*fps**order
                    r[name+'_'+label+'_rms']=float(np.sqrt(np.mean(d*d)))
                    r[name+'_'+label+'_peak']=float(np.max(np.abs(d)))
            r['duration_s']=len(rows)/fps
            r['state_motion_rms']=float(np.sqrt(np.mean(np.diff(x,axis=0)**2)))
            if r['state_motion_rms']==0: errors.append('entire_episode_static')
        r['sampled_video_frames']=0
        for cam in cameras:
            prefix='videos/'+cam+'/'
            v=p/info['video_path'].format(video_key=cam,chunk_index=m[prefix+'chunk_index'],file_index=m[prefix+'file_index'])
            start=m[prefix+'from_timestamp']; end=m[prefix+'to_timestamp']
            if not np.isclose(end-start,len(rows)/fps,atol=1/fps): errors.append('video_span:'+cam)
            try:
                if v not in containers: containers[v]=av.open(str(v))
                c=containers[v]; stream=c.streams.video[0]
                for offset in (0,len(rows)//2,len(rows)-1):
                    target=start+offset/fps
                    c.seek(int(target/float(stream.time_base)),stream=stream,backward=True,any_frame=False)
                    found=False
                    for frame in c.decode(stream):
                        if frame.time is not None and frame.time>=target-0.25/fps:
                            if abs(frame.time-target)>1.1/fps: raise ValueError('frame_timestamp')
                            rgb=frame.to_ndarray(format='rgb24')
                            if rgb.shape!=(info['features'][cam]['shape'][1],info['features'][cam]['shape'][2],3): raise ValueError('frame_shape')
                            found=True; r['sampled_video_frames']+=1; break
                    if not found: raise ValueError('missing_frame')
            except Exception as exc:
                errors.append('video_read:'+cam+':'+type(exc).__name__+':'+str(exc)[:100])
        result.append(r)
    for c in containers.values(): c.close()
    summary={'task':task,'episodes':len(result),'frames':len(df),'expected_episodes':info['total_episodes'],
             'expected_frames':info['total_frames'],'fps':fps,'cameras':cameras,
             'source_sha256':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    assert len(result)==info['total_episodes']==50, summary
    assert len(df)==info['total_frames'], summary
    assert set(df.episode_index.unique())==set(range(50)), summary
    (OUT/(task+'.json')).write_text(json.dumps({'summary':summary,'episodes':result},indent=2,allow_nan=False))
    print(task,len(result),sum(bool(r['errors']) for r in result),'errors',flush=True)
    return summary

if __name__=='__main__':
    start=time.time()
    manifest=Path('/RoboTwin/data/robotwin_demo_clean_joint_v30.txt').read_text()
    original_paths=[Path(line.split()[1]) for line in manifest.splitlines() if line.strip()]
    paths=[ROOT/p.name for p in original_paths]
    assert len(paths)==len(set(paths))==50
    assert all(p.parent==ROOT for p in paths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        summaries=list(pool.map(audit_task,paths))
    summary={'tasks':summaries,'elapsed_seconds':time.time()-start,'source_manifest':manifest,'manifest_paths_exist':{str(p):p.exists() for p in original_paths},
             'video_check':'start/middle/end frame per episode and camera; not full video decoding',
             'data_revision_from_image_guide':'a967b852afa21a9cbf19a198f7e653109042e87c'}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2))
    print('AUDIT_COMPLETE',len(summaries),'tasks',round(time.time()-start,1),'seconds',flush=True)

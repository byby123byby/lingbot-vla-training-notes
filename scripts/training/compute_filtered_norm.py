"""Compute LingBot RoboTwin statistics on CPU, without decoding video.

Matches official pre-padding feature mapping and action chunk/end padding.
Uses a fixed global 5000-bin histogram (no order-dependent histogram rebinning).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ARM = [0,1,2,3,4,5,7,8,9,10,11,12]
GRIPPER = [6,13]


def chunk_weights(length, chunk_size=50):
    if length < 1 or chunk_size < 1:
        raise ValueError("Empty episode or invalid chunk size")
    weights = np.minimum(np.arange(length)+1,chunk_size).astype(np.int64)
    weights[-1] += length*chunk_size-int(weights.sum())
    return weights


def stats(values, weights):
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or np.any(weights<=0):
        raise ValueError("Non-finite data or invalid weights")
    mean = np.average(values,axis=0,weights=weights)
    variance = np.average((values-mean)**2,axis=0,weights=weights)
    minimum, maximum = values.min(0),values.max(0)
    result = {"mean":mean.tolist(),"std":np.sqrt(variance).tolist(),
              "min":minimum.tolist(),"max":maximum.tolist()}
    quantiles = {k:[] for k in ("q01","q99","q02","q98")}
    for dim in range(values.shape[1]):
        edges = np.linspace(minimum[dim]-1e-10,maximum[dim]+1e-10,5001)
        hist,_ = np.histogram(values[:,dim],bins=edges,weights=weights)
        cumulative = hist.cumsum()
        for key,q in zip(quantiles,(.01,.99,.02,.98)):
            quantiles[key].append(float(edges[np.searchsorted(cumulative,q*weights.sum())]))
    return {**result,**quantiles}


def compute(root, destination):
    selection = json.loads((root/'selection.json').read_text())
    expected = json.loads((root/'selection_summary.json').read_text())
    file_hashes = json.loads((root/'file_sha256.json').read_text())
    mappings = {Path(k).name:v for k,v in json.loads((root/'episode_id_mapping.json').read_text()).items()}
    states,actions,weights,records = [],[],[],[]
    for source,ids in sorted(selection.items()):
        task = Path(source).name
        directory = root/'filtered_lerobot'/task
        paths = sorted(directory.glob('data/**/*.parquet'))
        if not paths:
            raise ValueError('No data: '+task)
        if [mappings[task][str(i)] for i in range(len(ids))]!=ids:
            raise ValueError('Original episode identities differ: '+task)
        for path in paths:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=file_hashes[str(path.relative_to(root))]:
                raise ValueError('Archived data hash differs: '+str(path))
        import pyarrow as pa
        table = pa.concat_tables([pq.read_table(p) for p in paths])
        state = np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
        action = np.asarray(table['action'].to_pylist(),dtype=np.float32)
        ep = np.asarray(table['episode_index']); frame = np.asarray(table['frame_index'])
        if state.shape != action.shape or state.shape[1] != 14:
            raise ValueError('Unexpected RoboTwin feature schema')
        episodes = np.unique(ep)
        if len(episodes)!=len(ids) or not np.array_equal(episodes,np.arange(len(ids))):
            raise ValueError('Episode selection/count mismatch: '+task)
        task_weights = np.zeros(len(ep),dtype=np.int64)
        for eid in episodes:
            ix = np.flatnonzero(ep==eid)
            if not np.array_equal(ix,np.arange(ix[0],ix[-1]+1)) or not np.array_equal(frame[ix],np.arange(len(ix))):
                raise ValueError('Non-contiguous episode: '+task)
            task_weights[ix] = chunk_weights(len(ix))
        info = json.loads((directory/'meta/info.json').read_text())
        if info['total_frames']!=len(ep) or info['total_episodes']!=len(ids):
            raise ValueError('Metadata differs from parquet: '+task)
        states.append(state); actions.append(action); weights.append(task_weights)
        records.append({'task':task,'episodes':len(ids),'frames':len(ep),
                        'parquet_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    state,action,weight = np.concatenate(states),np.concatenate(actions),np.concatenate(weights)
    count = len(state)
    if count!=expected['kept_frames'] or sum(r['episodes'] for r in records)!=expected['kept'] or len(records)!=expected['tasks']:
        raise ValueError('Final selection summary mismatch')
    norm = {}
    for prefix,array,w in [('observation.state',state,np.ones(count,dtype=np.int64)),('action',action,weight)]:
        for group,indices in [('arm.position',ARM),('effector.position',GRIPPER)]:
            norm[prefix+'.'+group] = stats(array[:,indices],w)
    destination.mkdir(parents=True,exist_ok=True)
    payload = {'norm_stats':norm,'count':count}
    (destination/'clean_2455.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    report = {'tasks':len(records),'episodes':sum(r['episodes'] for r in records),'frames':count,
              'action_chunk_size':50,'action_weighted_count':int(weight.sum()),
              'method':'official feature mapping; repeat-last action chunk; global fixed 5000-bin weighted histogram',
              'upstream_equivalence':'same population and mapping, histogram quantiles may differ from upstream adaptive rebinning',
              'norm_sha256':hashlib.sha256((destination/'clean_2455.json').read_bytes()).hexdigest(),
              'selection_sha256':hashlib.sha256((root/'selection.json').read_bytes()).hexdigest(),'datasets':records}
    (destination/'norm_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return {k:report[k] for k in ('tasks','episodes','frames','action_weighted_count','norm_sha256')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(compute(a.data_root,a.output),indent=2))

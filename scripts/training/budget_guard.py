"""Cooperative optimizer-boundary stop and durable DCP completion marker."""
import hashlib
import json
import os
import time
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def checkpoint_files(path, world_size):
    path=Path(path)
    required=[path/'model/.metadata',path/'optimizer/.metadata']
    required += [path/f'extra_state/extra_state_rank_{rank}.pt' for rank in range(world_size)]
    for folder in ('model','optimizer'):
        shards=list((path/folder).glob('*.distcp'))
        if not shards:
            raise ValueError('DCP has no '+folder+' shards')
    files=[p for p in path.rglob('*') if p.is_file() and p.name not in ('COMPLETE.json','COMPLETE.json.tmp')]
    if any(not p.is_file() or p.stat().st_size==0 for p in required) or any(p.suffix=='.tmp' for p in files):
        raise ValueError('Incomplete DCP structure')
    return files


def seal_checkpoint(path, world_size, step, norm_path):
    path=Path(path)
    files=checkpoint_files(path,world_size)
    entries={}
    for p in files:
        # Upstream writes DCP with sync_files=False; flush before reporting saved.
        with p.open('rb') as f:os.fsync(f.fileno())
        entries[str(p.relative_to(path))]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    atomic_json(path/'COMPLETE.json',{'schema_version':1,'project':'lingbot-vla-training-notes',
                'step':step,'world_size':world_size,'norm_sha256':sha256(norm_path),'files':entries})


def verify_checkpoint(path, world_size, norm_path):
    path=Path(path);marker=json.loads((path/'COMPLETE.json').read_text())
    if marker['project']!='lingbot-vla-training-notes' or marker['world_size']!=world_size or marker['norm_sha256']!=sha256(norm_path):
        raise ValueError('Checkpoint identity, world size or normalization differs')
    files=checkpoint_files(path,world_size)
    if set(marker['files'])!={str(p.relative_to(path)) for p in files}:
        raise ValueError('Checkpoint inventory differs')
    for name,entry in marker['files'].items():
        p=path/name
        if p.stat().st_size!=entry['bytes'] or sha256(p)!=entry['sha256']:
            raise ValueError('Checkpoint checksum differs: '+name)
    return marker


class BudgetGuard:
    def __init__(self):
        import torch
        import torch.distributed as dist
        self.torch,self.dist=torch,dist
        self.deadline=float(os.environ['LINGBOT_SAVE_DEADLINE_EPOCH'])
        self.output=Path(os.environ['LINGBOT_RUN_DIR'])
        self.stop_file=self.output/'STOP_REQUESTED'
        self.norm_path=Path(os.environ['LINGBOT_NORM_PATH'])
        if self.deadline<=time.time():
            raise RuntimeError('Save deadline already passed; refuse training')

    def should_stop(self, step):
        flag=self.torch.tensor([0],device='cuda',dtype=self.torch.int32)
        if self.dist.get_rank()==0:
            flag[0]=int(time.time()>=self.deadline or self.stop_file.exists())
            atomic_json(self.output/'heartbeat.json',{'step':step,'time':time.time(),'stop_requested':bool(flag.item())})
        self.dist.broadcast(flag,src=0)
        return bool(flag.item())

    def saved(self, checkpoint, step):
        if self.dist.get_rank()==0:
            seal_checkpoint(checkpoint,self.dist.get_world_size(),step,self.norm_path)
            atomic_json(self.output/'SAVED.json',{'checkpoint':str(checkpoint),'step':step,'time':time.time()})
        self.dist.barrier()


"""Future cloud entry point. Default is offline plan-only, not training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml
from budget_guard import atomic_json, sha256, verify_checkpoint


def preflight(config, source, norm, run_dir, save_deadline, resumed=None):
    import torch
    if time.time()>=save_deadline-300:
        raise ValueError('Insufficient setup time before saving deadline')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=8:
        raise ValueError('Expected eight GPUs; CPU checks do not authorize training')
    if not Path('/workspace').is_mount():
        raise ValueError('/workspace must be a separately mounted persistent volume')
    if shutil.disk_usage('/workspace').free < 85*1024**3:
        raise ValueError('Need at least 85 GiB FREE on persistent /workspace for full DCP')
    if not run_dir.resolve().is_relative_to(Path('/workspace')):
        raise ValueError('Training output must be persistent')
    if run_dir.exists():
        raise FileExistsError('Use a new run directory; do not overwrite previous results')
    if 'BudgetGuard' not in (source/'tasks/vla/train_lingbotvla.py').read_text():
        raise ValueError('Install the verified runtime patches first')
    if 'from .strong_augment import' not in (source/'lingbotvla/data/vla_data/transform.py').read_text():
        raise ValueError('Strong augmentation not installed')
    if json.loads(norm.read_text())['count']!=540182:
        raise ValueError('Wrong normalization dataset')
    norm_report=Path(__file__).resolve().parents[2]/'configs/norm_stats/norm_report.json'
    if sha256(norm)!=json.loads(norm_report.read_text())['norm_sha256']:
        raise ValueError('Normalization checksum differs from prepared statistics')
    t=config['train'];d=config['data']
    if t['micro_batch_size']*8*t['gradient_accumulation_steps']!=t['global_batch_size'] or t['global_batch_size']!=256:
        raise ValueError('Unexpected effective batch')
    if t['use_lora'] or t['train_expert_only'] or not d['use_future_image']:
        raise ValueError('Full-SFT/full teacher recipe changed')
    for path in (config['model']['model_path'],config['model']['tokenizer_path'],
                 t['align_params']['depth']['moge_path'],t['align_params']['depth']['morgbd_path'],
                 t['align_params']['video']['ckpt_path'],t['align_params']['video']['config_path']):
        if not Path(path).exists():raise FileNotFoundError(path)
    rows=Path(d['train_path']).read_text().splitlines()
    if len(rows)!=50:raise ValueError('Expected 50 dataset paths')
    for row in rows:
        robot,directory=row.split()
        if robot!='robotwin' or not (Path(directory)/'data').is_dir() or not (Path(directory)/'videos').exists():
            raise ValueError('Dataset or video mount missing: '+directory)
    if resumed:
        marker=verify_checkpoint(resumed,8,norm)
        if t['max_steps']<=marker['step']:
            raise ValueError('Resume max_steps must be the cumulative target above the saved step')
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--billing-start-epoch',type=float,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--pvc-confirmed',action='store_true')
    p.add_argument('--watchdog-armed',type=Path,help='Receipt copied from the independent controller')
    a=p.parse_args();config=yaml.safe_load(a.config.read_text())
    deadline=a.billing_start_epoch+135*60
    if not a.execute:
        print(json.dumps({'mode':'offline_plan_only','training_started':False,'cloud_calls':0,
                          'save_deadline_epoch':deadline,'release_request_epoch':a.billing_start_epoch+165*60,
                          'billing_ceiling_epoch':a.billing_start_epoch+180*60},indent=2));return
    if not a.pvc_confirmed or not a.watchdog_armed:
        raise ValueError('Persistent storage and external release watchdog must be confirmed before training')
    receipt=json.loads(a.watchdog_armed.read_text())
    if receipt.get('billing_start_epoch')!=a.billing_start_epoch or receipt.get('run_dir')!=str(a.run_dir) or not receipt.get('armed'):
        raise ValueError('Watchdog receipt differs from this run')
    if time.time()-receipt['armed_at']>300:
        raise ValueError('Watchdog receipt older than five minutes; re-arm before launching')
    norm=Path(config['data']['norm_stats_file'])
    if receipt.get('norm_sha256')!=sha256(norm):
        raise ValueError('Watchdog normalization identity differs')
    preflight(config,a.source,norm,a.run_dir,deadline,a.resume)
    a.run_dir.mkdir(parents=True)
    config['train']['output_dir']=str(a.run_dir)
    config['train']['enable_resume']=False
    config['train']['load_checkpoint_path']=str(a.resume) if a.resume else None
    runtime_config=a.run_dir/'config.yaml';runtime_config.write_text(yaml.safe_dump(config,sort_keys=False))
    shutil.copy2(norm,a.run_dir/'norm_stats.json')
    for filename in ('strong_augment.py','budget_guard.py','upstream-lock.json'):
        shutil.copy2(Path(__file__).parent/filename,a.run_dir/filename)
    atomic_json(a.run_dir/'run_manifest.json',{'project':'lingbot-vla-training-notes','billing_start_epoch':a.billing_start_epoch,
         'norm_sha256':sha256(norm),'config_sha256':sha256(runtime_config),'pvc_confirmed':True,'world_size':8,
         'training_source':str(a.source),'resume':str(a.resume) if a.resume else None})
    env=os.environ.copy();env.update({'LINGBOT_SAVE_DEADLINE_EPOCH':str(deadline),'LINGBOT_RUN_DIR':str(a.run_dir),
               'LINGBOT_NORM_PATH':str(norm),'HIP_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','WANDB_MODE':'disabled'})
    env.pop('ROCR_VISIBLE_DEVICES',None);env.pop('CUDA_VISIBLE_DEVICES',None)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
             '-m','tasks.vla.train_lingbotvla',str(runtime_config)]
    with (a.run_dir/'training.log').open('w') as log:
        result=subprocess.run(command,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(a.run_dir/'TRAINING_EXIT.json',{'returncode':result.returncode,'time':time.time()})
    raise SystemExit(result.returncode)


if __name__=='__main__':main()

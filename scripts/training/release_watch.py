"""External controller: observe/save receipts, delete through official RCC SDK.

Run on a separate always-on machine. Default is OFFLINE. No create endpoint.
The future live run requires an explicitly selected, browser-authorized account.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from amd_oneclick_sdk.errors import OneClickError

from budget_guard import atomic_json

CHINA='https://developer.amd.com.cn/radeon'


def validate_plan(plan):
    if plan['base_url']!=CHINA or plan['gpu_count']!=8 or plan['credits_ceiling']!=24:
        raise ValueError('Unexpected region, resource size or budget')
    if not plan['persistent_workspace_confirmed'] or not plan['exclusive_instance_lease']:
        raise ValueError('PVC and exclusive instance use are required')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',plan['ssh_alias']):raise ValueError('Use an SSH config alias')
    if not Path(plan['run_dir']).is_relative_to('/workspace') or '..' in Path(plan['run_dir']).parts:
        raise ValueError('Run must be inside persistent /workspace')
    if plan['billing_start_epoch']<=0:raise ValueError('Use actual platform billing start, not training start')


def absent(status):
    # Unknown/failed/terminating are NOT proof that billing has stopped.
    return status.get('status') in ('none','not_found','deleted') and not status.get('instance_id')


def same_instance(status, plan):
    expected=plan['expected_instance']
    return all(status.get(k)==expected.get(k) and expected.get(k) not in (None,'')
               for k in ('instance_id','ssh_host','ssh_port'))


def remote_json(plan, relative):
    command='cat -- '+shlex.quote(str(Path(plan['run_dir'])/relative))
    try:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',plan['ssh_alias'],command],
                         capture_output=True,text=True,timeout=15)
    except (OSError,subprocess.TimeoutExpired):
        return None
    if r.returncode!=0:return None
    try:return json.loads(r.stdout)
    except json.JSONDecodeError:return None


def watch(plan, client, output, read_remote=remote_json, now=time.time, sleep=time.sleep):
    validate_plan(plan)
    start=plan['billing_start_epoch'];release_at=start+165*60;ceiling=start+180*60
    if start>now():raise ValueError('Billing start cannot be in the future')
    current=client.instance_current()
    if not same_instance(current,plan):raise ValueError('Current instance differs; no deletion attempted')
    if now()>=release_at:raise ValueError('Too late to arm a new run; use the console to release')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'WATCHDOG_ARMED.json',{'armed':True,'armed_at':now(),'billing_start_epoch':start,
                'run_dir':plan['run_dir'],'norm_sha256':plan['norm_sha256']})
    deletion_requested=False;last_request=0;completed=False;saved=False
    while True:
        try:
            current=client.instance_current()
            if absent(current):
                atomic_json(output/'RELEASED.json',{'released':True,'observed_at':now(),'within_budget_window':now()<ceiling,
                            'training_completed':completed,'checkpoint_saved':saved})
                return
            if not deletion_requested and not same_instance(current,plan):
                raise ValueError('Instance identity changed; refusing to affect a different run')
            if deletion_requested and current.get('instance_id') not in (None,plan['expected_instance']['instance_id']):
                raise ValueError('A different instance appeared after deletion; controller stops')
            if not deletion_requested:
                # Do not depend on SSH at the hard release deadline.
                exit_report=read_remote(plan,'TRAINING_EXIT.json') if now()<release_at else None
                if exit_report:
                    completed=exit_report['returncode']==0
                    atomic_json(output/'TRAINING_EXIT.json',exit_report)
                    saved_report=read_remote(plan,'SAVED.json')
                    if saved_report:
                        cp=Path(saved_report['checkpoint'])
                        valid_path=cp.is_relative_to(Path(plan['run_dir'])/'checkpoints') and '..' not in cp.parts
                        marker=read_remote(plan,str(cp.relative_to(plan['run_dir'])/'COMPLETE.json')) if valid_path else None
                        if marker and marker.get('norm_sha256')==plan['norm_sha256'] and marker.get('world_size')==8:
                            atomic_json(output/'SAVED.json',saved_report);atomic_json(output/'CHECKPOINT_MANIFEST.json',marker);saved=True
                    # Success without a sealed checkpoint is not a successful run.
                    completed=completed and saved
                if exit_report or now()>=release_at:
                    client.delete_instance();deletion_requested=True;last_request=now()
                    atomic_json(output/'DELETE_REQUESTED.json',{'time':now(),'checkpoint_saved':saved})
            elif now()-last_request>=60 and current.get('status')!='terminating':
                # Recheck ID before idempotent retry. The account must remain exclusive.
                if same_instance(current,plan):client.delete_instance();last_request=now()
        except (OSError,TimeoutError,subprocess.TimeoutExpired,OneClickError) as e:
            atomic_json(output/'CONTROLLER_WARNING.json',{'time':now(),'error_type':type(e).__name__})
        if now()>=ceiling:
            atomic_json(output/'RELEASE_UNCONFIRMED.json',{'time':now(),'released':False,'manual_action_required':True})
            raise RuntimeError('Release not confirmed before billing boundary; open China console immediately')
        sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args();plan=json.loads(a.plan.read_text())
    if not a.execute:
        print(json.dumps({'mode':'offline_plan_only','cloud_calls':0,'instance_creation_supported':False,
                          'save_after_minutes':135,'release_after_minutes':165,'ceiling_minutes':180},indent=2))
    else:
        validate_plan(plan)
        from amd_oneclick_sdk.rcc_config import resolve_connection_info
        from amd_oneclick_sdk.client import OneClickClient
        connection=resolve_connection_info(account=plan['account'])
        if connection.base_url.rstrip('/')!=CHINA:raise ValueError('Named account is not China region')
        # Client retries are bounded; no browser cookies or passwords are extracted.
        client=OneClickClient(CHINA,connection.token,timeout=10,max_attempts=1)
        watch(plan,client,a.output)

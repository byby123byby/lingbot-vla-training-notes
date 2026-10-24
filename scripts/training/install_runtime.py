"""Install narrow patches onto the pinned upstream checkout; never launches it."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old)!=1:
        raise ValueError('Upstream anchor changed: '+old[:90])
    return text.replace(old,new,1)


def patched_training(text):
    text=replace_once(text,'import json\n','import json\nimport random\nfrom lingbotvla.utils.budget_guard import BudgetGuard\n')
    text=replace_once(text,'    reached_max_steps = False\n','    budget_guard = BudgetGuard()\n    reached_max_steps = False\n')
    # Verification is before the upstream resume loop; a malformed resume must not
    # silently fall back to a fresh base model and waste this training segment.
    text=replace_once(text,'    if candidates:\n        last_err = None',
        '    if args.train.enable_resume and not candidates:\n        raise RuntimeError("Requested resume has no checkpoint")\n'
        '    if candidates:\n        for candidate in candidates:\n'
        '            from lingbotvla.utils.budget_guard import verify_checkpoint\n'
        '            verify_checkpoint(candidate, dist.get_world_size(), os.environ["LINGBOT_NORM_PATH"])\n'
        '        last_err = None')
    text=replace_once(text,'            logger.info_rank0("Starting training from scratch. No valid checkpoint could be loaded.")',
        '            raise RuntimeError("Requested checkpoint could not be restored") from last_err')
    text=replace_once(text,'                torch.set_rng_state(state["extra_state"]["torch_rng_state"])',
        '                torch.set_rng_state(state["extra_state"]["torch_rng_state"])\n'
        '                random.setstate(state["extra_state"]["python_rng_state"])\n'
        '                np.random.set_state(state["extra_state"]["numpy_rng_state"])\n'
        '                torch.cuda.set_rng_state(state["extra_state"]["cuda_rng_state"])')
    anchor='"torch_rng_state": torch.get_rng_state(),'
    if text.count(anchor)!=3:raise ValueError('Upstream checkpoint blocks changed')
    text=text.replace(anchor,anchor+'\n                        "python_rng_state": random.getstate(),\n'
                          '                        "numpy_rng_state": np.random.get_state(),\n'
                          '                        "cuda_rng_state": torch.cuda.get_rng_state(),')
    anchor='                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")'
    # Two inner blocks, plus one epoch-level block with a different indentation.
    text=text.replace(anchor,anchor+'\n                budget_guard.saved(save_checkpoint_path, global_step)')
    anchor='            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")'
    # Match whole lines, avoiding the two already instrumented inner blocks.
    lines=text.splitlines()
    result=[]
    for line in lines:
        result.append(line)
        if line==anchor:result.append('            budget_guard.saved(save_checkpoint_path, global_step)')
    text='\n'.join(result)+'\n'
    text=replace_once(text,'            if args.train.max_steps is not None and global_step >= args.train.max_steps:',
        '            budget_due = budget_guard.should_stop(global_step)\n'
        '            if budget_due or (args.train.max_steps is not None and global_step >= args.train.max_steps):')
    return text


def install(source):
    lock=json.loads((HERE/'upstream-lock.json').read_text())
    changes={}
    for rel,expected in lock['files'].items():
        p=source/rel
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Source hash differs; no changes installed: '+rel)
    transform=source/'lingbotvla/data/vla_data/transform.py'
    text=transform.read_text();start=text.index('def sample_visual_augmentation_params(');end=text.index('def prepare_images(',start)
    text=text[:start]+'from .strong_augment import sample_visual_augmentation_params, apply_visual_augmentation\n\n\n'+text[end:]
    changes[transform]=text
    train=source/'tasks/vla/train_lingbotvla.py'
    changes[train]=patched_training(train.read_text())
    for text in changes.values():ast.parse(text)
    # Pre-validate all sources before any write; preserve upstream backups.
    for path,text in changes.items():
        backup=path.with_suffix(path.suffix+'.before-training-prep')
        if backup.exists():raise FileExistsError(backup)
    for path,text in changes.items():
        shutil.copy2(path,path.with_suffix(path.suffix+'.before-training-prep'));path.write_text(text)
    shutil.copy2(HERE/'strong_augment.py',source/'lingbotvla/data/vla_data/strong_augment.py')
    shutil.copy2(HERE/'budget_guard.py',source/'lingbotvla/utils/budget_guard.py')
    return {'status':'installed_without_training','upstream_commit':lock['commit'],'patched_files':[str(p.relative_to(source)) for p in changes]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    print(json.dumps(install(p.parse_args().source),indent=2))

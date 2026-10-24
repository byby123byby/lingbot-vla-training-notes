import ast,copy,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from typing import Dict
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--normalize-source',type=Path,required=True);args=parser.parse_args()
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'scripts/training'))
from strong_augment import sample_visual_augmentation_params,apply_visual_augmentation
p=args.source/'lingbotvla/data/vla_data/transform.py'
source=p.read_text();tree=ast.parse(source)
keep={'dict_apply','Normalizer','prepare_images'}
module=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in keep],type_ignores=[])
ns={'Dict':Dict,'Tensor':torch.Tensor,'torch':torch,'np':np,'sample_visual_augmentation_params':sample_visual_augmentation_params,'apply_visual_augmentation':apply_visual_augmentation,'IMAGE_KEYS':['camera_top','camera_wrist_left','camera_wrist_right']}
exec(compile(module,str(p),'exec'),ns)
image=torch.randint(0,256,(3,64,64),dtype=torch.uint8);state=torch.zeros(55)
obs={'state':state,'image':{k:image.clone() for k in ns['IMAGE_KEYS']}}
output=ns['prepare_images'](None,obs,train=True,use_depth_align=True,return_augment_params=True)
future=ns['prepare_images'](None,obs,train=True,use_depth_align=True,augment_params=output[-1])
assert torch.equal(output[0],future[0]);assert torch.equal(output[2],future[2])
assert torch.equal(obs['image']['camera_top'],image);assert torch.equal(state,torch.zeros(55))
np.testing.assert_allclose(output[0].numpy(),output[2].numpy()/127.5-1,atol=1e-7)
payload=json.loads((r/'configs/norm_stats/clean_2455.json').read_text());norm=ns['Normalizer'](copy.deepcopy(payload['norm_stats']),{k:'bounds_99_woclip' for k in payload['norm_stats']})
raw={k:torch.tensor(np.asarray(v['mean'])[None,:]).repeat(50,1) for k,v in payload['norm_stats'].items()}
back=norm.unnormalize(norm.normalize(raw))
for k in raw:np.testing.assert_allclose(back[k],raw[k],atol=1e-6)
spec=importlib.util.spec_from_file_location('official_normalize',args.normalize_source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.deserialize_json((r/'configs/norm_stats/clean_2455.json').read_text())
from compute_filtered_norm import stats,chunk_weights
v=np.random.default_rng(4).normal(size=(73,14));chunks=v[np.minimum(np.arange(73)[:,None]+np.arange(50),72)].reshape(-1,14)
o=mod.RunningStats();o.update(chunks);official=o.get_statistics();ours=stats(v,chunk_weights(73))
for key in ours:np.testing.assert_allclose(ours[key],getattr(official,key),atol=1e-10,rtol=1e-10)
train=(args.source/'tasks/vla/train_lingbotvla.py').read_text();ast.parse(train)
assert train.count('budget_guard.saved(save_checkpoint_path, global_step)')==3
assert train.count('"python_rng_state": random.getstate()')==3
report={'patched_upstream_parse':True,'all_three_save_paths_sealed':True,'current_future_and_teacher_alignment':True,'inputs_unchanged':True,'official_norm_schema_loads':True,'official_normalizer_roundtrip':True,'weighted_histogram_matches_official_single_batch_toy':True,'gpu_training_performed':False,'cloud_calls':0}
(r/'evidence/training_integration_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

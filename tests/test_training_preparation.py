"""CPU-only checks. No model weights, training steps or cloud requests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/training'))
import strong_augment as aug
from compute_filtered_norm import chunk_weights,stats,ARM,GRIPPER
from budget_guard import seal_checkpoint,verify_checkpoint
from release_watch import watch,absent,CHINA,remote_json


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.image=torch.randint(0,256,(3,64,80),dtype=torch.uint8)
        self.params=aug.sample_visual_augmentation_params(self.image)
        self.params['active']=True

    def test_original_branch_is_exact(self):
        self.params['active']=False
        self.assertTrue(torch.equal(aug.apply_visual_augmentation(self.image,self.params),self.image))

    def test_replay_frames_views_dtype_and_input_unchanged(self):
        before=self.image.clone();params=copy.deepcopy(self.params)
        a=aug.apply_visual_augmentation(self.image,params)
        b=aug.apply_visual_augmentation(self.image.clone(),params)
        clip=aug.apply_visual_augmentation(torch.stack([self.image,self.image]),params)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(clip[0],clip[1]))
        self.assertTrue(torch.equal(a,clip[0]));self.assertTrue(torch.equal(self.image,before))
        self.assertEqual(params,self.params);self.assertEqual(a.dtype,torch.uint8)
        self.assertEqual(a.shape,self.image.shape);self.assertFalse(torch.equal(a,before))

    def test_float_input_stays_bounded(self):
        out=aug.apply_visual_augmentation(self.image.float()/255,self.params)
        self.assertEqual(out.dtype,torch.float32);self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(bool((out>=0).all() and (out<=1).all()))
        with self.assertRaises(ValueError):aug.apply_visual_augmentation(self.image.float(),self.params)

    def test_occlusion_cannot_overwrite_protected_center(self):
        p=copy.deepcopy(self.params)
        for key in ('color','geometry','noise','salt_pepper','blur','motion','jpeg'):p[key]=False
        p['occlusion']=True;p['rectangles']=[[.4,.4,.1,.1,2]]
        image=np.full((64,80,3),.5,np.float32)
        np.testing.assert_array_equal(image,aug.apply_frame(image,p))

    def test_each_transform_handles_small_images(self):
        for name in ('color','geometry','occlusion','noise','salt_pepper','blur','motion','jpeg'):
            p=copy.deepcopy(self.params)
            for key in ('color','geometry','occlusion','noise','salt_pepper','blur','motion','jpeg'):p[key]=key==name
            output=aug.apply_visual_augmentation(self.image,p)
            self.assertEqual(output.shape,self.image.shape)

    def test_probability_is_seeded_and_original_fraction(self):
        torch.manual_seed(8)
        original=sum(not aug.sample_visual_augmentation_params(self.image)['active'] for _ in range(1000))
        self.assertGreater(original,65);self.assertLess(original,140)


class StatisticsTests(unittest.TestCase):
    def test_weighted_statistics_match_explicit_padded_chunks(self):
        for length in (1,2,7,49,50,73):
            values=np.random.default_rng(length).normal(size=(length,14))
            chunks=values[np.minimum(np.arange(length)[:,None]+np.arange(50),length-1)].reshape(-1,14)
            weight=chunk_weights(length)
            self.assertEqual(weight.sum(),length*50)
            got=stats(values,weight);expected=stats(chunks,np.ones(len(chunks),dtype=int))
            for key in got:np.testing.assert_allclose(got[key],expected[key],rtol=1e-10,atol=1e-10)
        self.assertEqual(ARM+GRIPPER,[0,1,2,3,4,5,7,8,9,10,11,12,6,13])

    def test_final_stats_have_native_keys_and_dimensions(self):
        payload=json.loads((ROOT/'configs/norm_stats/clean_2455.json').read_text())
        self.assertEqual(payload['count'],540182)
        for key,value in payload['norm_stats'].items():
            size=12 if '.arm.' in key else 2
            for array in value.values():self.assertEqual(len(array),size);self.assertTrue(np.isfinite(array).all())
            self.assertTrue(np.all(np.array(value['q01'])<=value['q99']))


class CheckpointTests(unittest.TestCase):
    def make_checkpoint(self,directory):
        p=Path(directory)/'global_step_2';norm=Path(directory)/'norm.json';norm.write_text('{}')
        for name in ('model/.metadata','model/__0_0.distcp','optimizer/.metadata','optimizer/__0_0.distcp','extra_state/extra_state_rank_0.pt'):
            f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'local-test-only')
        return p,norm

    def test_partial_checkpoint_rejected_and_corruption_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p,norm=self.make_checkpoint(d)
            with self.assertRaises(FileNotFoundError):verify_checkpoint(p,1,norm)
            seal_checkpoint(p,1,2,norm);self.assertEqual(verify_checkpoint(p,1,norm)['step'],2)
            (p/'model/__0_0.distcp').write_bytes(b'damaged')
            with self.assertRaises(ValueError):verify_checkpoint(p,1,norm)

    def test_wrong_rank_or_stats_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,norm=self.make_checkpoint(d);seal_checkpoint(p,1,2,norm)
            with self.assertRaises(ValueError):verify_checkpoint(p,8,norm)
            norm.write_text('{"changed":true}')
            with self.assertRaises(ValueError):verify_checkpoint(p,1,norm)


class ReleaseTests(unittest.TestCase):
    def plan(self):
        return {'base_url':CHINA,'gpu_count':8,'credits_ceiling':24,'persistent_workspace_confirmed':True,
                'exclusive_instance_lease':True,'ssh_alias':'test-only','run_dir':'/workspace/own-run',
                'billing_start_epoch':1000,'expected_instance':{'instance_id':'fake','ssh_host':'localhost','ssh_port':1234},
                'norm_sha256':'test-hash'}

    def client(self,plan):
        class Fake:
            deleted=0
            def instance_current(self):
                return {'status':'none'} if self.deleted else {'status':'ready',**plan['expected_instance']}
            def delete_instance(self):self.deleted+=1;return {'success':True}
        return Fake()

    def test_finished_run_seals_receipt_and_releases(self):
        p=self.plan();c=self.client(p);clock=[1050.]
        reports={'TRAINING_EXIT.json':{'returncode':0},'SAVED.json':{'checkpoint':'/workspace/own-run/checkpoints/global_step_2'},
                 'checkpoints/global_step_2/COMPLETE.json':{'norm_sha256':'test-hash','world_size':8}}
        with tempfile.TemporaryDirectory() as d:
            watch(p,c,d,lambda p,r:reports.get(r),lambda:clock[0],lambda s:clock.__setitem__(0,clock[0]+s))
            result=json.loads((Path(d)/'RELEASED.json').read_text())
            self.assertTrue(result['training_completed']);self.assertEqual(c.deleted,1)

    def test_deadline_releases_with_ssh_unavailable(self):
        p=self.plan();c=self.client(p);clock=[p['billing_start_epoch']+165*60-5]
        with tempfile.TemporaryDirectory() as d:
            watch(p,c,d,lambda p,r:None,lambda:clock[0],lambda s:clock.__setitem__(0,clock[0]+s))
            result=json.loads((Path(d)/'RELEASED.json').read_text())
            self.assertFalse(result['checkpoint_saved']);self.assertTrue(result['within_budget_window'])

    def test_terminating_is_not_released(self):
        self.assertFalse(absent({'status':'terminating'}));self.assertFalse(absent({'status':'failed'}))
        self.assertFalse(absent({'status':'none','instance_id':'still-present'}))

    def test_wrong_instance_not_deleted(self):
        p=self.plan();c=self.client(p);p['expected_instance']['ssh_port']=9999
        c.instance_current=lambda:{'status':'ready','instance_id':'other'}
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):
            watch(p,c,d,lambda p,r:None,lambda:1050,lambda s:None)
        self.assertEqual(c.deleted,0)

    def test_delete_network_error_retries(self):
        from amd_oneclick_sdk.errors import NetworkError
        p=self.plan();c=self.client(p);clock=[p['billing_start_epoch']+165*60-5];attempts=[]
        original=c.delete_instance
        def flaky():
            attempts.append(1)
            if len(attempts)==1:raise NetworkError('simulated outage')
            return original()
        c.delete_instance=flaky
        with tempfile.TemporaryDirectory() as d:
            watch(p,c,d,lambda p,r:None,lambda:clock[0],lambda s:clock.__setitem__(0,clock[0]+s))
            self.assertTrue((Path(d)/'RELEASED.json').exists())
        self.assertEqual(len(attempts),2)

    def test_unknown_status_does_not_silently_claim_release(self):
        p=self.plan();c=self.client(p);clock=[p['billing_start_epoch']+165*60-5]
        c.instance_current=lambda:{'status':'terminating',**p['expected_instance']}
        with tempfile.TemporaryDirectory() as d,self.assertRaises(RuntimeError):
            watch(p,c,d,lambda p,r:None,lambda:clock[0],lambda s:clock.__setitem__(0,clock[0]+s))


if __name__=='__main__':unittest.main(verbosity=2)

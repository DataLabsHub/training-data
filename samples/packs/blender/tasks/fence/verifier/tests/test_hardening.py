"""Regression tests of admission limits, bundle isolation and runtime identity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import support
import runtime
import reward


class HardeningTests(unittest.TestCase):
    def test_sparse_oversized_file_is_rejected_before_runtime_or_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'input').mkdir()
            source = root/'input/result.blend'
            with source.open('wb') as stream:
                stream.truncate(support.POLICY['max_submission_bytes']+1)
            with patch('reward.digest', side_effect=AssertionError('must not hash')), patch('reward.shutil.which', side_effect=AssertionError('must not load runtime')):
                self.assertEqual(reward.main(['--submission',str(source),'--reports',str(root/'report')]),0)
            result = support.read_json(root/'report/result.json')
            self.assertEqual((result['reward'],result['training_mask']),(0,1))
    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'file').write_bytes(b'BLENDER-fixture')
            (root/'link').symlink_to(root/'file')
            self.assertEqual(support.file_preflight(root/'link'),'submission_not_regular_file')
    def test_bundle_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'input').mkdir()
            source=root/'input/result.blend';source.write_bytes(b'BLENDER-fixture')
            (root/'input/escape').symlink_to(root/'outside')
            with self.assertRaises(ValueError):support.freeze_submission(source,root/'snapshot')
    def test_copy_is_immutable_from_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'input').mkdir()
            source=root/'input/result.blend';source.write_bytes(b'BLENDER-fixture')
            copy=support.freeze_submission(source,root/'snapshot')
            source.write_bytes(b'changed')
            self.assertEqual(copy.read_bytes(),b'BLENDER-fixture')
    def test_copy_cannot_recurse_into_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'result.blend';source.write_bytes(b'BLENDER-fixture')
            with self.assertRaises(ValueError):support.freeze_submission(source,root/'snapshot')
            self.assertFalse((root/'snapshot').exists())
    def test_bundle_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'input').mkdir()
            source=root/'input/result.blend';source.write_bytes(b'BLENDER-fixture')
            with patch.dict(support.POLICY,{'max_bundle_bytes':1}):
                with self.assertRaises(support.BudgetExceeded):support.freeze_submission(source,root/'snapshot')
    def test_strict_json_rejects_nan_and_duplicate_keys(self):
        for data in ('{"x":NaN}','{"x":1,"x":2}'):
            with self.assertRaises(ValueError):support.strict_json(data)
    def test_utf8_chinese_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'中文.json'; support.dump_json(path,{'标准':'可编辑'})
            self.assertEqual(support.read_json(path),{'标准':'可编辑'})
    def test_runtime_uses_policy_resources(self):
        import re
        metadata=(Path(__file__).resolve().parents[2]/'task.toml').read_text(encoding='utf-8')
        for key in ('cpus','memory_mb'):
            declared=int(re.search(r'^'+key+r' = ([0-9]+)$',metadata,re.M)[1])
            self.assertEqual(declared,support.POLICY[key])
        cmd=runtime.sandbox('unused')
        self.assertEqual(cmd[cmd.index('--cpus')+1],str(support.POLICY['cpus']))
        self.assertEqual(cmd[cmd.index('--memory')+1],str(support.POLICY['memory_mb'])+'m')
    def test_mutable_image_requires_matching_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock=Path(tmp)/'lock.json'
            with patch.object(runtime,'LOCK',lock):
                with self.assertRaises(RuntimeError):runtime.resolve_image('tag')
                support.dump_json(lock,{'image_id':'sha256:'+'a'*64,'source_ref':'tag','blender_version':support.POLICY['blender_version']})
                self.assertEqual(runtime.resolve_image('tag'),'sha256:'+'a'*64)
                with self.assertRaises(RuntimeError):runtime.resolve_image('different')
    def test_orphan_geometry_budget(self):
        from types import SimpleNamespace as S
        fake=S(data=S(objects=[],meshes=[S(vertices=range(support.POLICY['max_vertices']+1),polygons=[])]))
        with self.assertRaises(support.BudgetExceeded):support.native_preflight(fake)
    def test_native_stdout_budget(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(support.POLICY,{'max_native_log_bytes':128}):
            with self.assertRaises(support.BudgetExceeded):
                support.run_native([sys.executable,'-c','print("x"*10000)'],Path(tmp)/'log',2,tmp)
    def test_native_timeout(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.TimeoutExpired):
                support.run_native([sys.executable,'-c','import time;time.sleep(60)'],Path(tmp)/'log',.05,tmp)
    def test_report_disk_budget(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(support.POLICY,{'max_report_bytes':128}):
            (Path(tmp)/'oversized').write_bytes(b'x'*129)
            with self.assertRaises(support.BudgetExceeded):
                support.run_native([sys.executable,'-c','import time;time.sleep(60)'],Path(tmp)/'log',2,tmp)

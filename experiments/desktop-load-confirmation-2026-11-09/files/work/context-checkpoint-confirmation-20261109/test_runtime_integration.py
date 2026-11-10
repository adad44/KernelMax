import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evidence import Controller,Store,GUARDS,publish,sha_file
from runtime_integration import RuntimeSession,UserPause,validate_activation

SPEC=Path('/Users/alandiaz/Downloads/KernelMaxxxing/work/resumable-confirmation-contract-20261104/protocol.json')


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.protocol=json.loads(SPEC.read_text());self.store=Store(self.tmp.name,self.protocol)
        self.controller=Controller(self.store);self.controller.freeze({'synthetic_cpu_test':True})
        self.controller.open_epoch('one',identities={'synthetic_cpu_test':True})
        self.addCleanup(self.cleanup)
        self.runtime=RuntimeSession.__new__(RuntimeSession)
        self.runtime.controller=self.controller;self.runtime.store=self.store
        self.runtime.work=Path(self.tmp.name);self.runtime.protocol=self.protocol;self.runtime.epoch='one'
        self.calls=0;self.runtime.request_counter=lambda:self.calls
        self.cs={'resident_primers':[],'requests':[],'phase_memory_requests':[]}

    def cleanup(self):
        if self.controller.lease is not None: self.controller.lease.close();self.controller.lease=None

    def test_no_activation_file_blocks_before_any_model_loading(self):
        with self.assertRaises(FileNotFoundError): validate_activation(self.tmp.name,self.protocol)
        with self.assertRaises(FileNotFoundError): RuntimeSession(self.tmp.name,'new')
        self.assertFalse((Path(self.tmp.name)/'epochs').exists())

    def test_preparation_only_authority_cannot_activate(self):
        (Path(self.tmp.name)/'timing-authorization.json').write_text(json.dumps({'authorized':False}))
        with self.assertRaises(RuntimeError): validate_activation(self.tmp.name,self.protocol)

    def test_native_collection_wraps_two_calls_without_interposed_evidence_io(self):
        def native(*args):
            writes=list(self.store.audit_writes)
            self.calls+=1;self.cs['resident_primers'].append({'environment_assessment':{'continue_pass':True}})
            self.assertEqual(writes,self.store.audit_writes)
            self.calls+=1;self.cs['requests'].append({'environment_assessment':{'confirmation_sample_eligible':True}})
            self.assertEqual(writes,self.store.audit_writes)
        self.runtime._collect(native,self.cs,2048,'baseline','comparison',1,1)
        state=self.controller.state();self.assertEqual(state['starts_reserved'],2)
        self.assertTrue(all(row['executed'] and row['eligible'] for row in next(iter(state['collections'].values()))['artifacts']))

    def test_pause_after_primer_does_not_execute_or_uncharge_measurement(self):
        def native(*args):
            self.calls+=1;self.cs['resident_primers'].append({'environment_assessment':{'continue_pass':True}})
            self.controller.request_pause()
            raise UserPause('drained primer')
        with self.assertRaises(UserPause): self.runtime._collect(native,self.cs,2048,'baseline','comparison',1,1)
        collection=next(iter(self.controller.state()['collections'].values()))
        self.assertEqual([row['executed'] for row in collection['artifacts']],[True,False])
        self.assertEqual(self.controller.state()['starts_reserved'],2)

    def test_failed_started_model_call_is_not_reported_as_unexecuted(self):
        def native(*args):
            self.calls+=1
            raise RuntimeError('synthetic GPU error before a raw result exists')
        with self.assertRaises(RuntimeError): self.runtime._collect(native,self.cs,2048,'baseline','comparison',1,1)
        collection=next(iter(self.controller.state()['collections'].values()))
        self.assertTrue(collection['artifacts'][0]['executed'])
        self.assertFalse(collection['artifacts'][0]['eligible'])
        self.assertFalse(collection['artifacts'][1]['executed'])

    def test_catalog_keeps_all_three_immutable_revisions(self):
        seals={}
        self.runtime.controller=SimpleNamespace(state=lambda:{'seals':seals})
        for context in (512,2048,4096):
            receipt=publish(Path(self.tmp.name)/f'fake-{context}.json',{'synthetic':True})
            seals[str(context)]={'receipt':receipt}
            value={'context':context,'repeatable_gain_verified':True,
                'five_percent_verified':False,'claim_scope':'synthetic CPU-only catalog test'}
            with patch('runtime_integration.verify_context',return_value=value): self.runtime._retain(value)
        revisions=sorted((Path(self.tmp.name)/'catalog-revisions').glob('*.json'))
        self.assertEqual(len(revisions),3)
        self.assertEqual([row['context'] for row in json.loads(revisions[-1].read_text())['checkpoints']],[512,2048,4096])


class GeneratedRunnerTests(unittest.TestCase):
    def test_generated_native_trace_and_correctness_are_unchanged(self):
        root=Path(__file__).resolve().parent
        report=json.loads((root/'native-generation.json').read_text())
        self.assertEqual(sha_file(root/'native_runner.py'),report['generated_sha256'])
        self.assertEqual(sha_file(report['source']),report['source_sha256'])
        self.assertTrue(all(report['unchanged_AST'].values()))
        old=ast.parse(Path(report['source']).read_text());new=ast.parse((root/'native_runner.py').read_text())
        def collect(tree):
            return copy.deepcopy(next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='collect'))
        a=collect(old);b=collect(new)
        class StripPause(ast.NodeTransformer):
            def visit_If(self,node):
                if ast.unparse(node.test)=='integration.pause_requested': return None
                return self.generic_visit(node)
        b=StripPause().visit(b)
        self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))


if __name__=='__main__': unittest.main()

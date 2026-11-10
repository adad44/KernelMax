import copy
import json
from pathlib import Path
import tempfile
import unittest
from certificates import verify_context
from evidence import EvidenceError,GUARDS,publish
from test_verification import SyntheticPass,mock_sample
from verification import ContextVerifier


class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=SyntheticPass(self.tmp.name);self.addCleanup(self.run.cleanup)
        self.run.control(2048);self.run.block(2048,1)
        guards={**{key:True for key in GUARDS},'protected_source_sha256':{},'numerical_source_sha256':{}}
        self.value=self.run.verifier.seal(self.run.controller,2048,guards)

    def factory(self,store):
        return ContextVerifier(store,sample_validator=mock_sample,
            numeric_validator=lambda proof:{int(k):v for k,v in proof['tokens'].items()})

    def verify(self):
        return verify_context(self.run.store,2048,verifier_factory=self.factory,
            identity_validator=lambda *_:None)  # Synthetic CPU fixture, never a real catalog.

    def test_seal_recomputes_against_its_own_immutable_prefix(self):
        self.assertEqual(self.verify(),self.value)

    def test_later_context_activity_does_not_rewrite_prior_scope(self):
        self.run.collect('aa_warmup',4096,'A',1)
        self.assertEqual(self.verify(),self.value)

    def test_changed_context_artifact_is_rejected(self):
        path=Path(self.tmp.name)/'contexts/2048.json'
        data=json.loads(path.read_text());data['five_percent_verified']=True
        path.write_text(json.dumps(data))
        with self.assertRaises(EvidenceError): self.verify()

    def test_default_identity_verifier_cannot_accept_synthetic_cpu_identity(self):
        with self.assertRaises((EvidenceError,KeyError)):
            verify_context(self.run.store,2048,verifier_factory=self.factory)


if __name__=='__main__': unittest.main()

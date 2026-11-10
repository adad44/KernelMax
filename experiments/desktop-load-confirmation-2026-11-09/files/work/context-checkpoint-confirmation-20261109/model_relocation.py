"""Verify byte-identical checkpoint relocation without requiring the old volume."""
import hashlib
import json
from pathlib import Path
from evaluator.checkpoint_verification import validate_receipt

ORIGINAL_RECEIPT_SHA = 'a40793bc665c8fbb108028d391abf2b4d3bf0b5836d712e6eedf910545d5e2df'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_relocation(proof_path, *, receipt_path=None, model_path=None):
    proof_path = Path(proof_path)
    proof = json.loads(proof_path.read_text())
    original_path = Path(proof['original_receipt_path'])
    internal_path = Path(proof['internal_receipt_path'])
    if (proof.get('schema') != 1 or proof.get('status') != 'exact_copy_verified'
            or proof.get('original_receipt_sha256') != ORIGINAL_RECEIPT_SHA
            or digest(original_path) != ORIGINAL_RECEIPT_SHA
            or digest(internal_path) != proof['internal_receipt_sha256']
            or not all(proof.get(key) is True for key in (
                'source_identities_before_and_after_equal', 'source_bytes_match_original_manifest',
                'destination_bytes_match_original_manifest', 'original_receipt_and_files_unmodified',
                'no_precision_or_layout_change'))):
        raise RuntimeError('Checkpoint relocation proof is invalid or changed')
    original = json.loads(original_path.read_text())
    internal = json.loads(internal_path.read_text())
    manifest = original['model_files']
    if (internal['model_files'] != manifest or proof['model_files'] != manifest
            or original['byte_verification'] != 'passed'
            or internal['model_revision'] != original['model_revision']
            or proof['model_revision'] != original['model_revision']
            or proof['source_model_path'] != original['model_path']
            or internal['model_path'] != proof['internal_model_path']
            or proof['required_file_count'] != len(manifest)
            or proof['total_bytes'] != sum(identity[2] for identity in original['file_identity'].values())
            or (receipt_path is not None and Path(receipt_path).resolve() != internal_path.resolve())
            or (model_path is not None and Path(model_path).resolve() != Path(internal['model_path']).resolve())):
        raise RuntimeError('Relocated checkpoint does not match original pinned bytes/revision')
    validate_receipt(internal, internal['model_path'], original['model_revision'], manifest)
    sources = {str(path): digest(path) for path in (proof_path, original_path, internal_path)}
    return {'original_receipt_path': str(original_path), 'original_receipt_sha256': ORIGINAL_RECEIPT_SHA,
        'internal_receipt_path': str(internal_path), 'internal_model_path': internal['model_path'],
        'model_revision': internal['model_revision'], 'model_files': manifest,
        'proof_sha256': digest(proof_path), 'sources': sources}

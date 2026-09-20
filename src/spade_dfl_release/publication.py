"""Verify publication bytes separately from immutable historical source hashes."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from .paths import release_root

# Replaced by the release assembler after the exact text adaptations are recorded.
ADAPTATIONS_SHA256 = '700975df33b5e2633835b00d79228477445608a2170add447676ecb27f137233'

def publication_records():
    payload=(release_root()/'configs/publication_adaptations.json').read_bytes()
    if hashlib.sha256(payload).hexdigest()!=ADAPTATIONS_SHA256:
        raise ValueError('publication adaptation manifest hash mismatch')
    document=json.loads(payload)
    if document.get('schema_version')!=1:raise ValueError('unknown publication adaptation schema')
    records={}
    for row in document['files']:
        name=row['packaged_path'];path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or name in records:
            raise ValueError('invalid or duplicate publication adaptation path')
        records[name]=row
    return records

def verify_publication_file(path, historical_sha256, historical_bytes=None, *, root=None):
    """Validate original identity and exact current bytes; return current SHA256."""
    path=Path(path).resolve();root=Path(root or release_root()).resolve()
    relative=path.relative_to(root).as_posix()
    row=publication_records().get(relative)
    expected=historical_sha256;size=historical_bytes
    if row is not None:
        if row['historical_sha256']!=historical_sha256 or (historical_bytes is not None and row['historical_bytes']!=historical_bytes):
            raise ValueError('historical source identity differs from publication record')
        expected=row['published_sha256'];size=row['published_bytes']
    data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
    if (size is not None and len(data)!=size) or actual!=expected:
        raise ValueError('publication source size/hash mismatch: '+relative)
    return actual

def historical_module_hashes(package):
    """Return historical identities only after verifying each current module.

    This is a compatibility view for existing historical module-map contracts,
    not a report of current file hashes. Current hashes stay in the publication
    adaptation record and the archive integrity manifest.
    """
    package=Path(package);root=release_root().resolve();records=publication_records();result={}
    for path in package.rglob('*.py'):
        relative=path.resolve().relative_to(root).as_posix();row=records.get(relative)
        historical=row['historical_sha256'] if row else hashlib.sha256(path.read_bytes()).hexdigest()
        verify_publication_file(path,historical)
        result[path.relative_to(package).as_posix()]=historical
    return result

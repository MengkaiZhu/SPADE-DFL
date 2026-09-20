"""Current file integrity must not overwrite or bypass historical identity."""
import hashlib
import json
import pytest

@pytest.fixture
def adapted(tmp_path, monkeypatch):
    from spade_dfl_release import publication as p
    relative='src/spade_dfl_release/vendor/example/module.py'
    old=b'value = 1  # historical comment\n'
    new=b'value = 1  # published comment\n'
    path=tmp_path/relative;path.parent.mkdir(parents=True);path.write_bytes(new)
    row=dict(packaged_path=relative,historical_sha256=hashlib.sha256(old).hexdigest(),historical_bytes=len(old),published_sha256=hashlib.sha256(new).hexdigest(),published_bytes=len(new),change_types=['comment'])
    record=tmp_path/'configs/publication_adaptations.json';record.parent.mkdir()
    record.write_text(json.dumps({'schema_version':1,'files':[row]}),encoding='utf-8')
    monkeypatch.setattr(p,'release_root',lambda:tmp_path)
    monkeypatch.setattr(p,'ADAPTATIONS_SHA256',hashlib.sha256(record.read_bytes()).hexdigest())
    return p,path,row,record

def test_valid_adaptation_verifies_current_bytes_and_keeps_original_identity(adapted):
    p,path,row,_=adapted
    assert p.verify_publication_file(path,row['historical_sha256'],row['historical_bytes'])==row['published_sha256']
    assert p.historical_module_hashes(path.parent)=={'module.py':row['historical_sha256']}
    assert row['historical_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()

def test_changed_published_bytes_are_rejected(adapted):
    p,path,row,_=adapted;path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError):p.verify_publication_file(path,row['historical_sha256'],row['historical_bytes'])
    with pytest.raises(ValueError):p.historical_module_hashes(path.parent)

def test_wrong_historical_claim_is_rejected(adapted):
    p,path,row,_=adapted
    with pytest.raises(ValueError):p.verify_publication_file(path,'0'*64,row['historical_bytes'])
    with pytest.raises(ValueError):p.verify_publication_file(path,row['historical_sha256'],row['historical_bytes']+1)

def test_self_consistent_adaptation_map_tampering_is_rejected(adapted):
    p,path,row,record=adapted
    path.write_bytes(b'value = 2\n')
    data=json.loads(record.read_text());data['files'][0]['published_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();data['files'][0]['published_bytes']=path.stat().st_size
    record.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='adaptation manifest'):p.verify_publication_file(path,row['historical_sha256'])

def test_unadapted_file_still_requires_exact_historical_bytes(adapted):
    p,path,_,_=adapted
    other=path.with_name('unchanged.py');other.write_bytes(b'answer=42\n')
    digest=hashlib.sha256(other.read_bytes()).hexdigest()
    assert p.verify_publication_file(other,digest,10)==digest
    with pytest.raises(ValueError):p.verify_publication_file(other,'0'*64,10)

def test_missing_file_fails_without_success(adapted):
    p,path,row,_=adapted;path.unlink()
    with pytest.raises(FileNotFoundError):p.verify_publication_file(path,row['historical_sha256'])

def test_duplicate_and_unsafe_mapping_paths_are_rejected(adapted,monkeypatch):
    p,_,row,record=adapted
    for rows in ([row,row],[dict(row,packaged_path='../module.py')]):
        record.write_text(json.dumps({'schema_version':1,'files':rows}),encoding='utf-8')
        monkeypatch.setattr(p,'ADAPTATIONS_SHA256',hashlib.sha256(record.read_bytes()).hexdigest())
        with pytest.raises(ValueError):p.publication_records()

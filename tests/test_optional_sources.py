import json
from pathlib import Path

import pytest


def test_unconfigured_optional_source_index_reports_unavailable(monkeypatch):
    from spade_dfl_release.reference import resolve_source_index

    monkeypatch.delenv('SPADE_DFL_SOURCE_INDEX', raising=False)
    assert resolve_source_index() is None
    with pytest.raises(FileNotFoundError, match='SPADE_DFL_SOURCE_INDEX'):
        resolve_source_index(required=True)


def test_configured_missing_source_index_fails_instead_of_looking_elsewhere(monkeypatch, tmp_path):
    from spade_dfl_release.reference import resolve_source_index

    missing = tmp_path / 'missing-source-index.json'
    monkeypatch.setenv('SPADE_DFL_SOURCE_INDEX', str(missing))
    with pytest.raises(FileNotFoundError, match='missing-source-index.json'):
        resolve_source_index()


def test_explicit_source_index_overrides_environment(monkeypatch, tmp_path):
    from spade_dfl_release.reference import resolve_source_index

    configured = tmp_path / 'configured.json'
    configured.write_text('{"sources": []}', encoding='utf-8')
    explicit = tmp_path / 'explicit.json'
    explicit.write_text('{"sources": []}', encoding='utf-8')
    monkeypatch.setenv('SPADE_DFL_SOURCE_INDEX', str(configured))
    assert resolve_source_index(explicit) == explicit.resolve()


def test_relative_source_roots_are_resolved_from_index_parent(monkeypatch, tmp_path):
    from spade_dfl_release.reference import Sources

    index_parent = tmp_path / 'author-sources'
    source_root = index_parent / 'archives'
    source_root.mkdir(parents=True)
    index = index_parent / 'source-index.json'
    index.write_text(json.dumps({
        'schema_version': 1,
        'sources': [{'id': 'server_archive', 'root': 'archives/SPADE-DFL0828.zip', 'required': True}],
    }), encoding='utf-8')
    manifest = tmp_path / 'source-manifest.json'
    manifest.write_text('{"files": []}', encoding='utf-8')
    unrelated_working_directory = tmp_path / 'elsewhere'
    unrelated_working_directory.mkdir()
    monkeypatch.chdir(unrelated_working_directory)

    sources = Sources(index, manifest)
    assert sources.paths['server_archive'] == (source_root / 'SPADE-DFL0828.zip').resolve()


def test_direct_reference_import_requires_a_configured_source(monkeypatch, tmp_path):
    from spade_dfl_release.reference import import_reference_results

    monkeypatch.delenv('SPADE_DFL_SOURCE_INDEX', raising=False)
    output = tmp_path / 'reference-results'
    with pytest.raises(FileNotFoundError, match='SPADE_DFL_SOURCE_INDEX'):
        import_reference_results(output_root=output)
    assert not output.exists()

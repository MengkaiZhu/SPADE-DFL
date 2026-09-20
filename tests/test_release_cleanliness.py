"""Portable release audit; historical exceptions must be exact and evidenced."""
import hashlib
import json
import re
from pathlib import Path

import pytest
from spade_dfl_release.publication import verify_publication_file

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", "venv", "env", ".git", ".superpowers"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".log", ".aux", ".out", ".toc", ".fls", ".fdb_latexmk", ".bbl", ".blg"}
TEXT_SUFFIXES = {".py", ".json", ".md", ".toml", ".txt", ".csv", ".tex", ".svg", ".ps1", ".sh"}
PRIVATE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]+[\w\u0080-\uffff]|(?<![\w:])/(?:home|Users|mnt)/[\w]")
UNC_PATH = re.compile(r'''(?:^|[\s"'])\\\\[\w.-]+\\[\w$.-]+''', re.M)
TUNING_ALLOWLIST = {
    "src/spade_dfl_release/vendor/main_alzheimer/alzheimer_spade_dfl/configs/alzheimer_v3_tuning.json": {
        "sha256": "09e5f354712ddd739ad4e2436f1bfa92ddc121375d1496e40c302c2d65f405c0",
        "manifest": "configs/main_comparison/manifest.json"
    },
    "src/spade_dfl_release/vendor/main_alzheimer/alzheimer_spade_dfl/src/spade_dfl/alzheimer_tuning_protocol.py": {
        "sha256": "f9c9b08b7c13eb1655aa6e257c43098521546c9508c8884beb479ec821d44e79",
        "manifest": "configs/main_comparison/manifest.json"
    },
    "src/spade_dfl_release/vendor/main_alzheimer/alzheimer_spade_dfl/src/spade_dfl/alzheimer_v3_tuning.py": {
        "sha256": "67767e53f35d6f7a23be2adcd7c8653bee91be6eb3dfc6da37354e0f1c50bdf1",
        "manifest": "configs/main_comparison/manifest.json"
    },
    "src/spade_dfl_release/vendor/sensitivity/table34_n30_eps8_working/vendor/alzheimer/src/spade_dfl/alzheimer_tuning.py": {
        "sha256": "8588994a354fba5311ee49735deda2a30b257427f06e271ef180280085da5e82",
        "manifest": "configs/sensitivity/manifest.json"
    },
    "src/spade_dfl_release/vendor/sensitivity/table34_n30_eps8_working/vendor/alzheimer/src/spade_dfl/alzheimer_tuning_cache.py": {
        "sha256": "72f59d2f44d8108facc23a31850e5e26382efca294121a15589f9fff95038e51",
        "manifest": "configs/sensitivity/manifest.json"
    },
    "src/spade_dfl_release/vendor/sensitivity/table34_n30_eps8_working/vendor/alzheimer/src/spade_dfl/alzheimer_tuning_protocol.py": {
        "sha256": "f9c9b08b7c13eb1655aa6e257c43098521546c9508c8884beb479ec821d44e79",
        "manifest": "configs/sensitivity/manifest.json"
    },
    "src/spade_dfl_release/vendor/sensitivity/table34_n30_eps8_working/vendor/alzheimer/src/spade_dfl/alzheimer_v2_tuning.py": {
        "sha256": "85a4cf3c4f4d51ffd1a8fd87da542cf4841e4af4882764334474177d7533bcb0",
        "manifest": "configs/sensitivity/manifest.json"
    },
    "src/spade_dfl_release/vendor/sensitivity/table34_n30_eps8_working/vendor/alzheimer/src/spade_dfl/alzheimer_v3_tuning.py": {
        "sha256": "67767e53f35d6f7a23be2adcd7c8653bee91be6eb3dfc6da37354e0f1c50bdf1",
        "manifest": "configs/sensitivity/manifest.json"
    },
    "src/spade_dfl_release/vendor/sensitivity/table34_n30_eps8_working/vendor/brain/src/spade_dfl/brain_tumor_tuning.py": {
        "sha256": "db0f2479961eec7506a60ddc63ffb98d77a189458cb4b6fa179ade6bfef2825e",
        "manifest": "configs/sensitivity/manifest.json"
    }
}


def audit_tree(root, allowlist=None):
    allowlist = TUNING_ALLOWLIST if allowlist is None else allowlist
    findings = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append((relative, "symlink"))
            continue
        if any(p.lower() in FORBIDDEN_DIRS or p.lower().endswith('.egg-info') for p in path.relative_to(root).parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.endswith(".synctex.gz"):
            findings.append((relative, "cache/environment/log/latex"))
        if any(token in p.lower() for p in path.relative_to(root).parts for token in ("tuning", "candidate")):
            record = allowlist.get(relative)
            valid = path.is_file() and record is not None
            if valid:
                try:
                    verify_publication_file(path, record['sha256'], root=root)
                except ValueError:
                    valid = False
            if not valid:
                findings.append((relative, "unapproved tuning/candidate"))
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            content = path.read_text(encoding="utf-8-sig")
            # Decode JSON escapes before checking UNC paths; TeX commands in a
            # JSON template are not network shares. No file/directory exemption.
            unc_content = content
            if path.suffix.lower() == '.json':
                def strings(value):
                    if isinstance(value, str):
                        yield value
                    elif isinstance(value, dict):
                        for key, item in value.items():
                            yield key
                            yield from strings(item)
                    elif isinstance(value, list):
                        for item in value:
                            yield from strings(item)
                unc_content = '\n'.join(strings(json.loads(content)))
            if PRIVATE_PATH.search(content) or UNC_PATH.search(unc_content):
                findings.append((relative, "private absolute path"))
            if "\ufffd" in content:
                findings.append((relative, "replacement character"))
    return findings


def test_actual_release_tree_is_clean():
    assert audit_tree(ROOT) == []


@pytest.mark.parametrize("relative", ["__pycache__/x.pyc", ".venv/pyvenv.cfg", "src/package.egg-info/PKG-INFO", "run.log", "paper.aux", "candidate/plot.svg", "tuning/search.py"])
def test_audit_rejects_unrelated_artifacts(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unrelated", encoding="utf-8")
    assert audit_tree(tmp_path, {})


@pytest.mark.parametrize("content", ["X" + ":/private/data.csv", "/" + "home/private/data.csv", "bad" + chr(0xFFFD)])
def test_audit_rejects_private_paths_and_corrupt_text(tmp_path, content):
    (tmp_path / "file.md").write_text(content, encoding="utf-8")
    assert audit_tree(tmp_path, {})


@pytest.mark.parametrize("relative", ["README.md", "IMPLEMENTATION_NOTE.md", "data/README.md", "data/checksums.json"])
def test_reviewer_documentation_exists(relative):
    assert (ROOT / relative).is_file(), relative


def test_data_policy_is_backed_by_manifest_inputs():
    assert (ROOT / "data/checksums.json").is_file()
    policy = json.loads((ROOT / "data/checksums.json").read_text(encoding="utf-8"))
    assert policy["files"]
    assert all(row["included_in_release"] is False for row in policy["files"])
    expected = {}
    main = json.loads((ROOT / 'configs/main_comparison/manifest.json').read_text(encoding='utf-8'))
    for dataset, config in main['datasets'].items():
        for row in config['required_inputs']:
            placement = (ROOT / config['vendor_root'] / row['path']).resolve().relative_to(ROOT.resolve()).as_posix()
            expected[placement] = (row['sha256'], 'configs/main_comparison/manifest.json', row['path'])
    for family in ('sensitivity', 'heterogeneity'):
        manifest_path = 'configs/' + family + '/manifest.json'
        manifest = json.loads((ROOT / manifest_path).read_text(encoding='utf-8'))
        for row in manifest['required_inputs']:
            expected['src/spade_dfl_release/vendor/' + family + '/' + row['path']] = (row['sha256'], manifest_path, row['path'])
    actual = {r['placement']: (r['sha256'], r['manifest'], r['manifest_input_path']) for r in policy['files']}
    assert len(actual) == len(policy['files']) == 23
    assert actual == expected
    for row in policy['files']:
        assert row['release_builder_exclusion_required'] is True
        path = ROOT / row['placement']
    # Historical workspace-presence metadata does not require release inclusion.
        assert isinstance(row['workspace_source_present'], bool)
        if path.is_file():
            if 'bytes' in row:
                assert path.stat().st_size == row['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        else:
            assert row['included_in_release'] is False
            assert row['release_builder_exclusion_required'] is True
    present = [r['placement'] for r in policy['files'] if r['workspace_source_present']]
    assert present == ['src/spade_dfl_release/vendor/heterogeneity/data/mnist.npz']
    brain_raw = next(r for r in policy['files'] if r['manifest_input_path'].startswith('../'))
    assert brain_raw['hash_authority'] == 'frozen_config::dataset_sha256'
    assert brain_raw['original_bytes_locally_verified'] is False


def test_nine_exact_historical_exceptions_have_frozen_manifest_references():
    assert len(TUNING_ALLOWLIST) == 9
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and any(t in p.relative_to(ROOT).as_posix().lower() for t in ('tuning', 'candidate'))}
    assert actual == set(TUNING_ALLOWLIST)
    for path, record in TUNING_ALLOWLIST.items():
        manifest = json.loads((ROOT / record['manifest']).read_text(encoding='utf-8'))
        rows = [r for r in manifest['files'] if r['packaged_path'] == path]
        assert len(rows) == 1
        assert rows[0]['sha256'] == record['sha256']
        assert verify_publication_file(ROOT / path, record['sha256'], rows[0]['bytes']) == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if 'sensitivity/' in path:
            assert rows[0]['hash_authority'] == 'sensitivity_runtime/PLAN.json::sources'
            assert rows[0]['source_id'] == 'sensitivity_lineage'


def test_allowlist_never_accepts_changed_bytes_or_a_new_path(tmp_path):
    name, entry = next(iter(TUNING_ALLOWLIST.items()))
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / name).read_bytes())
    assert audit_tree(tmp_path) == []
    path.write_bytes(path.read_bytes() + b' ')
    assert (name, 'unapproved tuning/candidate') in audit_tree(tmp_path)
    path.rename(path.with_name('other_tuning.json'))
    assert any(reason == 'unapproved tuning/candidate' for _, reason in audit_tree(tmp_path))


def test_path_detector_literals_are_not_private_paths():
    assert PRIVATE_PATH.search("assert 'D" + ":/' not in encoded") is None
    assert PRIVATE_PATH.search(r'[A-Za-z]:[\\/]|/home/|/Users/') is None


def test_public_documentation_contract_and_parseable_commands():
    import argparse
    import ast
    import shlex
    from spade_dfl_release.cli import build_parser

    content = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert re.findall(r'^## .+$', content, re.M) == [
        '## Install',
        '## Data preparation',
        '## Verify saved results',
        '## Recompute statistics and figures',
        '## Training entry points',
        '## Package guide',
    ]
    protocols = (ROOT / 'docs/EXPERIMENT_PROTOCOLS.md').read_text(encoding='utf-8')
    assert re.findall(r'^## .+$', protocols, re.M) == [
        '## Main comparison: Figures 3–6 and Table II',
        '## Sensitivity: Tables III–IV',
        '## Heterogeneity: Figures 7–9',
        '## Communication: Figure 10',
    ]
    required_links = {'docs/KNOWN_LIMITATIONS.md', 'docs/EXPERIMENT_PROTOCOLS.md',
                      'docs/REPRODUCIBILITY.md', 'IMPLEMENTATION_NOTE.md', 'NOTICE.md',
                      'data/README.md', 'data/README.md#sensitivity-inputs-and-recovered-producer',
                      'reference_results/main_seed_records/run_index.csv'}
    links = set(re.findall(r'\]\(([^)]+)\)', content))
    assert required_links <= links
    assert all((ROOT / link.split('#', 1)[0]).is_file() for link in required_links)
    for link in required_links:
        if '#' in link:
            destination, anchor = link.split('#', 1)
            headings = re.findall(r'^#+ (.+)$', (ROOT / destination).read_text(encoding='utf-8'), re.M)
            assert anchor in {re.sub(r'[^\w\s-]', '', heading.lower()).replace(' ', '-') for heading in headings}
    assert (ROOT / 'data/INPUT_PREPARATION.json').is_file()

    commands = [shlex.split(line) for line in content.splitlines() if line.startswith('python experiments/')]
    families = {'experiments/01_main_comparison.py': 'main',
                'experiments/02_sensitivity.py': 'sensitivity',
                'experiments/03_heterogeneity.py': 'heterogeneity',
                'experiments/04_communication.py': 'communication'}
    expected_entries = set(families) | {'experiments/05_recompute_main.py', 'experiments/make_paper_outputs.py'}
    assert {parts[1] for parts in commands} == expected_entries
    for parts in commands:
        entry, arguments = parts[1], parts[2:]
        assert (ROOT / entry).is_file()
        if entry in families:
            # Parsing is safe: never call main(), which dispatches experiments.
            parsed = build_parser(fixed_family=families[entry]).parse_args(arguments)
            assert not parsed.rerun or parsed.output is not None
        else:
            # Extract literal argument declarations without importing or running
            # either result-processing entry point or its __main__ block.
            tree = ast.parse((ROOT / entry).read_text(encoding='utf-8'))
            parser = argparse.ArgumentParser()
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'):
                    continue
                flags = [ast.literal_eval(arg) for arg in node.args]
                options = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg in ('action', 'required', 'nargs', 'choices')}
                parser.add_argument(*flags, **options)
            parser.parse_args(arguments)
    for entry in families:
        assert any(parts[1] == entry and '--verify-existing' in parts for parts in commands)
        assert any(parts[1] == entry and '--rerun' in parts and '--output' in parts for parts in commands)
    assert any(parts[1] == 'experiments/make_paper_outputs.py' and '--verify' in parts for parts in commands)
    assert (ROOT / 'reference_results/aggregates/main_final_accuracy.csv').is_file()
    data = '\n'.join(path.read_text(encoding='utf-8') for path in (ROOT / 'data').rglob('*.md'))
    assert {
        'https://www.kaggle.com/datasets/hojjatk/mnist-dataset',
        'https://www.kaggle.com/datasets/zalando-research/fashionmnist',
        'https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset',
        'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset',
    } <= set(re.findall(r'https://[^)\s]+', data))


def test_no_implementation_overclaims_or_table_notes():
    text = '\n'.join((ROOT / p).read_text(encoding='utf-8') for p in ('README.md', 'IMPLEMENTATION_NOTE.md'))
    for phrase in ('strictly implements', 'exactly matches the latest algorithm', 'client-level DP', '\\tablenotes', '\\footnotetext'):
        assert phrase not in text


@pytest.mark.parametrize('suffix', ['.md', '.json'])
def test_network_share_is_a_private_path(tmp_path, suffix):
    value = chr(92) * 2 + 'private-host' + chr(92) + 'private-share' + chr(92) + 'input.npz'
    (tmp_path / ('file' + suffix)).write_text(json.dumps({'input': value}) if suffix == '.json' else value, encoding='utf-8')
    assert ('file' + suffix, 'private absolute path') in audit_tree(tmp_path, {})


def test_symlink_is_rejected(tmp_path):
    target = tmp_path / 'target.txt'
    target.write_text('content', encoding='utf-8')
    link = tmp_path / 'link.txt'
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip('host does not permit symlink creation')
    assert ('link.txt', 'symlink') in audit_tree(tmp_path, {})


@pytest.mark.parametrize('suffix', ['.ps1', '.sh'])
@pytest.mark.parametrize('content,reason', [('X' + ':/private/input.npz', 'private absolute path'), ('bad' + chr(0xFFFD), 'replacement character')])
def test_shell_scripts_cannot_hide_text_contamination(tmp_path, suffix, content, reason):
    (tmp_path / ('run' + suffix)).write_text(content, encoding='utf-8')
    assert ('run' + suffix, reason) in audit_tree(tmp_path, {})


@pytest.fixture
def data_policy_tree(tmp_path, monkeypatch):
    original = ROOT
    for relative in ('data/checksums.json', 'configs/main_comparison/manifest.json', 'configs/sensitivity/manifest.json', 'configs/heterogeneity/manifest.json'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original / relative).read_bytes())
    monkeypatch.setitem(globals(), 'ROOT', tmp_path)
    return tmp_path, original


def test_cleanroom_may_omit_excluded_snapshot_input(data_policy_tree):
    test_data_policy_is_backed_by_manifest_inputs()


def _present_input_fixture(root):
    # Synthetic bytes exercise presence policy only; never require the excluded
    # external NPZ to be available in a clean-room test environment.
    raw = b'self-contained presence-policy fixture'
    relative = 'src/spade_dfl_release/vendor/heterogeneity/data/mnist.npz'
    for name, field in [('data/checksums.json', 'files'), ('configs/heterogeneity/manifest.json', 'required_inputs')]:
        document = root / name
        data = json.loads(document.read_text(encoding='utf-8'))
        row = next(r for r in data[field] if r.get('placement', r.get('path')) in (relative, 'data/mnist.npz'))
        row.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        document.write_text(json.dumps(data), encoding='utf-8')
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    return path


def test_matching_workspace_snapshot_input_is_accepted(data_policy_tree):
    root, _ = data_policy_tree
    _present_input_fixture(root)
    test_data_policy_is_backed_by_manifest_inputs()


@pytest.mark.parametrize('field,value', [('included_in_release', True), ('release_builder_exclusion_required', False)])
def test_missing_input_cannot_weaken_zip_exclusion(data_policy_tree, field, value):
    root, _ = data_policy_tree
    path = root / 'data/checksums.json'
    policy = json.loads(path.read_text(encoding='utf-8'))
    policy['files'][0][field] = value
    path.write_text(json.dumps(policy), encoding='utf-8')
    with pytest.raises(AssertionError):
        test_data_policy_is_backed_by_manifest_inputs()


@pytest.mark.parametrize('mutation', ['size', 'sha256'])
def test_present_excluded_snapshot_input_must_match_bytes(data_policy_tree, mutation):
    root, _ = data_policy_tree
    path = _present_input_fixture(root)
    raw = bytearray(path.read_bytes())
    if mutation == 'size':
        raw.append(0)
    else:
        raw[0] ^= 1
    path.write_bytes(raw)
    with pytest.raises(AssertionError):
        test_data_policy_is_backed_by_manifest_inputs()

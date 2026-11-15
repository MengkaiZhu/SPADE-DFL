import pytest
import numpy as np


def test_population_and_sample_sd_are_not_mixed():
    from spade_dfl_release.statistics import mean_sd
    assert mean_sd([1, 2, 3], ddof=0) == pytest.approx((2, np.sqrt(2 / 3)))
    assert mean_sd([1, 2, 3], ddof=1) == pytest.approx((2, 1))


def test_alzheimer_folds_are_pooled_within_seed():
    from spade_dfl_release.statistics import pool_fold_counts
    rows = [dict(seed=0, correct=8, count=10), dict(seed=0, correct=18, count=20),
            dict(seed=1, correct=7, count=10), dict(seed=1, correct=16, count=20)]
    assert pool_fold_counts(rows) == {0: pytest.approx(26 / 30), 1: pytest.approx(23 / 30)}


def test_missing_seed_is_rejected():
    from spade_dfl_release.statistics import mean_sd
    with pytest.raises(ValueError, match='expected seeds'):
        mean_sd([.9], ddof=1, expected_count=5)


@pytest.mark.parametrize('values,ddof', [([], 0), ([1], 1), ([float('nan')], 0), ([1, 2], -1)])
def test_invalid_statistics_rejected(values, ddof):
    from spade_dfl_release.statistics import mean_sd
    with pytest.raises(ValueError):
        mean_sd(values, ddof)


def test_invalid_fold_counts_rejected():
    from spade_dfl_release.statistics import pool_fold_counts
    with pytest.raises(ValueError):
        pool_fold_counts([dict(seed=0, correct=2, count=1)])


def test_history_requires_all_rounds_and_finite_accuracy():
    from spade_dfl_release.reference import read_history
    assert read_history(b'round,accuracy\n0,0.25\n1,0.5\n', rounds=1).tolist() == [.25, .5]
    for data in [b'round,accuracy\n0,0.2\n', b'round,accuracy\n0,0.2\n0,0.3\n',
                 b'round,accuracy\n0,0.2\n1,nan\n']:
        with pytest.raises(ValueError):
            read_history(data, rounds=1)


def test_duplicate_and_missing_keys_rejected():
    from spade_dfl_release.reference import require_keys
    require_keys([('a', 0), ('a', 1)], {('a', 0), ('a', 1)})
    for keys in [[('a', 0)], [('a', 0), ('a', 0)]]:
        with pytest.raises(ValueError):
            require_keys(keys, {('a', 0), ('a', 1)})


def test_hash_mismatch_is_rejected_before_read(tmp_path):
    from spade_dfl_release.reference import verify_hash
    p = tmp_path / 'source.csv'
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        verify_hash(p, '0' * 64)


def test_bh_uses_entire_family_and_monotonic_adjustment():
    from spade_dfl_release.statistics import benjamini_hochberg
    assert benjamini_hochberg([.01, .04, .03, .8]) == pytest.approx([.04, .05333333333333334, .05333333333333334, .8])


def test_missing_reference_outputs_fail_closed(tmp_path):
    from spade_dfl_release.reference import verify_current_paper
    manuscript = tmp_path / 'paper.tex'
    manuscript.write_text('No tables')
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_current_paper(tmp_path, manuscript)


def test_figure10_values_are_parsed_from_prose_by_schedule():
    from spade_dfl_release.reference import parse_communication_prose
    text = r'''\subsection{Communication--Optimization Trade-off}
At $T=29{,}791$, the transmitted volume decreased from $23{,}832{,}800$ to $768{,}800$ model scalars, corresponding to a reduction of approximately $96.8\%$.
At the largest budget, its mean value was $0.010392$ under the local-update schedule and $0.000372$ under $\tau=1$, based on 30 paired seeds.'''
    values = parse_communication_prose(text)
    assert values == {'local_mean': '0.010392', 'baseline_mean': '0.000372', 'budget': '29791', 'baseline_scalars': '23832800', 'local_scalars': '768800', 'reduction_percent': '96.8'}
    with pytest.raises(ValueError):
        parse_communication_prose('no communication prose')


def test_checked_output_publisher_is_atomic_on_existing_conflict(tmp_path):
    from spade_dfl_release.reference import publish_outputs
    root = tmp_path / 'out'
    (root / 'paper_values').mkdir(parents=True)
    (root / 'paper_values/verification.json').write_text('old')
    with pytest.raises(ValueError):
        publish_outputs(root, {'a.csv': b'x\n1\n'}, '{}\n')
    assert not (root / 'aggregates/a.csv').exists()


def test_main_statistics_scale_after_fraction_aggregation():
    from spade_dfl_release.reference import aggregate_main_curves
    mean, sd = aggregate_main_curves(np.array([[.25, .5], [.75, 1.0]]))
    assert mean.tolist() == [50., 75.]
    assert sd.tolist() == [25., 25.]


def test_display_uses_approved_decimal_half_up():
    from spade_dfl_release.reference import format_display
    assert format_display(74.425, 2) == '74.43'
    assert format_display(93.275, 2) == '93.28'


def test_main_decimal_mean_avoids_binary_accumulation_below_tie():
    from spade_dfl_release.reference import aggregate_main_curves
    mean, _ = aggregate_main_curves(np.full((30, 51), .93275))
    assert mean[-1] == 93.275


@pytest.fixture
def paper_contract_fixture(tmp_path):
    """Portable table/prose contract; never redistributes or reads author prose."""
    from spade_dfl_release.paths import release_root
    # The six fixed display tokens are the already-audited Figure 10 fixture
    # used by test_figure10_values_are_parsed_from_prose_by_schedule above.
    prose = r'''\subsection{Communication--Optimization Trade-off}
At $T=29{,}791$, the transmitted volume decreased from $23{,}832{,}800$ to $768{,}800$ model scalars, corresponding to a reduction of approximately $96.8\%$.
At the largest budget, its mean value was $0.010392$ under the local-update schedule and $0.000372$ under $\tau=1$, based on 30 paired seeds.'''
    tables = release_root() / 'paper_outputs/tables'
    path = tmp_path / 'paper-contract.tex'
    path.write_text(prose + '\n' + '\n'.join((tables / name).read_text(encoding='utf-8') for name in ('table_II.tex', 'table_III.tex', 'table_IV.tex')), encoding='utf-8')
    return path


def test_verifier_rejects_tampered_non_table_evidence(tmp_path, paper_contract_fixture):
    import shutil
    from spade_dfl_release.paths import release_root
    from spade_dfl_release.reference import verify_current_paper
    original = release_root() / 'reference_results'
    shutil.copytree(original / 'aggregates', tmp_path / 'aggregates')
    shutil.copytree(original / 'paper_values', tmp_path / 'paper_values')
    path = tmp_path / 'aggregates/s15_roundwise.csv'
    data = path.read_text(encoding='utf-8')
    path.write_text(data.replace('0.5176233635448136', '0.6176233635448136', 1), encoding='utf-8')
    assert path.read_bytes() != (original / 'aggregates/s15_roundwise.csv').read_bytes()
    with pytest.raises(ValueError, match='hash'):
        verify_current_paper(tmp_path, paper_contract_fixture)


def test_published_frozen_results_verify_all_display_fields(monkeypatch, paper_contract_fixture):
    from pathlib import Path
    from spade_dfl_release.paths import release_root
    from spade_dfl_release.reference import verify_current_paper
    original_read = Path.read_text
    def read_without_author_workspace(path, *args, **kwargs):
        if path.name == 'spade_dfl_v1.tex':
            raise AssertionError('clean-room tests must not read the author manuscript')
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_without_author_workspace)
    result = verify_current_paper(release_root() / 'reference_results', paper_contract_fixture)
    assert result['displayed_numeric_fields'] == 326
    assert result['difference_count'] == 0
    assert result['missing_cells'] == result['duplicate_cells'] == 0


def test_formal_manifest_rejects_smoke_seed_set_and_duplicate_runs():
    from spade_dfl_release.reference import validate_suite_manifest
    manifest = dict(status='completed', rounds=50, seeds=list(range(30)), methods=['m'], partitions=['p'])
    validate_suite_manifest(manifest, ('p',), ('m',), 1)
    with pytest.raises(ValueError):
        validate_suite_manifest(dict(manifest, seeds=[0]), ('p',), ('m',), 1)
    with pytest.raises(ValueError):
        validate_suite_manifest(dict(manifest, runs=[dict(partition='p', method='m',seed=0)]*30), ('p',), ('m',), 1)


@pytest.mark.parametrize('field', ['query_count', 'transmitted_scalars'])
def test_communication_rejects_coordinated_raw_and_summary_accounting_error(tmp_path, field):
    import csv
    import hashlib
    import io
    import json
    from spade_dfl_release.paths import release_root
    from spade_dfl_release.reference import Sources, csv_rows, import_communication

    release = release_root()
    packaged = release / 'reference_results/source_data/communication'
    relative_paths = ('results/communication_cost_per_seed.csv',
                      'results/communication_cost_summary.csv', 'results/experiment_metadata.json')
    entries = []
    for relative in relative_paths:
        data = (packaged / relative.rsplit('/', 1)[-1]).read_bytes()
        if relative.endswith('.csv'):
            rows = csv_rows(data)
            for row in rows:
                row[field] = str(int(row[field]) + 1)
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
            data = stream.getvalue().encode('utf-8')
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        entries.append(dict(source_id='communication_figure10', path=relative,
                            sha256=hashlib.sha256(data).hexdigest()))
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'sources': [{'id': 'communication_figure10', 'root': str(tmp_path)}]}))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': entries}))
    changed = Sources(index, manifest)
    # Both CSVs and their hashes agree; only independent configuration-based
    # accounting can reject this coordinated historical accounting error.
    with pytest.raises(ValueError, match='accounting formula mismatch: ' + field):
        import_communication(changed)

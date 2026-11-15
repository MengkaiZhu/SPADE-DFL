"""Pure saved-record tests; no optimizer, dataset loader or training is invoked."""
import importlib.util
import pytest

def module():
    assert importlib.util.find_spec('spade_dfl_release.main_recompute') is not None, 'public seed-level recomputation is missing'
    from spade_dfl_release import main_recompute
    return main_recompute

def test_public_main_recomputation_matches_all_frozen_statistics(release_root):
    result=module().verify(release_root)
    assert result['runs']==6720
    assert result['points']==342720
    assert result['final_rows']==112
    assert result['curve_rows']==5712
    assert result['max_numeric_difference']==0

def test_missing_run_is_rejected(release_root):
    m=module();rows=m.read_index(release_root)
    with pytest.raises(ValueError,match='run identities'):
        m.validate_index(rows[:-1])

def test_duplicate_run_is_rejected(release_root):
    m=module();rows=m.read_index(release_root)
    with pytest.raises(ValueError,match='run identities'):
        m.validate_index(rows+[rows[0]])

def test_incomplete_rounds_are_rejected():
    with pytest.raises(ValueError,match='rounds'):
        module().read_curve([{'round':'0','accuracy':'0.5'}])

def test_nonfinite_accuracy_is_rejected():
    rows=[{'round':str(i),'accuracy':'0.5'} for i in range(51)]
    rows[1]['accuracy']='nan'
    with pytest.raises(ValueError,match='accuracy'):
        module().read_curve(rows)

def test_changed_statistics_are_rejected():
    a=[dict(dataset='mnist',partition='iid',method='spade_dfl',round=50,mean_percent=50,sd_percent=2,n_trajectories=30,ddof=0)]
    b=[dict(a[0],mean_percent=51)]
    with pytest.raises(ValueError,match='statistics differ'):
        module().compare(a,b)

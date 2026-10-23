"""Non-physical cache regression tests. Do not replace scientific benchmarks."""
from types import SimpleNamespace
import numpy as np
import pytest
from cualni_cryst.theory_equivalence import PhysicalBranchMatcher
from cualni_cryst.theory_unified import ComparisonRow, TheoryKind, PredictionKind


class FakeOrientationService:
    """Instrument calls without changing the real SO(3) or symmetry solver."""
    def __init__(self):
        self.created = 0
        self.reexpressed = 0

    def state_from_matrix(self, *args, **kwargs):
        self.created += 1
        matrix = np.asarray(args[2], dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Bad OR matrix")
        return SimpleNamespace(R=matrix.copy())

    def reexpress(self, state, *_args):
        self.reexpressed += 1
        return SimpleNamespace(R_reference_from_moving=state.R + np.eye(3)*0.125)


def matcher():
    obj = object.__new__(PhysicalBranchMatcher)
    obj.parent = SimpleNamespace(phase_id='A')
    obj.product = SimpleNamespace(phase_id='M')
    obj.transformation_id = 'T'
    obj.orientation_service = FakeOrientationService()
    obj._or_frame_cache = {}
    return obj


def row(id_, R):
    return ComparisonRow(row_id=id_, theory=TheoryKind.PTMC,
        prediction_kind=PredictionKind.PTMC_HABIT,
        branch_label=id_,
        or_parent_from_product=tuple(tuple(float(v) for v in a) for a in R))


def _uncached_oracle(obj, entry):
    """Old, unoptimized public-service route, constructed independently."""
    from cualni_cryst.representation import CartesianConvention
    state = obj.orientation_service.state_from_matrix(
        obj.parent.phase_id, obj.product.phase_id,
        np.asarray(entry.or_parent_from_product, dtype=float),
        orientation_id=f'equivalence_{entry.row_id}',
        reference_convention=CartesianConvention.PTCLAB_A_X_C_XZ,
        moving_convention=CartesianConvention.PTCLAB_A_X_C_XZ,
        transformation_id=obj.transformation_id)
    symmetric = obj.orientation_service.reexpress(state,
        CartesianConvention.SYMMETRIC_METRIC,
        CartesianConvention.SYMMETRIC_METRIC)
    return np.asarray(symmetric.R_reference_from_moving, dtype=float)


def test_repeated_same_input_has_identical_numbers_and_one_conversion():
    obj = matcher()
    entry = row('A1', np.eye(3))
    expected = _uncached_oracle(obj, entry)
    obj.orientation_service.created = obj.orientation_service.reexpressed = 0
    actual = obj._or_in_symmetric_metric_frame(entry)
    repeated = obj._or_in_symmetric_metric_frame(entry)
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(expected, repeated)
    assert (obj.orientation_service.created, obj.orientation_service.reexpressed)==(1,1)


def test_return_value_mutation_does_not_poison_cache():
    obj=matcher()
    entry=row('A2', np.eye(3))
    first=obj._or_in_symmetric_metric_frame(entry)
    first[0,0]=17.0
    second=obj._or_in_symmetric_metric_frame(entry)
    assert second[0,0]==1.125
    second[0,0]=19.0
    assert obj._or_in_symmetric_metric_frame(entry)[0,0]==1.125


def test_exactly_distinct_matrix_does_not_alias():
    obj=matcher()
    R=np.eye(3)
    entry1=row('A3', R)
    R[0,1]=1e-12
    entry2=row('A3', R)
    x=obj._or_in_symmetric_metric_frame(entry1)
    y=obj._or_in_symmetric_metric_frame(entry2)
    assert y[0,1]!=x[0,1]
    assert obj.orientation_service.created==2
    assert len(obj._or_frame_cache)==2


def test_different_row_ids_do_not_alias():
    obj=matcher()
    obj._or_in_symmetric_metric_frame(row('A4',np.eye(3)))
    obj._or_in_symmetric_metric_frame(row('A5',np.eye(3)))
    assert obj.orientation_service.created==2


def test_absent_or_preserves_none():
    obj=matcher()
    e=row('A6',np.eye(3))
    from dataclasses import replace
    assert obj._or_in_symmetric_metric_frame(replace(e,or_parent_from_product=None)) is None
    assert obj.orientation_service.created==0


def test_bounded_cache_discard_preserves_equality():
    obj=matcher()
    for n in range(4097):
        R=np.eye(3); R[0,1]=n*1e-11
        actual=obj._or_in_symmetric_metric_frame(row(str(n),R))
        assert actual[0,1]==n*1e-11
    assert len(obj._or_frame_cache)<=4096


def test_invalid_or_rejected_without_cache_pollution():
    obj=matcher()
    invalid=row('bad',np.full((3,3),np.nan))
    with pytest.raises(ValueError):
        obj._or_in_symmetric_metric_frame(invalid)
    assert not obj._or_frame_cache

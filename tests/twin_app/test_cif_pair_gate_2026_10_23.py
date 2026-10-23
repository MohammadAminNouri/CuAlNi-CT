"""Regression: no mixed-phase CIF calculations or stale CIF acknowledgments."""
import ast
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=(ROOT/'twin_app/streamlit_app.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE)
NAMES={'_cif_pair_signature','_cif_admission_reason'}
FUNCTIONS=[node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name in NAMES]
assert {n.name for n in FUNCTIONS}==NAMES
ns={'hashlib':hashlib}
exec(compile(ast.Module(body=FUNCTIONS,type_ignores=[]),'cif_pair_gate','exec'),ns)


def test_missing_or_extra_phase_is_rejected():
    f=ns['_cif_pair_signature']
    assert f({}) is None
    assert f({'parent':b'P'}) is None
    assert f({'product':b'M'}) is None
    assert f({'parent':b'P','product':b'M','extra':b'E'}) is None


def test_pair_hash_is_role_sensitive_and_content_sensitive():
    f=ns['_cif_pair_signature']
    a=f({'parent':b'P','product':b'M'})
    assert a
    assert a!=f({'parent':b'M','product':b'P'})
    assert a!=f({'parent':b'P','product':b'M1'})


def test_import_must_be_applied_and_confirmed():
    f=ns['_cif_admission_reason']
    s='valid-pair'
    assert f(pair_signature=None,applied_signature=None,confirmed=True)
    assert f(pair_signature=s,applied_signature=None,confirmed=True)
    assert f(pair_signature=s,applied_signature='stale',confirmed=True)
    assert f(pair_signature=s,applied_signature=s,confirmed=False)
    assert f(pair_signature=s,applied_signature=s,confirmed=True) is None


def test_ui_contains_fail_closed_calculation_gate():
    assert 'disabled=cif_pair_signature is None' in SOURCE
    assert 'source == "Import CIF structures" and cif_admission_reason is not None' in SOURCE
    assert '"cif_pair": cif_pair_signature, "project": payload' in SOURCE


def test_numerical_solver_pipeline_is_preserved():
    for phrase in (
        'build_twin_family_report(', 'validate_report_integrity(report)',
        'build_mode_catalogue(report, payload)',
    ):
        assert phrase in SOURCE

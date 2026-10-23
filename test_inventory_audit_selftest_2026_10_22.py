"""Standalone tests of the independent auditor, without requiring repository imports."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mpmath as mp

p=Path(__file__).with_name('independent_solution_inventory_audit_2026_10_22.py')
spec=importlib.util.spec_from_file_location('inventory_audit',p)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def test_oracle_synthetic_root_inventory():
    with mp.workdps(audit.MP_DPS):
        U=audit.mat(np.diag([0.8,0.9,1.2]))
        r=SimpleNamespace(a=np.array([1.,0.,0.]),n_reference=np.array([0.,1.,0.]))
        row=audit.root_diagnostics(U,r,[0.2615339366124404])
        assert row['classification']=='MATCH'
        row2=audit.root_diagnostics(U,r,[])
        assert row2['classification']=='MISMATCH' and len(row2['missing'])==1
        row3=audit.root_diagnostics(U,r,[0.5])
        assert row3['classification']=='MISMATCH' and len(row3['unexpected'])==1

def test_discriminant_topology():
    with mp.workdps(audit.MP_DPS):
        assert audit.isolated_roots(tuple(map(mp.mpf,['1','0','1'])))[0]=='no_real_roots'
        assert audit.isolated_roots(tuple(map(mp.mpf,['0.25','-1','1'])))[0]=='double_root'
        assert audit.isolated_roots(tuple(map(mp.mpf,['-1','0','1'])))[0]=='two_real_roots'
        assert audit.isolated_roots(tuple(map(mp.mpf,['0','0','0'])))[0]=='exact_zero_polynomial'
        assert audit.isolated_roots(tuple(map(mp.mpf,['0','1','0'])))[0]=='linear'

def test_pair_checker_incompatible_pair():
    V=[SimpleNamespace(U=np.diag([.7,.8,1.1])),SimpleNamespace(U=np.diag([.9,1.3,1.4]))]
    with mp.workdps(audit.MP_DPS):
        rows=audit.pair_oracle(V,[])
    assert len(rows)==2
    assert [r['classification'] for r in rows]==['MATCH','MATCH'],rows

def test_complete_mock_report_with_no_compatible_pairs(monkeypatch):
    V=[SimpleNamespace(U=np.diag([.7,.8,1.1])),SimpleNamespace(U=np.diag([.9,1.3,1.4]))]
    report=SimpleNamespace(variants=V,twin_relations=[],parameter_roots=[],solutions=[])
    mock_model=SimpleNamespace(analyze_all_twinning=lambda: report)
    monkeypatch.setattr(audit,'case_input',lambda case:mock_model)
    with mp.workdps(audit.MP_DPS):
        result=audit.audit_case('mock')
    assert result['ordered_pair_count']==2
    assert result['mismatches']==[] and result['inconclusive']==[]
    assert result['root_parameter_records']==0


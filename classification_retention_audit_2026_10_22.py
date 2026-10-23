#!/usr/bin/env python3
"""Read-only classification-provenance and *data-model* solution-retention audit.

Scope: 2H Bhattacharya and 6M James--Hane encoded test inputs.
Each report correspondence construction (NOT stretch pair alone) must occur
exactly once in the ModeCatalogue, carrying the identical immutable
classification, evidence string, and complete habit-solution inventory.
Geometric twofold witnesses are independently recomputed as a necessary
screen; they are not sufficient CT subtype proofs. Never force labels.

Does NOT certify actual browser/Streamlit rendering, specimen input, or
complete discrete-CT correctness. Does not modify application code.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import sympy as sp

TYPES = ("Type I", "Type II", "Compound")
TENSOR_TOL = 2.e-7
GEO_TOL = 2.e-7


def model_input(case):
    """Build the *same encoded literature inputs* through the public project API."""
    from app.application import CalculationRequest, LatticeInput, PhaseInput, TransformationInput
    def phase(name, x, group, phase_id):
        return PhaseInput(phase_id, name,
            LatticeInput(float(x.a),float(x.b),float(x.c),
                         float(x.alpha_deg),float(x.beta_deg),float(x.gamma_deg),
                         'angstrom'),group)
    if case == '2h':
        bench=json.loads(Path('data/benchmarks/bhattacharya_cualni_orthorhombic_habit_v1.json').read_text(encoding='utf-8'))
        inp=bench['input']
        def p(k,pid):
            x=inp[k]
            return PhaseInput(pid,k,LatticeInput(x['a'],x['b'],x['c'],x['alpha'],x['beta'],x['gamma'],'angstrom'),x['point_group'])
        a=p('parent','A');m=p('product','M');C=inp['C_M_from_A']
    elif case == '6m':
        from cualni_cryst.cualni_models import james_hane_6m_example_lattices,do3_to_6m_branch
        la,lm=james_hane_6m_example_lattices()
        a=phase('Parent',la,'m-3m','A');m=phase('Martensite',lm,'2/m','M')
        C=[[str(x) for x in row] for row in do3_to_6m_branch().correspondence.C_M_from_A.tolist()]
    else:raise ValueError(case)
    req=CalculationRequest('independent_classification_retention_'+case,'Audit '+case,
        a,m,TransformationInput.from_rows('A_to_M','A','M',C))
    return req.to_project_payload(),'A_to_M'


def normalized_dyad(a,n):
    x=np.outer(np.asarray(a,float),np.asarray(n,float))
    assert np.isfinite(x).all() and np.linalg.norm(x)>0
    return x


def dyad_error(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1.e-20))


def label_and_status(branch):
    # Preserve unresolved and explicitly proved nonconventional distinctions.
    from twin_app.mode_type_status import supported_classical_type
    typ=str(branch.classification)
    status=str(branch.classification_status)
    if typ in TYPES:
        return (typ, bool(supported_classical_type(typ,status)))
    return ('Unresolved',False)


def generator_screen(base,other,dyad,model):
    """Necessary twofold geometric screen. Not a sufficient CT/compound proof."""
    from cualni_cryst.point_groups import point_group_operations
    M=np.asarray(model.parent_lattice.metric(),float)
    w,V=np.linalg.eigh(M)
    B=(V*np.sqrt(w))@V.T; Bi=np.linalg.inv(B)
    # Principal vector of D is the shear line; row space of D is plane normal.
    left,s,right=np.linalg.svd(dyad)
    if s[1]/max(s[0],1.e-200)>1.e-7: return {'candidate':'Unresolved','reason':'dyad not numerically rank one','witnesses':[]}
    a=left[:,0];n=right[0,:]
    results=[]
    for gi,g in enumerate(point_group_operations(model.parent_point_group)):
        Q=B@np.asarray(g,float)@Bi
        if (abs(np.linalg.det(Q)-1)>1.e-8 or abs(np.trace(Q)+1)>1.e-8
             or np.linalg.norm(Q@Q-np.eye(3))>1.e-8):continue
        mapping=dyad_error(Q@base@Q.T,other)
        if mapping>1.e-8:continue
        vals,vecs=np.linalg.eigh((Q+Q.T)/2)
        axis=vecs[:,int(np.argmin(abs(vals-1)))]
        i_err=1-abs(float(n@axis))
        Ua=base@axis
        ii_err=1-abs(float(a@Ua))/max(float(np.linalg.norm(Ua)),1.e-25)
        results.append({'operation':gi,'mapping_error':mapping,
                        'type_i_normal_alignment_error':i_err,
                        'type_ii_shear_alignment_error':ii_err})
    has_i=any(q['type_i_normal_alignment_error']<GEO_TOL for q in results)
    has_ii=any(q['type_ii_shear_alignment_error']<GEO_TOL for q in results)
    candidate='Compound' if has_i and has_ii else 'Type I' if has_i else 'Type II' if has_ii else 'Unresolved'
    return {'candidate':candidate,'has_I_witness':has_i,'has_II_witness':has_ii,
            'proper_twofold_generators':len(results),'witnesses':results}


def build_items_from_report(report):
    """Keys are identity in correspondence topology, NOT solely stretches/dyads."""
    rows={};duplicates=[]
    for fam in report.families:
        for pair in fam.pair_records:
            for b in pair.constructions:
                key=(str(fam.family_id),str(pair.pair_id),str(b.construction_id),int(b.branch))
                if key in rows:duplicates.append(key)
                rows[key]=(fam,pair,b)
    return rows,duplicates


def build_items_from_catalogue(cat):
    rows=defaultdict(list)
    for mode in cat.modes:
        for entry in mode.entries:
            p=entry.pair;b=entry.construction
            key=(str(entry.operator_family),str(p.pair_id),str(b.construction_id),int(b.branch))
            rows[key].append((mode,p,b))
    return rows


def check_retention(records,mode_records,catalogue):
    """Strict multiset equality and no modification of individual solution records."""
    failures=[];rows=[]
    for key,(family,pair,branch) in records.items():
        matches=mode_records.get(key,[])
        if len(matches)!=1:
            failures.append({'key':key,'problem':'construction missing or duplicated in mode catalogue','occurrences':len(matches)})
            continue
        mode,mpair,mb=matches[0]
        # Every individual field, including classification provenance & habit tensors.
        identical=asdict(branch)==asdict(mb) and asdict(pair)==asdict(mpair)
        if not identical:failures.append({'key':key,'problem':'construction or pair data changed during mode grouping'})
        key_type,valid=label_and_status(branch)
        if key_type in TYPES and not valid:
            failures.append({'key':key,'problem':'classical type lacks allowed direct/transported evidence','type':key_type,
                             'status':str(branch.classification_status)})
        if branch.continuum_fraction and branch.habit_solutions:
            failures.append({'key':key,'problem':'continuum record improperly carries discrete habit roots'})
        rows.append({'key':list(key),'mode_id':mode.mode_id,'type':key_type,
                     'status':str(branch.classification_status),
                     'direct':str(branch.classification_status).startswith('cross-locked:'),
                     'transported':str(branch.classification_status).startswith('symmetry-transported from directly cross-locked '),
                     'habits':len(branch.habit_solutions),
                     'stretches':[pair.stretch_i,pair.stretch_j]})
    for key,v in mode_records.items():
        if key not in records:failures.append({'key':key,'problem':'invented mode entry','occurrences':len(v)})
    for mode in catalogue.modes:
        declared=set(mode.types)
        expected={label_and_status(x.construction)[0] for x in mode.entries}
        if declared!=expected:failures.append({'mode':mode.mode_id,'problem':'mode.types differs from actual branch types',
                                                'declared':sorted(declared),'expected':sorted(expected)})
        for kind in ('Compound','Type I','Type II','Unresolved'):
            expected_keys={id(e) for e in mode.entries if label_and_status(e.construction)[0]==kind}
            got_keys={id(e) for e in mode.by_type(kind)}
            if expected_keys!=got_keys:
                failures.append({'mode':mode.mode_id,'type':kind,'problem':'mode.by_type filters incorrectly'})
        if not mode.entries:failures.append({'mode':mode.mode_id,'problem':'empty physical mode'})
    return rows,failures


def check_classification(records,model,ptmc):
    """Geometry candidate is a necessary witness screen; it never creates labels."""
    failures=[];reviews=[];rows=[]
    relations={}
    for r in ptmc.twin_relations:
        key=(int(r.base_variant_index),int(r.other_variant_index),int(r.branch))
        if key in relations:failures.append({'key':key,'problem':'duplicate PTMC twin relation'})
        relations[key]=r
    for key,(fam,p,b) in records.items():
        base=p.stretch_i; other=p.stretch_j
        row={'key':list(key),'label':str(b.classification),'status':str(b.classification_status),
             'geometry_candidate':None,'max_tensor_residual':None,'witnesses':[]}
        if base is None or other is None or base==other:
            failures.append({'key':key,'problem':'construction without two distinct mapped stretches'});rows.append(row);continue
        pp=(int(base)-1,int(other)-1,int(b.branch))
        relation=relations.get(pp)
        if relation is None:
            failures.append({'key':key,'problem':'constructed branch missing in PTMC ordered relation inventory','pair_branch':pp})
            rows.append(row);continue
        independent=normalized_dyad(b.a_parent_cartesian,b.n_parent_cartesian)
        physical=normalized_dyad(relation.a,relation.n_reference)
        err=dyad_error(independent,physical)
        row['max_tensor_residual']=err
        if err>TENSOR_TOL:failures.append({'key':key,'problem':'classified construction has wrong physical dyad','error':err})
        ui=np.asarray(ptmc.variants[pp[0]].U,float)
        uj=np.asarray(ptmc.variants[pp[1]].U,float)
        screen=generator_screen(ui,uj,independent,model)
        row['geometry_candidate']=screen['candidate'];row['witnesses']=screen['witnesses']
        typ,valid=label_and_status(b)
        if typ in TYPES:
            if not valid:failures.append({'key':key,'problem':'classical label without accepted provenance'})
            # A necessary geometry witness can falsify classical type claims
            # only when a proper twofold exists. A separate CT mirror path
            # may exist even without a proper twofold, so log as REVIEW.
            if (screen['proper_twofold_generators'] and
                    ((typ=='Type I' and not screen['has_I_witness']) or
                     (typ=='Type II' and not screen['has_II_witness']) or
                     (typ=='Compound' and not(screen['has_I_witness'] and screen['has_II_witness'])))):
                reviews.append({'key':key,'type':typ,'geometry_candidate':screen['candidate'],
                                'reason':'classical label lacks expected twofold-alignment witness; direct CT/mirror-frame cross-lock needs review'})
            if not str(b.classification_status).startswith(('cross-locked:',
              'symmetry-transported from directly cross-locked ')):
                failures.append({'key':key,'problem':'classical label status contains no recognized provenance'})
        else:
            if screen['candidate'] in TYPES:
                # Not a defect: geometric witness is NOT a discrete CT proof.
                reviews.append({'key':key,'type':'Unresolved','geometry_candidate':screen['candidate'],
                                'reason':'geometric witness exists but discrete CT cross-lock is not certified; leave unresolved'})
        if str(b.classification)=='Compound' and set(b.representations)!=set(('I','II')):
            failures.append({'key':key,'problem':'compound not supported by both representation tags'})
        if str(b.classification)=='Type I' and 'I' not in b.representations:
            failures.append({'key':key,'problem':'Type I has no I route tag'})
        if str(b.classification)=='Type II' and 'II' not in b.representations:
            failures.append({'key':key,'problem':'Type II has no II route tag'})
        rows.append(row)
    return rows,failures,reviews


def check_discrete_sources(report,parent_symmetries):
    """Inspect source CT records without reusing the CT solver's decision logic.

    Necessary crystallographic invariants only: a claimed generator must be
    an actual parent symmetry, (K,eta) must obey reciprocal/direct incidence,
    nonzero shear, and directly cross-locked IDs must point to actual systems.
    Not a proof of the full Type I/II/Compound classification formula.
    """
    errors=[];reviews=[];provenance=[]
    ops=[np.asarray(Q,float) for Q in parent_symmetries]
    for fam in report.families:
        systems={x.system_id:x for x in fam.classical_systems}
        for system in fam.classical_systems:
            source_routes=set()
            for representation in system.representations:
                source_routes.add(representation.route)
                tag=(fam.family_id,system.system_id,representation.route)
                G=np.array(sp.Matrix(representation.generator_parent_symmetry),dtype=float)
                sym_error=min((dyad_error(G,Q) for Q in ops),default=float('inf'))
                if sym_error>1.e-8:
                    errors.append({'system':tag,'problem':'CT representation generator not present in parent symmetry group','error':sym_error})
                p=np.asarray(representation.plane_product_crystal,float)
                d=np.asarray(representation.direction_product_crystal,float)
                if not (np.isfinite(p).all() and np.isfinite(d).all()
                            and np.linalg.norm(p)>0 and np.linalg.norm(d)>0):
                    errors.append({'system':tag,'problem':'nonfinite or zero crystallographic plane/direction'})
                else:
                    incidence=abs(float(p@d))/(np.linalg.norm(p)*np.linalg.norm(d))
                    if incidence>2.e-6:
                        errors.append({'system':tag,'problem':'shear direction not in twinning plane','relative_incidence':incidence})
                if not math.isfinite(representation.shear_magnitude) or representation.shear_magnitude<=0:
                    errors.append({'system':tag,'problem':'nonphysical CT representation shear'})
                # Mirror and twofold character are supporting, not absolute:
                # general noncentrosymmetric settings may expose other routes.
                det=float(np.linalg.det(G))
                if representation.route=='I' and det>0:
                    reviews.append({'system':tag,'reason':'Type-I CT generator has positive determinant; inspect mirror basis convention'})
                if representation.route=='II' and det<0:
                    reviews.append({'system':tag,'reason':'Type-II CT generator has negative determinant; inspect twofold basis convention'})
                provenance.append({'system':list(tag),'parent_symmetry_match_error':sym_error,
                                   'determinant':det})
            if system.classification=='Compound' and not {'I','II'}.issubset(source_routes):
                errors.append({'system':system.system_id,'problem':'compound discrete system lacks both I and II routes'})
        for pair in fam.pair_records:
            for b in pair.constructions:
                typ,status=label_and_status(b)
                direct=str(b.classification_status).startswith('cross-locked:')
                ids=list(b.classical_system_ids)
                if direct and typ in TYPES:
                    if not ids:
                        errors.append({'branch':(fam.family_id,pair.pair_id,b.construction_id),'problem':'directly certified subtype has no CT system provenance'})
                    for sid in ids:
                        if sid not in systems:
                            errors.append({'branch':(fam.family_id,pair.pair_id,b.construction_id),'problem':'direct subtype cites nonexistent CT system ID','system_id':sid})
                        elif abs(systems[sid].shear_magnitude-b.shear_magnitude)>2.e-6*max(1,abs(b.shear_magnitude)):
                            errors.append({'branch':(fam.family_id,pair.pair_id,b.construction_id),'problem':'direct subtype cites CT system with conflicting shear','system_id':sid})
    return errors,reviews,provenance

def audit_case(case, details=True):
    from twin_app.scientific_engine import build_twin_family_report
    from twin_app.mode_catalogue import build_mode_catalogue
    from cualni_cryst.project_io import project_from_dict
    from cualni_cryst.ptmc_adapter import PTMCAdapter
    payload,tid=model_input(case)
    report=build_twin_family_report(payload,tid)
    catalogue=build_mode_catalogue(report,payload)
    project=project_from_dict(dict(payload),source='read_only_audit').project
    ptmc=PTMCAdapter(project,tid).analyze_all_twinning()
    from cualni_cryst.ptmc_adapter import PTMCCrystalInput
    trans=project.transformation(tid)
    model=PTMCCrystalInput(project.phase(trans.parent_phase_id).lattice,
                           project.phase(trans.product_phase_id).lattice,trans.correspondence,
                           project.phase(trans.parent_phase_id).point_group_symbol,
                           project.phase(trans.product_phase_id).point_group_symbol)
    source,duplicates=build_items_from_report(report)
    displayed=build_items_from_catalogue(catalogue)
    retention,losses=check_retention(source,displayed,catalogue)
    classification,invalid,reviews=check_classification(source,model,ptmc)
    ct_errors,ct_reviews,ct_provenance=check_discrete_sources(report,
        project.phase(trans.parent_phase_id).symmetry_matrices())
    errors=losses+invalid+ct_errors+[{'key':x,'problem':'duplicate report construction key'} for x in duplicates]
    reviews.extend(ct_reviews)
    # Layer-0 coverage: each *unoriented* PTMC-compatible stretch pair needs
    # at least one correspondence-topology representation if all stretch
    # variants are representable by correspondence records. Ordered PTMC
    # solutions are NOT expected one-for-one in the mode catalogue.
    ptmc_pairs={tuple(sorted((int(t.base_variant_index)+1,int(t.other_variant_index)+1)))
                for t in ptmc.twin_relations}
    report_pairs={tuple(sorted((int(p.stretch_i),int(p.stretch_j))))
                  for _,p,b in source.values()
                  if p.stretch_i is not None and p.stretch_j is not None}
    missing_pair_coverage=sorted(ptmc_pairs-report_pairs)
    represented_stretches={int(r.stretch_variant_index) for r in report.correspondence_variants
                          if r.stretch_variant_index is not None}
    full_mapping=len(represented_stretches)==len(ptmc.variants)
    if missing_pair_coverage:
        finding={'problem':'PTMC-compatible unordered stretch pairs absent from all report constructions',
                 'pairs':missing_pair_coverage,'all_stretches_mapped':full_mapping}
        if full_mapping:errors.append(finding)
        else:reviews.append({**finding,'reason':'not every stretch has an accessible topological correspondence representative'})
    bylabel=Counter(x['type'] for x in retention)
    bycandidate=Counter(x['geometry_candidate'] for x in classification)
    source_habits=sum(len(x[2].habit_solutions) for x in source.values())
    output_habits=sum(len(x[2].habit_solutions) for xs in displayed.values() for x in xs)
    if source_habits!=output_habits:errors.append({'problem':'habit solution count lost in mode grouping',
                                                   'source':source_habits,'catalogue':output_habits})
    # We deliberately do NOT require these to equal raw PTMC ordered totals.
    # Topology mapping and orientation/order symmetry can create a different
    # number of correspondence records from physical stretch relations.
    out={'name':case,'topology_variant_count':report.audit.topology_variant_count,
         'stretch_variant_count':report.audit.stretch_variant_count,
         'raw_ptmc_twin_branches':len(ptmc.twin_relations),
         'raw_ptmc_habit_solutions':len(ptmc.solutions),
         'report_constructions':len(source),'mode_constructions':sum(map(len,displayed.values())),
         'report_habits':source_habits,'mode_habits':output_habits,
         'physical_modes':len(catalogue.modes),'incompatible_pair_orbits':len(catalogue.incompatible_pair_orbits),
         'ptmc_unordered_compatible_pairs':len(ptmc_pairs),'represented_unordered_stretch_pairs':len(report_pairs),
         'missing_compatible_unordered_pairs':missing_pair_coverage,
         'type_counts':dict(bylabel),'geometry_candidates':dict(bycandidate),
         'mismatches':errors,'reviews':reviews,'ct_generator_provenance':ct_provenance,
         'status':'MISMATCH' if errors else 'REVIEW_REQUIRED' if reviews else 'PASS_CONDITIONAL'}
    if details:
        out['retained_constructions']=retention
        out['classification_screens']=classification
    return out


def selftest():
    from dataclasses import dataclass
    from twin_app.scientific_models import PairTwinConstruction, VariantPairRecord
    from twin_app.mode_type_status import supported_classical_type
    D=normalized_dyad([1,0,0],[0,1,0])
    assert dyad_error(D,D)==0
    assert dyad_error(D,-D)>1.9
    assert not supported_classical_type('Type II','cross-locked claim appears in rejection')
    # Mutation guard: count and provenance checks must detect duplication/loss.
    @dataclass(frozen=True)
    class Fake:
        classification:str='bad'
        classification_status:str='unresolved'
        representations:tuple=()
        continuum_fraction:bool=False
        habit_solutions:tuple=()
        shear_magnitude:float=0.1
    @dataclass(frozen=True)
    class Pair:
        pair_id:str='M1-M2'
        stretch_i:int=1
        stretch_j:int=2
    @dataclass(frozen=True)
    class Family:
        family_id:str='F1'
    @dataclass(frozen=True)
    class Mode:
        mode_id:str='Mode 1'
        types:tuple=('Unresolved',)
        entries:tuple=()
        def by_type(self,k):return tuple(e for e in self.entries if label_and_status(e.construction)[0]==k)
    @dataclass(frozen=True)
    class Entry:
        pair:object
        construction:object
        operator_family:str
    @dataclass(frozen=True)
    class Catalogue:
        modes:tuple
    k=('F1','M1-M2','P1.B1',1)
    b=Fake();b=type('B',(),dict(Fake().__dict__,construction_id='P1.B1',branch=1))()
    p=Pair(); f=Family();e=Entry(p,b,'F1'); mode=Mode(entries=(e,))
    # This is deliberately a missing-record/extra-record test only.
    assert check_retention({k:(f,p,b)},{},Catalogue((mode,)))[1]
    assert check_retention({},{k:[(mode,p,b)]},Catalogue((mode,)))[1]
    print('SELFTEST PASS: dyad identity, source identity, missing/extra mode entries, provenance contract')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--case',choices=['2h','6m','both'],default='both')
    parser.add_argument('--summary-only',action='store_true',help='omit large per-record evidence')
    args=parser.parse_args()
    if args.selftest:selftest();return
    cases=[]
    for name in (('2h','6m') if args.case=='both' else (args.case,)):
        try:res=audit_case(name,details=not args.summary_only)
        except Exception as exc:
            res={'name':name,'status':'EXECUTION_ERROR','exception':repr(exc),'traceback':traceback.format_exc()}
        cases.append(res)
        print(f'{name}: {res["status"]} constructions {res.get("report_constructions")} -> {res.get("mode_constructions")}; habits {res.get("report_habits")} -> {res.get("mode_habits")}; errors {len(res.get("mismatches",[]))}; classification reviews {len(res.get("reviews",[]))}',flush=True)
    sha=''
    try:
        sha=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,timeout=5).stdout.strip()
    except Exception:pass
    # Fingerprints capture the actual working tree including uncommitted
    # repairs; git HEAD alone cannot reproduce this audit.
    source_files=('src/cualni_cryst/ptmc_adapter.py','src/cualni_cryst/variant_identity.py',
                  'twin_app/scientific_engine.py','twin_app/mode_catalogue.py',
                  'twin_app/symmetry_completion.py','twin_app/p0_classification.py',
                  'twin_app/mode_type_status.py')
    fingerprints={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None
                  for p in source_files}
    obj={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
         'scope':'data-model retention and necessary twofold geometry; not rendering or full CT classification proof',
         'git_head':sha,'working_tree_may_be_dirty':True,
         'working_source_sha256':fingerprints,
         'cases':cases,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=Path('audit_findings/classification_retention_2026_10_22.json')
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')
    print('EVIDENCE:',path)
    if any(x['status'] in ('MISMATCH','EXECUTION_ERROR') for x in cases):sys.exit(2)
    if any(x['status']=='REVIEW_REQUIRED' for x in cases):sys.exit(3)

if __name__=='__main__':main()

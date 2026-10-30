from __future__ import annotations

"""Read-only and non-executing migration of 2021 ARPGE phase/transition records.

The old archive contains unsafe Python pickle streams without a redistribution
license. Never unpickle them or execute Python-2 GUI code. This module reads
ONLY protocol-0 list/number/string transition records with a tiny stack VM;
crystal metrics are extracted from the first primitive opcodes, BEFORE a
NumPy GLOBAL can reconstruct any object. Remaining opcodes are inspected,
never dispatched. No phase atom positions are inferred or certified.
"""

import hashlib
import math
from pathlib import PurePosixPath
import pickletools
from zipfile import ZipFile


def _safe_transition_pickle(raw: bytes) -> list:
    if len(raw)>8192 or not raw or raw[:1] not in (b"(",):
        raise ValueError("Transition is not a small protocol-0 list")
    stack=[];mark=object();memo={};finished=False
    for op,arg,_ in pickletools.genops(raw.replace(b"\r\n",b"\n")):
        name=op.name
        if name=="MARK":stack.append(mark)
        elif name=="LIST":
            k=max((i for i,v in enumerate(stack) if v is mark),default=-1)
            if k<0:raise ValueError("Malformed transition list")
            items=stack[k+1:]
            stack[k:]=[items]
        elif name in ("STRING","INT","FLOAT"):
            if isinstance(arg,str) and len(arg)>150:
                raise ValueError("Oversized transition string")
            stack.append(arg)
        elif name=="PUT":
            if not stack or len(memo)>1000:raise ValueError("Invalid transition memo")
            memo[int(arg)]=stack[-1]
        elif name=="GET":
            if arg not in memo:raise ValueError("Unresolved transition memo")
            stack.append(memo[arg])
        elif name=="APPEND":
            if len(stack)<2 or not isinstance(stack[-2],list):
                raise ValueError("Transition APPEND must target a list")
            value=stack.pop();stack[-1].append(value)
        elif name=="STOP":
            finished=True;break
        else:
            raise ValueError(f"Unsupported or executable pickle opcode {name}")
    if not finished or len(stack)!=1 or not isinstance(stack[0],list):
        raise ValueError("Invalid protocol-0 transition record")
    return stack[0]


def _vector3(value, label:str)->list[int]|None:
    if value==[]:return None
    if (not isinstance(value,list) or len(value)!=3 or
        any(type(x) is not int or abs(x)>1000 for x in value) or
        not any(value)):
        raise ValueError(f"Invalid {label} index triplet")
    return value


def decode_transition(raw:bytes)->dict:
    value=_safe_transition_pickle(raw)
    if len(value)!=11 or any(not isinstance(x,str) for x in value[:3]):
        raise ValueError("Expected an 11-field ARPGE transition")
    name,parent,product=value[:3]
    # CRITICAL legacy order: fields 3/4 and 7/8 are DAUGHTER/PARENT,
    # despite the phase-name order being parent then daughter!\n    # NiTi_AQ proves this: (010)_B19prime//(110)_B2 and\n    # [101]_B19prime//[-1 1 1]_B2. Burgers confirms (0001)_hcp//(110)_bcc.
    pairs=[(_vector3(value[i],"daughter hkl/uvw"),
            _vector3(value[i+1],"parent hkl/uvw"))
           for i in (3,5,7,9)]
    if sum(a is not None and b is not None for a,b in pairs)<2:
        raise ValueError("At least two complete plane/direction pairs required")
    if any((a is None)!=(b is None) for a,b in pairs):
        raise ValueError("Incomplete parent/product matching index pairs")
    return {
        "name":name,"parent":parent,"product":product,
        "primary_parent_plane_hkl":pairs[0][1],
        "primary_product_plane_hkl":pairs[0][0],
        "secondary_parent_plane_hkl":pairs[1][1],
        "secondary_product_plane_hkl":pairs[1][0],
        "primary_parent_direction_uvw":pairs[2][1],
        "primary_product_direction_uvw":pairs[2][0],
        "secondary_parent_direction_uvw":pairs[3][1],
        "secondary_product_direction_uvw":pairs[3][0],
        "source_index_order":"daughter_then_parent",
        "source":"2021 user archive, static protocol-0 primitives; not validated OR",
    }


def decode_cryst_metrics_without_unpickling(raw:bytes)->dict:
    if len(raw)>256000:
        raise ValueError("Oversized original CRYST record")
    # Parse opcodes to check structure but never evaluate GLOBAL/REDUCE/BUILD.
    entries=list(pickletools.genops(raw.replace(b"\r\n",b"\n")))
    if len(entries)<12 or entries[0][0].name!="MARK":
        raise ValueError("Not an original CRYST metric record")
    expected=["STRING","PUT","FLOAT","FLOAT","FLOAT",
              "FLOAT","FLOAT","FLOAT","STRING","PUT","STRING","PUT","STRING"]
    # Some legacy cell metrics use I (integer) rather than F.
    names=[a.name for a,_,_ in entries]
    if names[1:3]!=["STRING","PUT"] or names[9:11]!=["STRING","PUT"]:
        raise ValueError("Unexpected crystal prefix")
    if any(names[i] not in ("FLOAT","INT") for i in range(3,9)):
        raise ValueError("Missing primitive six-parameter crystal cell")
    name=entries[1][1];values=[float(entries[i][1]) for i in range(3,9)]
    if (not isinstance(name,str) or not name or any(not math.isfinite(x) for x in values)
        or any(x<=0 for x in values[:3]) or
        any(not 0<x<math.pi for x in values[3:])):
        raise ValueError("Invalid crystal phase cell metrics")
    system=entries[9][1] if names[9]=="STRING" else ""
    centering=entries[11][1] if names[11]=="STRING" else ""
    point_group=entries[13][1] if len(entries)>13 and names[13]=="STRING" else ""
    # Legacy exact stored system fields may vary; always expose a primitive,
    # explicitly audited metric instead of silently constructing atomic data.
    return {
        "name":name,"crystal_system":system,"centering":centering,"point_group_legacy":point_group,"a_angstrom":values[0],"b_angstrom":values[1],
        "c_angstrom":values[2],
        "alpha_deg":math.degrees(values[3]),
        "beta_deg":math.degrees(values[4]),
        "gamma_deg":math.degrees(values[5]),
        "source":"untrusted CRYST prefix, no unpickle, no atomic-site certification",
        "atom_positions_validated":False,
    }


def inspect_arpge_archive(filename:str,*, max_members:int=1200,
                          max_uncompressed:int=240*1024*1024)->dict:
    """Safely inspect original ZIP on disk; never extract/execute any member."""
    if not filename.lower().endswith(".zip"):
        raise ValueError("Provide original ZIP")
    transitions=[];crystals=[];errors=[];summed=0
    with ZipFile(filename) as archive:
        if len(archive.infolist())>max_members:
            raise ValueError("Archive exceeds file count limit")
        for member in archive.infolist():
            path=PurePosixPath(member.filename)
            if (path.is_absolute() or ".." in path.parts or
                "\\" in member.filename):
                raise ValueError("Unsafe archive member path")
            summed+=member.file_size
            if summed>max_uncompressed:
                raise ValueError("Archive size exceeds permitted limit")
            if member.is_dir():continue
            if path.suffix.lower() not in (".trans",".cryst"):continue
            if member.file_size>256000:raise ValueError("Oversized archive record")
            raw=archive.read(member)
            try:
                if path.suffix.lower()==".trans":
                    record=decode_transition(raw);transitions.append(record)
                else:
                    record=decode_cryst_metrics_without_unpickling(raw)
                    crystals.append(record)
                record["original_path"]=member.filename
                record["sha256"]=hashlib.sha256(raw).hexdigest()
            except (ValueError,TypeError,IndexError) as exc:
                errors.append({"path":member.filename,"error":str(exc)})
    return {
        "schema":"cualni.arpge2021-source-inventory/1.0",
        "transition_count":len(transitions),"crystal_metric_count":len(crystals),
        "transitions":transitions,"crystal_metrics":crystals,
        "rejected_records":errors,
        "legacy_source_executed":False,"pickle_unserialized":False,
        "source_original_licence_verified":False,
        "status":"read-only primitive extraction; numerical GenOVa parity NOT claimed",
    }


def audit_legacy_orientation_presets(inventory:dict,
                                    *, geometric_tolerance_deg:float=2.0)->dict:
    """Evaluate ALL decoded 2021 presets without assuming their OR is exact.

    A 2021 transformation may contain two planes, two directions, or one
    plane and one direction. Direct and reciprocal constraints are converted
    correctly. A proper Kabsch rotation is fitted and residuals reported;
    inconsistent/ambiguous presets are not silently accepted.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .lattice import Lattice
    from .niti_literature_or import direction_cartesian, plane_normal_cartesian
    if not np.isfinite(geometric_tolerance_deg) or not 0<geometric_tolerance_deg<=15:
        raise ValueError("Invalid phase geometry tolerance")
    phases={c["name"]:c for c in inventory["crystal_metrics"]}
    out=[]
    for trans in inventory["transitions"]:
        name=trans["name"];pa=phases.get(trans["parent"])
        ma=phases.get(trans["product"])
        if pa is None or ma is None:
            out.append({"name":name,"status":"missing original phase definition"})
            continue
        def phase(rec):
            return Lattice(
                rec["a_angstrom"],rec["b_angstrom"],rec["c_angstrom"],
                rec["alpha_deg"],rec["beta_deg"],rec["gamma_deg"],
            )
        try:
            A=phase(pa);M=phase(ma)
            expected=[];observed=[]
            for label in ("primary","secondary"):
                hkA=trans[f"{label}_parent_plane_hkl"]
                hkM=trans[f"{label}_product_plane_hkl"]
                if hkA is not None:
                    expected.append(plane_normal_cartesian(A,hkA))
                    observed.append(plane_normal_cartesian(M,hkM))
                dA=trans[f"{label}_parent_direction_uvw"]
                dM=trans[f"{label}_product_direction_uvw"]
                if dA is not None:
                    expected.append(direction_cartesian(A,dA))
                    observed.append(direction_cartesian(M,dM))
            expected=np.asarray(expected);observed=np.asarray(observed)
            if len(expected)<2 or len(observed)<2:
                raise ValueError("Fewer than two geometric constraints")
            singular=np.linalg.svd(expected.T@observed,compute_uv=False)
            if singular[1]<=1e-7:
                raise ValueError("Collinear constraints do not define unique OR")
            rotation,_=Rotation.align_vectors(expected,observed)
            Q=rotation.as_matrix()
            angles=np.rad2deg(np.arccos(np.clip(
                np.sum(expected*(Q@observed.T).T,axis=1),-1.,1.,
            )))
            max_error=float(np.max(angles))
            out.append({
                "name":name,"parent":trans["parent"],"daughter":trans["product"],
                "n_constraints":len(expected),
                "max_direction_plane_residual_deg":max_error,
                "Q_parent_from_daughter":Q.tolist(),
                "proper_rotation":True,
                "status":("geometric candidate" if max_error<=geometric_tolerance_deg
                          else "inconsistent archived constraints"),
                "physical_or_certified":False,
            })
        except (ValueError,ArithmeticError,TypeError) as exc:
            out.append({"name":name,"status":"rejected","reason":str(exc)})
    from collections import Counter
    return {
        "schema":"cualni.arpge2021-57-preset-geometry-audit/0.1",
        "tested_transitions":len(out),
        "status_counts":dict(Counter(x["status"] for x in out)),
        "results":out,
        "note":"Geometric Q and metadata test only; no original GUI/double-coset parity or diffraction certification",
    }

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
    # The source uses fields 3/4 for one parent/product plane pair,
    # 7/8 for an in-plane parent/product direction pair. Optional
    # second plane/direction constraints occupy fields 5/6, 9/10.
    pairs=[(_vector3(value[i],"parent hkl/uvw"),
            _vector3(value[i+1],"daughter hkl/uvw"))
           for i in (3,5,7,9)]
    if not pairs[0][0] or not pairs[0][1] or not pairs[2][0] or not pairs[2][1]:
        raise ValueError("Missing primary plane/direction relationship")
    if any((a is None)!=(b is None) for a,b in pairs):
        raise ValueError("Incomplete parent/product matching index pairs")
    return {
        "name":name,"parent":parent,"product":product,
        "primary_parent_plane_hkl":pairs[0][0],
        "primary_product_plane_hkl":pairs[0][1],
        "secondary_parent_plane_hkl":pairs[1][0],
        "secondary_product_plane_hkl":pairs[1][1],
        "primary_parent_direction_uvw":pairs[2][0],
        "primary_product_direction_uvw":pairs[2][1],
        "secondary_parent_direction_uvw":pairs[3][0],
        "secondary_product_direction_uvw":pairs[3][1],
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
    if names[1:3]!=["STRING","PUT"] or names[8:10]!=["STRING","PUT"]:
        raise ValueError("Unexpected crystal prefix")
    if any(names[i] not in ("FLOAT","INT") for i in range(3,9)):
        raise ValueError("Missing primitive six-parameter crystal cell")
    name=entries[1][1];values=[float(entries[i][1]) for i in range(3,9)]
    if (not isinstance(name,str) or not name or any(not math.isfinite(x) for x in values)
        or any(x<=0 for x in values[:3]) or
        any(not 0<x<math.pi for x in values[3:])):
        raise ValueError("Invalid crystal phase cell metrics")
    system=entries[9][1] if names[9]=="STRING" else ""
    # Legacy exact stored system fields may vary; always expose a primitive,
    # explicitly audited metric instead of silently constructing atomic data.
    return {
        "name":name,"a_angstrom":values[0],"b_angstrom":values[1],
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

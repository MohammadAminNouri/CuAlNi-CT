from __future__ import annotations

import importlib
import json
from pathlib import Path
import pkgutil
import numpy as np

import validation.engineered as engineered
from validation.engineered.jsonutil import to_jsonable
from validation.engineered.mutation_score import mutation_score


def test_every_engineered_validation_module_imports_cleanly():
    failures=[]
    for item in pkgutil.iter_modules(engineered.__path__,engineered.__name__+"."):
        try:
            importlib.import_module(item.name)
        except Exception as exc:
            failures.append((item.name,repr(exc)))
    assert not failures,failures


def test_json_conversion_handles_numpy_complex_and_mutation_report():
    payload={
        "np_bool":np.bool_(True),
        "np_float":np.float64(1.25),
        "np_int":np.int64(7),
        "np_complex":np.complex128(1.5-2.25j),
        "py_complex":3.0+4.0j,
        "complex_array":np.array([1+2j,3-4j]),
        "mutations":mutation_score(),
    }
    encoded=json.dumps(to_jsonable(payload),sort_keys=True)
    restored=json.loads(encoded)
    assert restored["np_bool"] is True
    assert restored["np_int"]==7
    assert restored["np_complex"]=={"real":1.5,"imag":-2.25}
    assert restored["py_complex"]=={"real":3.0,"imag":4.0}
    assert isinstance(restored["mutations"],dict)


def test_runner_has_repository_root_bootstrap_before_validation_import():
    text=Path("tools/run_engineered_falsification_v2.py").read_text(encoding="utf-8")
    assert text.index("sys.path.insert") < text.index(
        "from validation.engineered.harness import run"
    )

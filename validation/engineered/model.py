from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any
import numpy as np


class TruthClass(str, Enum):
    FIRST_ORDER_COMPATIBLE = "first_order_compatible"
    SECOND_ORDER_DEGENERATE = "second_order_degenerate"
    THIRD_ORDER_IDENTITY = "third_order_identity"
    ZERO_EIGENVALUE_SAME_SIGN = "zero_eigenvalue_same_sign"
    INCOMPATIBLE = "incompatible"


class FailureClass(str, Enum):
    ORACLE_MISMATCH = "oracle_mismatch"
    THEORY_EQUIVALENCE_VIOLATION = "theory_equivalence_violation"
    REPRESENTATION_COVARIANCE = "representation_covariance"
    RECIPROCAL_DIRECT_DUALITY = "reciprocal_direct_duality"
    NUMERICAL_BACKWARD_ERROR = "numerical_backward_error"
    CONDITIONING_INSTABILITY = "conditioning_instability"
    SYMMETRY_TOPOLOGY = "symmetry_topology"
    PTMC_ROOT_STRUCTURE = "ptmc_root_structure"
    COFACTOR_FORMULA = "cofactor_formula"
    END_TO_END_INPUT_PIPELINE = "end_to_end_input_pipeline"
    INVALID_INPUT_ACCEPTED = "invalid_input_accepted"
    MUTANT_SURVIVED = "mutant_survived"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PhysicalTruth:
    case_id: str
    lambdas: np.ndarray
    U_physical: np.ndarray
    R_physical: np.ndarray
    F_physical: np.ndarray
    truth_class: TruthClass
    exact_compatible: bool
    expected_degeneracy_order: int
    notes: str = ""


@dataclass(frozen=True)
class CrystalEncoding:
    case_id: str
    B_parent: np.ndarray
    B_product: np.ndarray
    C_m_from_a: np.ndarray
    M_parent: np.ndarray
    M_product: np.ndarray
    truth: PhysicalTruth
    provenance: tuple[str, ...] = ()

    @property
    def condition_parent_basis(self) -> float:
        return float(np.linalg.cond(self.B_parent))

    @property
    def condition_product_basis(self) -> float:
        return float(np.linalg.cond(self.B_product))

    @property
    def condition_correspondence(self) -> float:
        return float(np.linalg.cond(self.C_m_from_a))

    @property
    def condition_metric_pair(self) -> float:
        return max(float(np.linalg.cond(self.M_parent)), float(np.linalg.cond(self.M_product)))


@dataclass(frozen=True)
class ErrorBudget:
    machine_epsilon: float
    condition_measure: float
    multiplier: float
    floor: float
    ceiling: float
    allowed: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ContractResult:
    name: str
    passed: bool
    residual: float | None
    allowed: float | None
    failure_class: FailureClass
    details: dict[str, Any]


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    contracts: tuple[ContractResult, ...]
    diagnostics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.contracts)

    @property
    def failed_contracts(self) -> tuple[ContractResult, ...]:
        return tuple(item for item in self.contracts if not item.passed)


def array_json(a: np.ndarray) -> list:
    return np.asarray(a, dtype=float).tolist()


def encoding_to_dict(case: CrystalEncoding) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "B_parent": array_json(case.B_parent),
        "B_product": array_json(case.B_product),
        "C_m_from_a": array_json(case.C_m_from_a),
        "M_parent": array_json(case.M_parent),
        "M_product": array_json(case.M_product),
        "conditions": {
            "B_parent": case.condition_parent_basis,
            "B_product": case.condition_product_basis,
            "C": case.condition_correspondence,
            "metrics": case.condition_metric_pair,
        },
        "truth": {
            "lambdas": array_json(case.truth.lambdas),
            "U_physical": array_json(case.truth.U_physical),
            "R_physical": array_json(case.truth.R_physical),
            "F_physical": array_json(case.truth.F_physical),
            "truth_class": case.truth.truth_class.value,
            "exact_compatible": case.truth.exact_compatible,
            "expected_degeneracy_order": case.truth.expected_degeneracy_order,
            "notes": case.truth.notes,
        },
        "provenance": list(case.provenance),
    }

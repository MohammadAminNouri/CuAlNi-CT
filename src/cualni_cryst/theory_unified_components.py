from __future__ import annotations

"""Cacheable staged orchestration for unified CT/Ball--James/PTMC comparison.

The theory implementations remain in their original backends.  This module
only exposes the already-existing independent stages as immutable components so
an interactive application can cache CT, Ball--James and PTMC separately and
reassemble the same :class:`UnifiedTheoryReport` without recomputing unchanged
stages.
"""

from dataclasses import dataclass
from typing import Any

from .ball_james_adapter import BallJamesReport
from .ct import CTAMResult
from .ptmc_adapter import PTMCReport, PTMCSlipSystemInput, PTMCTwinPairInput, PTMCTwinPlaneInput
from .theory_unified import (
    ComparisonRow,
    ExperimentalObservation,
    PTMCMode,
    TheoryComparisonAdapter,
    UnifiedTheoryReport,
)


@dataclass(frozen=True)
class CTTheoryComponent:
    report: CTAMResult
    rows: tuple[ComparisonRow, ...]
    include_closing_gap: bool
    include_supercompatibility: bool
    natural_orientation_supplied: bool


@dataclass(frozen=True)
class BallJamesTheoryComponent:
    report: BallJamesReport
    rows: tuple[ComparisonRow, ...]
    fraction_samples: int


@dataclass(frozen=True)
class PTMCTheoryComponent:
    report: PTMCReport | None
    rows: tuple[ComparisonRow, ...]
    mode: PTMCMode


class StagedTheoryComparisonAdapter(TheoryComparisonAdapter):
    """Public staged façade over the existing unified-theory orchestrator."""

    def ct_component(
        self,
        *,
        natural_orientation: Any | None = None,
        include_closing_gap: bool = True,
        include_supercompatibility: bool = True,
    ) -> CTTheoryComponent:
        report, rows = self._ct_rows(
            natural_orientation=natural_orientation,
            include_closing_gap=include_closing_gap,
            include_supercompatibility=include_supercompatibility,
        )
        return CTTheoryComponent(
            report=report,
            rows=tuple(rows),
            include_closing_gap=bool(include_closing_gap),
            include_supercompatibility=bool(include_supercompatibility),
            natural_orientation_supplied=natural_orientation is not None,
        )

    def ball_james_component(
        self,
        *,
        fraction_samples: int = 101,
    ) -> BallJamesTheoryComponent:
        if fraction_samples < 2:
            raise ValueError("fraction_samples must be >= 2")
        report, rows = self._ball_james_rows(fraction_samples=fraction_samples)
        return BallJamesTheoryComponent(
            report=report,
            rows=tuple(rows),
            fraction_samples=int(fraction_samples),
        )

    def ptmc_component(
        self,
        *,
        mode: PTMCMode | str = PTMCMode.ALL_TWINNING,
        request: PTMCSlipSystemInput | PTMCTwinPlaneInput | PTMCTwinPairInput | None = None,
        base_variant_index: int | None = None,
        dilatational_factor: float = 1.0,
    ) -> PTMCTheoryComponent:
        mode_value = PTMCMode(mode)
        if dilatational_factor <= 0.0:
            raise ValueError("dilatational_factor must be positive")
        if base_variant_index is not None and mode_value is not PTMCMode.ALL_TWINNING:
            raise ValueError(
                "base_variant_index is only valid for mode='all_twinning'; other PTMC modes carry selection in request"
            )
        report = self._ptmc_report(
            mode=mode_value,
            request=request,
            base_variant_index=base_variant_index,
            dilatational_factor=dilatational_factor,
        )
        rows = tuple() if report is None else tuple(self._ptmc_rows(report))
        return PTMCTheoryComponent(report=report, rows=rows, mode=mode_value)

    def assemble(
        self,
        ct: CTTheoryComponent,
        ball_james: BallJamesTheoryComponent,
        ptmc: PTMCTheoryComponent,
        *,
        experiments: tuple[ExperimentalObservation, ...] = (),
    ) -> UnifiedTheoryReport:
        rows = tuple(ct.rows) + tuple(ball_james.rows) + tuple(ptmc.rows) + tuple(
            self._experiment_rows(experiments)
        )
        ids = [row.row_id for row in rows]
        if len(ids) != len(set(ids)):
            raise AssertionError("Unified comparison generated duplicate row_id values")

        warnings: list[str] = []
        if ptmc.report is None:
            warnings.append("PTMC was explicitly disabled for this comparison.")
        if ct.include_closing_gap and not ct.natural_orientation_supplied:
            warnings.append(
                "CT closing-gap candidates are all retained because no natural OR was supplied; no preferred CT OR is selected."
            )
        if not ct.report.exact_habit_planes:
            warnings.append(
                "CT has no exact finite A/M habit-plane branch for these inputs; approximate CMC diagnostics, when admissible, remain explicitly marked approximate."
            )

        return UnifiedTheoryReport(
            transformation_id=self.transformation_id,
            parent_phase_id=self.parent.phase_id,
            product_phase_id=self.product.phase_id,
            rows=rows,
            ct_report=ct.report,
            ball_james_report=ball_james.report,
            ptmc_report=ptmc.report,
            warnings=tuple(warnings),
            notes=(
                "All three theory branches consume the same ProjectState transformation.",
                "PTMC LIS data and a CT natural OR are theory-specific auxiliary hypotheses, not shared crystallographic inputs.",
                "Residuals retain names and native units; this report deliberately does not compute an overall winner score.",
                "Ball-James rank-one rotations are not silently relabelled as physical parent/product OR matrices.",
                "Missing observables are represented as None / N/A.",
            ),
        )

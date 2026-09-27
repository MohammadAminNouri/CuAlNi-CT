from __future__ import annotations

"""User-safe application errors for the public crystallography workbench."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    title: str
    message: str
    hint: str = ""
    technical_detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "title": self.title,
            "message": self.message,
            "hint": self.hint,
            "technical_detail": self.technical_detail,
        }


class ApplicationError(Exception):
    def __init__(self, info: ErrorInfo):
        super().__init__(info.message)
        self.info = info


def input_error(
    message: str, *, detail: str = "", hint: str = ""
) -> ApplicationError:
    return ApplicationError(
        ErrorInfo(
            code="INVALID_INPUT",
            title="Invalid input",
            message=message,
            hint=hint or "Correct the indicated value and try again.",
            technical_detail=detail,
        )
    )


def scientific_domain_error(
    message: str, *, detail: str = "", hint: str = ""
) -> ApplicationError:
    return ApplicationError(
        ErrorInfo(
            code="SCIENTIFIC_DOMAIN_ERROR",
            title="Crystallographic state is not admissible",
            message=message,
            hint=hint
            or (
                "No input was repaired and no scientific tolerance was relaxed. "
                "Correct the stated inconsistency and calculate again."
            ),
            technical_detail=detail,
        )
    )


def stale_state_error() -> ApplicationError:
    return ApplicationError(
        ErrorInfo(
            code="STALE_RESULTS",
            title="Recalculation required",
            message=(
                "Setup has changed since the last successful calculation. "
                "Results from the previous state are not shown as current."
            ),
            hint="Return to Setup and calculate the transformation again.",
        )
    )


def internal_error(message: str, *, detail: str = "") -> ApplicationError:
    return ApplicationError(
        ErrorInfo(
            code="INTERNAL_APPLICATION_ERROR",
            title="Application-layer failure",
            message=message,
            hint=(
                "The input was not silently modified. Preserve the project and "
                "report the technical detail to the maintainer."
            ),
            technical_detail=detail,
        )
    )

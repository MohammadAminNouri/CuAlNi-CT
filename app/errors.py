from __future__ import annotations

"""User-safe application errors for the public crystallography workbench.

The scientific package keeps raising precise Python exceptions.  The public UI
must preserve their meaning without exposing a traceback to a normal user.
This module is deliberately outside :mod:`cualni_cryst`; it does not alter or
reinterpret any scientific result.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorInfo:
    """Stable error payload suitable for GUI display and JSON export."""

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
    """Expected application-boundary failure with a user-safe explanation."""

    def __init__(self, info: ErrorInfo):
        super().__init__(info.message)
        self.info = info


def input_error(message: str, *, detail: str = "") -> ApplicationError:
    return ApplicationError(
        ErrorInfo(
            code="INVALID_INPUT",
            title="Input could not be interpreted",
            message=message,
            hint=(
                "Check lattice parameters, point-group/cell consistency, and the "
                "3×3 correspondence entries. Fractions such as 1/2 are accepted."
            ),
            technical_detail=detail,
        )
    )


def scientific_domain_error(message: str, *, detail: str = "") -> ApplicationError:
    return ApplicationError(
        ErrorInfo(
            code="SCIENTIFIC_DOMAIN_ERROR",
            title="The supplied crystallographic state is not admissible",
            message=message,
            hint=(
                "The application has not altered the input or relaxed a scientific "
                "criterion. Correct the stated crystallographic inconsistency and "
                "calculate again."
            ),
            technical_detail=detail,
        )
    )


def internal_error(message: str, *, detail: str = "") -> ApplicationError:
    return ApplicationError(
        ErrorInfo(
            code="INTERNAL_APPLICATION_ERROR",
            title="Application-layer failure",
            message=message,
            hint=(
                "Your input has not been silently modified. Preserve/export the "
                "project and report the technical detail to the maintainer."
            ),
            technical_detail=detail,
        )
    )

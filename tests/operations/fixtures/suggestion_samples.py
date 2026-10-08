"""Fixture constructions for the AST scan. Never imported at runtime."""

from organelleverse.core.result import OperationSuggestion

GOOD_ID = "annotation.write"


def literal_suggestion() -> OperationSuggestion:
    return OperationSuggestion(operation_id="annotation.write", reason_code="demo")


def constant_suggestion() -> OperationSuggestion:
    return OperationSuggestion(operation_id=GOOD_ID, reason_code="demo")


def unknown_suggestion() -> OperationSuggestion:
    return OperationSuggestion(operation_id="io.does_not_exist", reason_code="demo")


def dynamic_suggestion(name: str) -> OperationSuggestion:
    return OperationSuggestion(operation_id=f"io.{name}", reason_code="demo")

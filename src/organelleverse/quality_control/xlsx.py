"""Dependency-free deterministic XLSX tables for QC reports."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape


def write_xlsx_tables(tables: dict[str, list[dict[str, Any]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_items = list(tables.items()) or [("data", [])]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _zip_write(archive, "[Content_Types].xml", _content_types(len(sheet_items)))
        _zip_write(archive, "_rels/.rels", _root_rels())
        _zip_write(
            archive,
            "xl/workbook.xml",
            _workbook_xml([name for name, _ in sheet_items]),
        )
        _zip_write(
            archive,
            "xl/_rels/workbook.xml.rels",
            _workbook_rels(len(sheet_items)),
        )
        for index, (_, rows) in enumerate(sheet_items, 1):
            _zip_write(archive, f"xl/worksheets/sheet{index}.xml", _worksheet_xml(rows))


def _zip_write(archive: zipfile.ZipFile, name: str, data: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, data.encode())


def _headers(rows: list[dict[str, Any]]) -> list[str]:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    return headers or ["value"]


def _cell(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _worksheet_xml(rows: list[dict[str, Any]]) -> str:
    headers = _headers(rows)
    body = [_xlsx_row(1, headers)]
    for row_index, row in enumerate(rows, 2):
        body.append(_xlsx_row(row_index, [_cell(row.get(header, "")) for header in headers]))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>" + "".join(body) + "</sheetData></worksheet>"
    )


def _xlsx_row(row_index: int, values: list[Any]) -> str:
    cells = [
        _xlsx_cell(f"{_excel_col(column_index)}{row_index}", value)
        for column_index, value in enumerate(values, 1)
    ]
    return f'<row r="{row_index}">{"".join(cells)}</row>'


def _xlsx_cell(reference: str, value: Any) -> str:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return f'<c r="{reference}"><v>{value}</v></c>'
    text = xml_escape("" if value is None else str(value))
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _content_types(sheet_count: int) -> str:
    worksheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + worksheets
        + "</Types>"
    )


def _root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{xml_escape(_sheet_name(name))}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def _workbook_rels(sheet_count: int) -> str:
    relationships = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + relationships
        + "</Relationships>"
    )


def _excel_col(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _sheet_name(name: str) -> str:
    safe = "".join(character if character not in "[]:*?/\\'" else "_" for character in name)
    return (safe or "data")[:31]


__all__ = ["write_xlsx_tables"]

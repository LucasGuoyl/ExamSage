"""Publish JUnit failures as GitHub Actions check annotations."""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


SOURCE_LOCATION = re.compile(r"(?m)^([^:\n]+\.py):(\d+):")
MAX_ANNOTATIONS = 20


def _escape(value: str, *, property_value: bool = False) -> str:
    escaped = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_value:
        escaped = escaped.replace(":", "%3A").replace(",", "%2C")
    return escaped


def _failure_title(testcase: ET.Element) -> str:
    classname = testcase.get("classname", "")
    name = testcase.get("name", "pytest failure")
    return f"{classname}::{name}" if classname else name


def _failure_location(failure: ET.Element) -> tuple[str, str]:
    match = SOURCE_LOCATION.search(failure.text or "")
    if match is None:
        return "pytest-results.xml", "1"
    return match.group(1), match.group(2)


def annotate(report: Path) -> int:
    root = ET.parse(report).getroot()
    emitted = 0
    for testcase in root.iter("testcase"):
        failure = testcase.find("failure")
        if failure is None:
            failure = testcase.find("error")
        if failure is None:
            continue
        path, line = _failure_location(failure)
        title = _failure_title(testcase)
        message = failure.text or failure.get("message") or "pytest failed"
        print(
            f"::error file={_escape(path, property_value=True)},"
            f"line={line},title={_escape(title, property_value=True)}::"
            f"{_escape(message)}"
        )
        emitted += 1
        if emitted >= MAX_ANNOTATIONS:
            break
    return emitted


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: annotate_pytest_failures.py REPORT.xml", file=sys.stderr)
        return 2
    try:
        emitted = annotate(Path(sys.argv[1]))
    except (OSError, ET.ParseError) as error:
        print(f"Unable to parse pytest report: {error}", file=sys.stderr)
        return 1
    print(f"Published {emitted} pytest failure annotation(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

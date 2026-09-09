"""Input normalization for ASCII maps supplied to the spatial-memory probe."""

from __future__ import annotations


def map_rows(map_text: str) -> tuple[str, ...]:
    """Normalize a pasted ASCII map while retaining every map-space character.

    Dataset maps are rectangular, but terminal/editor copy-paste commonly adds
    outer blank lines, MAP tags, or drops *trailing* spaces. Outer decoration
    has no spatial meaning; omitted trailing spaces do, so restore them by
    right-padding to the longest retained row. An empty line in the map body
    remains an error because it is almost certainly a paste corruption.
    """

    rows = list(map_text.removeprefix("\ufeff").splitlines())
    while rows and not rows[0].strip():
        rows.pop(0)
    while rows and not rows[-1].strip():
        rows.pop()
    if len(rows) >= 2 and rows[0].strip() == "<MAP>" and rows[-1].strip() == "</MAP>":
        rows = rows[1:-1]
        while rows and not rows[0].strip():
            rows.pop(0)
        while rows and not rows[-1].strip():
            rows.pop()
    if not rows:
        raise ValueError("map contains no map rows")
    empty_rows = [index + 1 for index, row in enumerate(rows) if not row]
    if empty_rows:
        raise ValueError(f"map has empty row(s) {empty_rows}; remove blank lines inside the map")
    width = max(map(len, rows))
    return tuple(row.ljust(width) for row in rows)

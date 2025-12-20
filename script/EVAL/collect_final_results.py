from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_PIPE_ROW_RE = re.compile(r"^\|(?P<cells>.+)\|\s*$")


def _parse_eval_md_table(md_text: str) -> List[Dict[str, str]]:
    """Parse the first GitHub-style pipe table in EVAL.md.

    Expected columns: model | code | Zelpha | seed | robust accuracy (%)
    Returns list of dicts with those keys (as they appear in header).
    """

    lines = [ln.rstrip("\n") for ln in md_text.splitlines()]

    header_cells: Optional[List[str]] = None
    rows: List[Dict[str, str]] = []
    in_table = False

    for ln in lines:
        m = _PIPE_ROW_RE.match(ln.strip())
        if not m:
            if in_table:
                break
            continue

        cells = [c.strip() for c in m.group("cells").split("|")]

        # Skip separator row like |---|---|
        if all(set(c) <= {"-", ":"} and c for c in cells):
            in_table = True
            continue

        if header_cells is None:
            header_cells = cells
            in_table = True
            continue

        if not in_table:
            continue

        if len(cells) != len(header_cells):
            # Be forgiving: skip malformed rows.
            continue

        row = {header_cells[i]: cells[i] for i in range(len(cells))}
        # Ignore HTML comment rows or empty code
        if row.get("code", "").startswith("<!--"):
            continue
        if not row.get("code"):
            continue
        rows.append(row)

    if header_cells is None:
        raise ValueError("No pipe table header found in EVAL.md")

    return rows


_KV_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z0-9_.-]+)\s*:\s*(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def _extract_last_final_results_block(log_text: str) -> Dict[str, float]:
    """Extract the last 'Final Results:' key-value block from a log."""

    marker = "Final Results:"
    idx = log_text.rfind(marker)
    if idx == -1:
        return {}

    after = log_text[idx + len(marker) :]
    # Start from the next line
    lines = after.splitlines()

    results: Dict[str, float] = {}
    for ln in lines:
        if not ln.strip():
            # blank line ends the block
            if results:
                break
            continue

        m = _KV_RE.match(ln)
        if not m:
            # non key-value line ends the block once we've started collecting
            if results:
                break
            continue

        key = m.group("key")
        val = float(m.group("val"))
        results[key] = val

    return results


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def build_rows(eval_rows: List[Dict[str, str]], logs_dir: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    """Return (rows, dynamic_final_result_keys)."""

    dynamic_keys: List[str] = []
    dynamic_key_set = set()

    out_rows: List[Dict[str, object]] = []

    for r in eval_rows:
        code = r.get("code", "").strip()
        log_path = logs_dir / f"{code}.log"

        final_results: Dict[str, float] = {}
        if log_path.exists():
            final_results = _extract_last_final_results_block(_read_text(log_path))

        for k in final_results.keys():
            if k not in dynamic_key_set:
                dynamic_key_set.add(k)
                dynamic_keys.append(k)

        row: Dict[str, object] = {}
        # Keep original EVAL columns as-is
        for k, v in r.items():
            row[k] = v

        row["log_path"] = str(log_path.as_posix())
        row["has_log"] = log_path.exists()

        # Also keep robust_acc duplicated if present (useful for sanity checking vs EVAL.md)
        if "robust_acc" in final_results:
            row["log_robust_acc"] = final_results.get("robust_acc")

        for k, v in final_results.items():
            row[k] = v

        out_rows.append(row)

    return out_rows, dynamic_keys


def write_csv(rows: List[Dict[str, object]], out_path: Path, dynamic_keys: List[str]) -> None:
    # Determine base columns from first row for stable ordering
    base_cols: List[str] = []
    if rows:
        for k in rows[0].keys():
            if k not in dynamic_keys:
                base_cols.append(k)

    # Ensure these metadata fields are early
    preferred_front = ["model", "code", "Zelpha", "seed", "robust accuracy (%)", "log_robust_acc", "has_log", "log_path"]
    cols: List[str] = []
    for k in preferred_front:
        if any(k in r for r in rows) and k not in cols:
            cols.append(k)

    for k in base_cols:
        if k not in cols and k not in dynamic_keys:
            cols.append(k)

    # Append dynamic final results keys at the end
    cols.extend(dynamic_keys)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect 'Final Results:' blocks from logs and align with EVAL.md codes.")
    ap.add_argument("--eval", dest="eval_md", type=Path, default=Path("script/EVAL/EVAL.md"))
    ap.add_argument("--logs", dest="logs_dir", type=Path, default=Path("logs"))
    ap.add_argument("--out", dest="out_csv", type=Path, default=Path("script/EVAL/final_results.csv"))
    args = ap.parse_args()

    eval_rows = _parse_eval_md_table(_read_text(args.eval_md))
    rows, dynamic_keys = build_rows(eval_rows, args.logs_dir)
    write_csv(rows, args.out_csv, dynamic_keys)

    missing = sum(1 for r in rows if not bool(r.get("has_log")))
    print(f"Wrote {args.out_csv} with {len(rows)} rows. Missing logs: {missing}.")
    print(f"Dynamic Final Results keys: {len(dynamic_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

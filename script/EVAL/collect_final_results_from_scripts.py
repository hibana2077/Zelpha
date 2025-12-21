from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ScriptMeta:
    code: str
    model: str
    seed: int
    zelpha: str  # 'Y' or 'N'
    num_prototypes: Optional[int] = None
    beta: Optional[float] = None
    margin: Optional[float] = None
    image_size: Optional[int] = None
    linear_epochs: Optional[int] = None
    finetune_epochs: Optional[int] = None
    no_lipschitz: bool = False
    no_scale_pooling: bool = False


_CODE_RE = re.compile(r"\b(E\d{3}[A-Z])\b")
_ARG_RE = re.compile(r"--(?P<key>[a-zA-Z0-9_\-]+)\s+(?P<val>[^\\\s]+)")


_KV_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z0-9_.-]+)\s*:\s*(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_last_final_results_block(log_text: str) -> Dict[str, float]:
    marker = "Final Results:"
    idx = log_text.rfind(marker)
    if idx == -1:
        return {}

    after = log_text[idx + len(marker) :]
    lines = after.splitlines()

    results: Dict[str, float] = {}
    for ln in lines:
        if not ln.strip():
            if results:
                break
            continue

        m = _KV_RE.match(ln)
        if not m:
            if results:
                break
            continue

        results[m.group("key")] = float(m.group("val"))

    return results


def _parse_bool_flags(sh_text: str) -> Tuple[bool, bool]:
    no_lipschitz = "--no_lipschitz" in sh_text
    no_scale_pooling = "--no_scale_pooling" in sh_text
    return no_lipschitz, no_scale_pooling


def _parse_args_from_sh(sh_text: str) -> Dict[str, str]:
    # Very lightweight parsing: capture --key value pairs.
    args: Dict[str, str] = {}
    for m in _ARG_RE.finditer(sh_text):
        args[m.group("key")] = m.group("val")
    return args


def parse_script(path: Path) -> Optional[ScriptMeta]:
    text = _read_text(path)

    # Determine code (E###X) from filename first, fallback to content.
    code = path.stem
    if not _CODE_RE.search(code):
        m = _CODE_RE.search(text)
        if not m:
            return None
        code = m.group(1)

    args = _parse_args_from_sh(text)
    model = args.get("model_name")
    seed_s = args.get("seed")

    if not model or not seed_s:
        return None

    # Determine zelpha: finetune_epochs == 0 => Vanilla (N), else ZELPHA (Y)
    finetune_epochs_s = args.get("finetune_epochs")
    finetune_epochs = int(finetune_epochs_s) if finetune_epochs_s is not None else None
    zelpha = "N" if finetune_epochs == 0 else "Y"

    no_lipschitz, no_scale_pooling = _parse_bool_flags(text)

    def _maybe_int(key: str) -> Optional[int]:
        v = args.get(key)
        return int(v) if v is not None else None

    def _maybe_float(key: str) -> Optional[float]:
        v = args.get(key)
        return float(v) if v is not None else None

    return ScriptMeta(
        code=code,
        model=str(model),
        seed=int(seed_s),
        zelpha=zelpha,
        num_prototypes=_maybe_int("num_prototypes"),
        beta=_maybe_float("beta"),
        margin=_maybe_float("margin"),
        image_size=_maybe_int("image_size"),
        linear_epochs=_maybe_int("linear_epochs"),
        finetune_epochs=finetune_epochs,
        no_lipschitz=no_lipschitz,
        no_scale_pooling=no_scale_pooling,
    )


def iter_scripts(scripts_dir: Path) -> Iterable[Path]:
    for p in sorted(scripts_dir.glob("E*.sh")):
        yield p


def build_rows(scripts_dir: Path, logs_dir: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    dynamic_keys: List[str] = []
    dynamic_key_set = set()

    rows: List[Dict[str, object]] = []

    for sh_path in iter_scripts(scripts_dir):
        meta = parse_script(sh_path)
        if meta is None:
            continue

        log_path = logs_dir / f"{meta.code}.log"
        final_results: Dict[str, float] = {}
        if log_path.exists():
            final_results = _extract_last_final_results_block(_read_text(log_path))

        for k in final_results.keys():
            if k not in dynamic_key_set:
                dynamic_key_set.add(k)
                dynamic_keys.append(k)

        row: Dict[str, object] = {
            "model": meta.model,
            "code": meta.code,
            "Zelpha": meta.zelpha,
            "seed": meta.seed,
            "num_prototypes": meta.num_prototypes,
            "beta": meta.beta,
            "margin": meta.margin,
            "image_size": meta.image_size,
            "linear_epochs": meta.linear_epochs,
            "finetune_epochs": meta.finetune_epochs,
            "no_lipschitz": meta.no_lipschitz,
            "no_scale_pooling": meta.no_scale_pooling,
            "log_path": str(log_path.as_posix()),
            "has_log": log_path.exists(),
        }

        # Keep legacy column name for compatibility with existing plots/docs
        if "robust_acc" in final_results:
            row["robust accuracy (%)"] = float(final_results["robust_acc"])

        # Add parsed final results keys
        for k, v in final_results.items():
            row[k] = v

        rows.append(row)

    return rows, dynamic_keys


def write_csv(rows: List[Dict[str, object]], out_path: Path, dynamic_keys: List[str]) -> None:
    preferred_front = [
        "model",
        "code",
        "Zelpha",
        "seed",
        "robust accuracy (%)",
        "has_log",
        "log_path",
        "num_prototypes",
        "beta",
        "margin",
        "image_size",
        "linear_epochs",
        "finetune_epochs",
        "no_lipschitz",
        "no_scale_pooling",
    ]

    cols: List[str] = []
    for k in preferred_front:
        if any(k in r for r in rows) and k not in cols:
            cols.append(k)

    # Add any other base cols in stable order
    if rows:
        for k in rows[0].keys():
            if k not in cols and k not in dynamic_keys:
                cols.append(k)

    cols.extend(dynamic_keys)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Collect 'Final Results:' blocks from logs and align with script/EVAL/*.sh metadata."
    )
    ap.add_argument("--scripts", type=Path, default=Path("script/EVAL"), help="Directory containing E*.sh")
    ap.add_argument("--logs", type=Path, default=Path("logs"), help="Logs directory")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs/results/final_results.csv"),
        help="Output CSV path (default: docs/results/final_results.csv)",
    )
    args = ap.parse_args()

    rows, dynamic_keys = build_rows(args.scripts, args.logs)
    write_csv(rows, args.out, dynamic_keys)

    missing = sum(1 for r in rows if not bool(r.get("has_log")))
    print(f"Wrote {args.out} with {len(rows)} rows. Missing logs: {missing}.")
    print(f"Dynamic Final Results keys: {len(dynamic_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

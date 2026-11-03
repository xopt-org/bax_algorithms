#!/usr/bin/env python3
"""Execute Xopt HTTP notebooks with per-cell progress reporting."""

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def run_notebook(path: Path, timeout: int) -> tuple[bool, float]:
    notebook = nbformat.read(path, as_version=4)
    code_cells = [
        (index, cell)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
    ]

    total = len(code_cells)
    position = {
        index: number
        for number, (index, _) in enumerate(code_cells, start=1)
    }
    started = {}

    print(f"\n{'=' * 72}", flush=True)
    print(f"NOTEBOOK: {path.name}", flush=True)
    print(f"CODE CELLS: {total}", flush=True)
    print("=" * 72, flush=True)

    def on_cell_start(cell, cell_index, **kwargs):
        if cell_index not in position:
            return

        source = cell.source.strip().splitlines()
        description = source[0][:75] if source else "(empty cell)"

        started[cell_index] = time.monotonic()

        print(
            f"  [{position[cell_index]:02d}/{total:02d}] "
            f"RUNNING: {description}",
            flush=True,
        )

    def on_cell_executed(cell, cell_index, **kwargs):
        if cell_index not in position:
            return

        elapsed = time.monotonic() - started[cell_index]

        elapsed_text = f"{elapsed:.3f}s" if elapsed < 1.0 else f"{elapsed:.1f}s"

        print(
            f"  [{position[cell_index]:02d}/{total:02d}] "
            f"DONE ({elapsed_text})",
            flush=True,
        )

    start = time.monotonic()

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent.resolve())}},
        on_cell_start=on_cell_start,
        on_cell_executed=on_cell_executed,
    )

    try:
        client.execute()
        success = True
    except Exception as exc:
        success = False
        print(f"\nFAILED: {path.name}", flush=True)
        print(f"{type(exc).__name__}: {exc}", flush=True)

    elapsed = time.monotonic() - start

    print(
        f"\n{'PASS' if success else 'FAIL'}: "
        f"{path.name} ({elapsed:.1f}s)",
        flush=True,
    )

    return success, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    results = []

    for path in args.notebooks:
        success, elapsed = run_notebook(path, args.timeout)
        results.append((path.name, success, elapsed))

        if not success:
            break

    print(f"\n{'=' * 72}")
    print("FINAL RESULTS")
    print("=" * 72)

    for name, success, elapsed in results:
        print(
            f"{'PASS' if success else 'FAIL':4s} "
            f"{elapsed:7.1f}s  {name}"
        )

    return 0 if all(result[1] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

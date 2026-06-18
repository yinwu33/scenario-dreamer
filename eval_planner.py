from __future__ import annotations

import sys

from critical_scene.benchmark import benchmark_artifacts


def _parse_args(argv: list[str]) -> dict:
    options = {"overrides": []}
    for item in argv:
        if "=" not in item:
            options["overrides"].append(item)
            continue
        key, value = item.split("=", 1)
        if key.startswith("+") or "." in key:
            options["overrides"].append(item)
        else:
            options[key] = value
    return options


def main() -> int:
    result = benchmark_artifacts(_parse_args(sys.argv[1:]))
    print(f"out_dir={result['out_dir']}")
    print(f"summary_path={result['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

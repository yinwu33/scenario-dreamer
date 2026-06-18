from __future__ import annotations

import sys

from critical_scene.generate import generate_artifact


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
    result = generate_artifact(_parse_args(sys.argv[1:]))
    print(f"artifact_path={result['artifact_path']}")
    print(f"manifest_path={result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


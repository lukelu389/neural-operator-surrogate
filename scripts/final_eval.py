"""Reserved evaluation command; never opens test or OOD data in this scaffold."""

import sys


def main() -> int:
    """Fail closed until T3.4 implements frozen-hash validation and evaluation."""
    print(
        "Final evaluation is unavailable until T3.4. "
        "Frozen data, protocol, and model hashes must be verified before evaluation. "
        "This scaffold always refuses to evaluate; no test or OOD data were opened.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
